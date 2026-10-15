defmodule Handwired.Learner do
  @moduledoc """
  Immutable, incremental learning with bounded replay, persistent optimizer state,
  exact bounded undo, and circuit-bound snapshots. Supports scalar regression,
  binary outcomes (logits), and pairwise preferences. All program derivatives
  come from Handwired; the loss and optimizer run outside the compiled circuit.

  Each observation performs `:steps` updates over the newest `:replay_capacity`
  observations. Older samples have weight `:decay ** age` (age in observations,
  not wall time). `:anchor` adds L2 distance from the starting parameters.
  Only `:trainable` parameters change; declared parameter ranges remain binding.

  `snapshot/1` returns a versioned, string-keyed map suitable for JSON encoders.
  `restore/2` requires the original circuit, including its starting parameters.
  `dump/1` and `load/2` provide an uncompressed Erlang binary for local storage.
  Snapshots contain feedback and are not encrypted or authenticated. Applications
  own storage, event identity/deduplication, concurrency and access controls.
  """
  alias Handwired.{Circuit, Evaluator, Number, Optimizer}
  alias Handwired.Learner.Snapshot

  defstruct [
    :circuit,
    :initial_parameters,
    :options,
    :last_loss,
    step: 0,
    observations: 0,
    moments: %{},
    replay: [],
    history: []
  ]

  @defaults [
    loss: :mse,
    optimizer: :adam,
    learning_rate: 0.03,
    steps: 1,
    replay_capacity: 64,
    decay: 1.0,
    anchor: 0.0,
    pairwise_scale: 1.0,
    undo_limit: 32
  ]

  @doc "Create a learner. Options are fixed for its lifetime; reset starts from the same prior."
  def new(%Circuit{} = circuit, opts \\ []) do
    unless Keyword.keyword?(opts) and
             length(Keyword.keys(opts)) == length(Enum.uniq(Keyword.keys(opts))),
           do: raise(ArgumentError, "expected unique learner options")

    unknown = Keyword.keys(opts) -- (Keyword.keys(@defaults) ++ [:trainable])
    unless unknown == [], do: raise(ArgumentError, "unknown learner options: #{inspect(unknown)}")
    options = @defaults |> Keyword.merge(opts) |> Map.new()
    names = Map.get(options, :trainable, circuit.parameters |> Map.keys() |> Enum.sort())

    unless is_list(names) and names != [] and names -- Map.keys(circuit.parameters) == [] and
             names == Enum.uniq(names),
           do: raise(ArgumentError, "trainable must name distinct existing parameters")

    unless options.loss in [:mse, :binary_cross_entropy, :pairwise],
      do: raise(ArgumentError, "loss must be :mse, :binary_cross_entropy, or :pairwise")

    unless options.optimizer in [:sgd, :adam],
      do: raise(ArgumentError, "optimizer must be :sgd or :adam")

    for {key, low, high} <- [
          {:steps, 1, 10_000},
          {:replay_capacity, 1, 10_000},
          {:undo_limit, 0, 1_000}
        ] do
      value = options[key]

      unless is_integer(value) and value >= low and value <= high,
        do: raise(ArgumentError, "#{key} must be an integer in #{low}..#{high}")
    end

    options =
      Enum.reduce([:learning_rate, :decay, :anchor, :pairwise_scale], options, fn key, acc ->
        Map.put(acc, key, Number.cast(acc[key], :f64))
      end)

    unless options.learning_rate > 0 and options.decay > 0 and options.decay <= 1 and
             options.anchor >= 0 and options.pairwise_scale > 0,
           do: raise(ArgumentError, "invalid learning rate, decay, anchor, or pairwise scale")

    unless match?({:wire, _, :number}, circuit.output),
      do: raise(ArgumentError, "learner needs a scalar numeric output")

    # Validation of output type and arithmetic happens before the first update.
    %__MODULE__{
      circuit: circuit,
      initial_parameters: circuit.parameters,
      options: Map.put(options, :trainable, names)
    }
  end

  @doc "Learn a numeric target, or a binary 0/1 outcome when using logit cross entropy."
  def observe(%__MODULE__{options: %{loss: loss}} = learner, inputs, target)
      when loss in [:mse, :binary_cross_entropy] do
    learn(learner, %{inputs: inputs, target: target})
  end

  def observe(_, _, _),
    do: raise(ArgumentError, "observe requires a regression or binary learner")

  @doc "Learn that the first input should score above the second input."
  def prefer(%__MODULE__{options: %{loss: :pairwise}} = learner, winner, loser),
    do: learn(learner, %{winner: winner, loser: loser})

  def prefer(_, _, _), do: raise(ArgumentError, "prefer requires a pairwise learner")

  @doc "Evaluate the current learned circuit."
  def predict(learner, inputs), do: Evaluator.run(learner.circuit, inputs)

  @doc "Restore the complete state before the last observation, including replay and Adam moments."
  def undo(%__MODULE__{history: [previous | rest]} = learner) do
    learner |> restore_checkpoint(previous) |> Map.put(:history, rest)
  end

  def undo(_), do: raise(ArgumentError, "no retained observation to undo")

  @doc "Return to the original parameters with empty optimizer, replay and undo history."
  def reset(learner),
    do:
      new(
        Circuit.put_parameters(learner.circuit, learner.initial_parameters),
        Map.to_list(learner.options)
      )

  defdelegate snapshot(learner), to: Snapshot, as: :encode
  defdelegate restore(circuit, snapshot), to: Snapshot, as: :decode
  def dump(learner), do: learner |> snapshot() |> :erlang.term_to_binary()

  @doc "Load an uncompressed local snapshot binary (maximum 16 MiB). Does not evaluate serialized code."
  def load(circuit, <<131, tag, _::binary>> = binary)
      when tag != 80 and byte_size(binary) <= 16_777_216 do
    {snapshot, used} = :erlang.binary_to_term(binary, [:safe, :used])
    unless used == byte_size(binary), do: raise(ArgumentError, "trailing snapshot data")
    restore(circuit, snapshot)
  end

  def load(_, _), do: raise(ArgumentError, "invalid, compressed, or oversized learner snapshot")

  @doc "Current replay objective and gradients, including recency weights and the anchor penalty."
  def loss_and_grad(learner) do
    names = learner.options.trainable
    zero = Map.new(names, &{&1, 0.0})

    {loss, gradients, total} =
      learner.replay
      |> Enum.reverse()
      |> Enum.with_index()
      |> Enum.reduce({0.0, zero, 0.0}, fn {event, age}, {loss, sums, total} ->
        weight = :math.pow(learner.options.decay, age)
        {value, gradient} = event_gradient(learner, event)

        {loss + weight * value, Map.new(names, &{&1, sums[&1] + weight * gradient[&1]}),
         total + weight}
      end)

    denominator = max(total, 1.0e-300)

    penalty =
      Enum.reduce(names, 0.0, fn name, sum ->
        sum +
          learner.options.anchor *
            :math.pow(learner.circuit.parameters[name] - learner.initial_parameters[name], 2)
      end)

    gradients =
      Map.new(names, fn name ->
        difference = learner.circuit.parameters[name] - learner.initial_parameters[name]

        {name,
         Number.cast(
           gradients[name] / denominator + 2 * learner.options.anchor * difference,
           :f64
         )}
      end)

    {Number.cast(loss / denominator + penalty, :f64), gradients}
  end

  @doc false
  def checkpoint(learner),
    do: %{
      parameters: learner.circuit.parameters,
      moments: learner.moments,
      step: learner.step,
      observations: learner.observations,
      replay: learner.replay,
      last_loss: learner.last_loss
    }

  @doc false
  def restore_checkpoint(learner, state) do
    %{
      learner
      | circuit: Circuit.put_parameters(learner.circuit, state.parameters),
        moments: state.moments,
        step: state.step,
        observations: state.observations,
        replay: state.replay,
        last_loss: state.last_loss
    }
  end

  @doc false
  def validate_event(learner, %{winner: winner, loser: loser})
      when learner.options.loss == :pairwise,
      do: %{winner: validate_inputs(learner, winner), loser: validate_inputs(learner, loser)}

  def validate_event(learner, %{inputs: inputs, target: target})
      when learner.options.loss != :pairwise do
    target = Number.cast(target, learner.circuit.precision)

    if learner.options.loss == :binary_cross_entropy and target not in [0.0, 1.0],
      do: raise(ArgumentError, "binary outcomes must be 0 or 1; the circuit output is a logit")

    %{inputs: validate_inputs(learner, inputs), target: target}
  end

  def validate_event(_, _), do: raise(ArgumentError, "invalid feedback for learner loss")

  defp validate_inputs(learner, inputs) do
    inputs = Circuit.named(inputs)
    names = for {name, %{role: :input}} <- learner.circuit.sources, do: name

    unless Enum.sort(Map.keys(inputs)) == Enum.sort(names),
      do: raise(ArgumentError, "feedback must supply exactly the circuit input names")

    # Evaluation validates ranges, grids, finite arithmetic and scalar output.
    {_, _} = Evaluator.value_and_grad(learner.circuit, inputs, wrt: learner.options.trainable)
    Map.new(inputs, fn {name, value} -> {name, Number.cast(value, learner.circuit.precision)} end)
  end

  defp learn(learner, event) do
    event = validate_event(learner, event)
    history = Enum.take([checkpoint(learner) | learner.history], learner.options.undo_limit)

    learner = %{
      learner
      | replay: Enum.take(learner.replay ++ [event], -learner.options.replay_capacity),
        observations: learner.observations + 1,
        history: history
    }

    learner =
      Enum.reduce(1..learner.options.steps, learner, fn _, state ->
        {_, gradients} = loss_and_grad(state)
        step = state.step + 1

        {values, moments} =
          Enum.map_reduce(state.options.trainable, state.moments, fn name, moments ->
            {delta, moments} =
              Optimizer.update(
                state.options.optimizer,
                name,
                gradients[name],
                step,
                state.options.learning_rate,
                moments
              )

            value = Number.cast(state.circuit.parameters[name] - delta, :f64)
            {{name, Number.clamp(value, state.circuit.sources[name].range)}, moments}
          end)

        %{
          state
          | circuit: Circuit.put_parameters(state.circuit, values),
            moments: moments,
            step: step
        }
      end)

    {loss, _} = loss_and_grad(learner)
    %{learner | last_loss: loss}
  end

  defp event_gradient(learner, %{winner: a, loser: b}) do
    {a, da} = Evaluator.value_and_grad(learner.circuit, a, wrt: learner.options.trainable)
    {b, db} = Evaluator.value_and_grad(learner.circuit, b, wrt: learner.options.trainable)
    scale = learner.options.pairwise_scale
    margin = scale * (a - b)
    derivative = -scale * sigmoid(-margin)
    {softplus(-margin), Map.new(da, fn {name, d} -> {name, derivative * (d - db[name])} end)}
  end

  defp event_gradient(learner, %{inputs: inputs, target: target}) do
    {value, gradients} =
      Evaluator.value_and_grad(learner.circuit, inputs, wrt: learner.options.trainable)

    {loss, derivative} =
      case learner.options.loss do
        :mse ->
          {:math.pow(value - target, 2), 2 * (value - target)}

        :binary_cross_entropy ->
          {if(target == 1.0, do: softplus(-value), else: softplus(value)),
           sigmoid(value) - target}
      end

    {loss, Map.new(gradients, fn {name, d} -> {name, derivative * d} end)}
  end

  defp sigmoid(x) when x >= 0, do: 1.0 / (1.0 + :math.exp(-x))

  defp sigmoid(x) do
    exponential = :math.exp(x)
    exponential / (1.0 + exponential)
  end

  defp softplus(x) when x > 0, do: x + :math.log(1.0 + :math.exp(-x))
  defp softplus(x), do: :math.log(1.0 + :math.exp(x))
end
