defmodule Handwired.FitResult do
  @moduledoc "Fit result, including the updated circuit and MSE history `{step, loss}`."
  defstruct [:circuit, :initial_loss, :loss, :steps, :history]
end

defmodule Handwired.Trainer do
  @moduledoc false
  alias Handwired.{Circuit, Evaluator, FitResult, Number}

  def fit(circuit, examples, opts \\ []) do
    unknown = Keyword.keys(opts) -- [:steps, :learning_rate, :optimizer, :tolerance, :trainable]
    if unknown != [], do: raise(ArgumentError, "unknown fit options: #{inspect(unknown)}")

    if not is_list(examples) or examples == [],
      do: raise(ArgumentError, "fit needs a nonempty list of examples")

    examples =
      Enum.map(examples, fn
        {inputs, target} -> {Circuit.named(inputs), Number.cast(target, circuit.precision)}
        _ -> raise ArgumentError, "each example must be {inputs, numeric_target}"
      end)

    steps = Keyword.get(opts, :steps, 1000)
    rate = Number.cast(Keyword.get(opts, :learning_rate, 0.03), :f64)
    tolerance = Number.cast(Keyword.get(opts, :tolerance, 1.0e-10), :f64)
    optimizer = Keyword.get(opts, :optimizer, :adam)
    names = Keyword.get(opts, :trainable, Map.keys(circuit.parameters))

    unless is_integer(steps) and steps >= 0,
      do: raise(ArgumentError, "steps must be a nonnegative integer")

    unless rate > 0 and tolerance >= 0,
      do: raise(ArgumentError, "learning_rate must be positive and tolerance nonnegative")

    unless optimizer in [:adam, :sgd], do: raise(ArgumentError, "optimizer must be :adam or :sgd")

    unless is_list(names) and names != [] and names -- Map.keys(circuit.parameters) == [],
      do: raise(ArgumentError, "trainable must name existing parameters")

    names = Enum.uniq(names)
    {initial_loss, gradients} = batch(circuit, examples, names)

    initial = %{
      circuit: circuit,
      loss: initial_loss,
      gradients: gradients,
      moments: %{},
      step: 0,
      history: [{0, initial_loss}]
    }

    iterations = if steps == 0, do: [], else: 1..steps

    final =
      Enum.reduce_while(iterations, initial, fn step, state ->
        if state.loss <= tolerance do
          {:halt, state}
        else
          {values, moments} =
            Enum.map_reduce(names, state.moments, fn name, moments ->
              gradient = state.gradients[name]
              {delta, moments} = update(optimizer, name, gradient, step, rate, moments)

              value =
                Number.clamp(
                  state.circuit.parameters[name] - delta,
                  state.circuit.sources[name].range
                )

              {{name, value}, moments}
            end)

          circuit = Circuit.put_parameters(state.circuit, values)
          {loss, gradients} = batch(circuit, examples, names)

          {:cont,
           %{
             circuit: circuit,
             loss: loss,
             gradients: gradients,
             moments: moments,
             step: step,
             history: [{step, loss} | state.history]
           }}
        end
      end)

    %FitResult{
      circuit: final.circuit,
      initial_loss: initial_loss,
      loss: final.loss,
      steps: final.step,
      history: Enum.reverse(final.history)
    }
  end

  defp batch(circuit, examples, names) do
    count = length(examples)
    zero = Map.new(names, &{&1, 0.0})

    Enum.reduce(examples, {0.0, zero}, fn {inputs, target}, {loss, gradients} ->
      {value, partials} = Evaluator.value_and_grad(circuit, inputs, wrt: names)
      error = value - target

      gradients =
        Map.new(names, fn name ->
          {name, Number.cast(gradients[name] + 2.0 * error * partials[name] / count, :f64)}
        end)

      {Number.cast(loss + error * error / count, :f64), gradients}
    end)
  end

  defp update(:sgd, _, gradient, _, rate, moments), do: {rate * gradient, moments}

  defp update(:adam, name, gradient, step, rate, moments) do
    {m, v} = Map.get(moments, name, {0.0, 0.0})
    m = 0.9 * m + 0.1 * gradient
    v = 0.999 * v + 0.001 * gradient * gradient
    corrected_m = m / (1.0 - :math.pow(0.9, step))
    corrected_v = v / (1.0 - :math.pow(0.999, step))
    delta = rate * corrected_m / (:math.sqrt(corrected_v) + 1.0e-8)
    {delta, Map.put(moments, name, {m, v})}
  end
end
