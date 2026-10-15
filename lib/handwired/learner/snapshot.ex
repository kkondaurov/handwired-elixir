defmodule Handwired.Learner.Snapshot do
  @moduledoc false
  alias Handwired.{Circuit, Learner, Number}

  @option_keys ~w(loss optimizer learning_rate steps replay_capacity decay anchor pairwise_scale undo_limit trainable)
  @frame_keys ~w(parameters moments step observations replay last_loss)

  def encode(learner) do
    original = Circuit.put_parameters(learner.circuit, learner.initial_parameters)

    %{
      "version" => 1,
      "circuit" => fingerprint(original),
      "options" =>
        Map.new(learner.options, fn {k, v} -> {Atom.to_string(k), option_value(v)} end),
      "state" => frame(learner |> Learner.checkpoint()),
      "undo" => Enum.map(learner.history, &frame/1)
    }
  end

  def decode(circuit, data) do
    keys!(data, ~w(version circuit options state undo))
    unless data["version"] == 1, do: raise(ArgumentError, "unsupported learner snapshot version")

    unless data["circuit"] == fingerprint(circuit),
      do: raise(ArgumentError, "snapshot belongs to a different circuit or starting parameters")

    options = options(circuit, data["options"])
    learner = Learner.new(circuit, options)
    current = state(learner, data["state"])
    history = data["undo"]

    unless is_list(history) and
             length(history) <= min(current.observations, learner.options.undo_limit),
           do: raise(ArgumentError, "invalid undo history length")

    history =
      history
      |> Enum.with_index(1)
      |> Enum.map(fn {item, offset} ->
        checkpoint = state(learner, item)

        unless checkpoint.observations == current.observations - offset,
          do: raise(ArgumentError, "undo history is not consecutive")

        checkpoint
      end)

    %{Learner.restore_checkpoint(learner, current) | history: history}
  rescue
    e in ArgumentError -> reraise e, __STACKTRACE__
    _ -> raise ArgumentError, "invalid learner snapshot"
  end

  defp fingerprint(circuit),
    do:
      :crypto.hash(
        :sha256,
        :erlang.term_to_binary(circuit, [:deterministic, {:minor_version, 1}])
      )
      |> Base.encode16(case: :lower)

  defp option_value(value) when is_atom(value), do: Atom.to_string(value)
  defp option_value(value) when is_list(value), do: Enum.map(value, &Atom.to_string/1)
  defp option_value(value), do: value
  defp named(values), do: Map.new(values, fn {k, v} -> {Atom.to_string(k), v} end)

  defp frame(state) do
    %{
      "parameters" => named(state.parameters),
      "step" => state.step,
      "observations" => state.observations,
      "last_loss" => state.last_loss,
      "moments" =>
        Map.new(state.moments, fn {name, {m, v}} -> {Atom.to_string(name), [m, v]} end),
      "replay" => Enum.map(state.replay, &event/1)
    }
  end

  defp event(%{winner: a, loser: b}), do: %{"winner" => named(a), "loser" => named(b)}

  defp event(%{inputs: inputs, target: target}),
    do: %{"inputs" => named(inputs), "target" => target}

  defp options(circuit, data) do
    keys!(data, @option_keys)

    enum = %{
      "mse" => :mse,
      "binary_cross_entropy" => :binary_cross_entropy,
      "pairwise" => :pairwise,
      "adam" => :adam,
      "sgd" => :sgd
    }

    names = Map.new(circuit.parameters, fn {name, _} -> {Atom.to_string(name), name} end)
    unless is_list(data["trainable"]), do: raise(ArgumentError, "invalid trainable names")

    [
      loss: Map.fetch!(enum, data["loss"]),
      optimizer: Map.fetch!(enum, data["optimizer"]),
      learning_rate: data["learning_rate"],
      steps: data["steps"],
      replay_capacity: data["replay_capacity"],
      decay: data["decay"],
      anchor: data["anchor"],
      pairwise_scale: data["pairwise_scale"],
      undo_limit: data["undo_limit"],
      trainable: Enum.map(data["trainable"], &Map.fetch!(names, &1))
    ]
  end

  defp state(learner, data) do
    keys!(data, @frame_keys)
    observations = data["observations"]

    unless is_integer(observations) and observations >= 0 and observations <= 1_000_000_000_000 and
             is_integer(data["step"]) and data["step"] == observations * learner.options.steps,
           do: raise(ArgumentError, "invalid observation or optimizer step count")

    parameters = decode_named(data["parameters"], Map.keys(learner.circuit.parameters))
    circuit = Circuit.put_parameters(learner.circuit, parameters)

    for name <- Map.keys(parameters) -- learner.options.trainable do
      unless circuit.parameters[name] == learner.initial_parameters[name],
        do: raise(ArgumentError, "snapshot changed a frozen parameter")
    end

    moment_names =
      if learner.options.optimizer == :adam and observations > 0,
        do: learner.options.trainable,
        else: []

    moments =
      decode_named(data["moments"], moment_names)
      |> Map.new(fn {name, pair} ->
        [m, v] = pair
        m = Number.cast(m, :f64)
        v = Number.cast(v, :f64)
        unless v >= 0, do: raise(ArgumentError, "negative optimizer variance")
        {name, {m, v}}
      end)

    replay = data["replay"]

    unless is_list(replay) and
             length(replay) == min(observations, learner.options.replay_capacity),
           do: raise(ArgumentError, "invalid replay buffer length")

    replay = Enum.map(replay, &decode_event(%{learner | circuit: circuit}, &1))

    loss =
      if observations == 0 do
        unless data["last_loss"] == nil and circuit.parameters == learner.initial_parameters,
          do: raise(ArgumentError, "invalid initial learner state")

        nil
      else
        loss = Number.cast(data["last_loss"], :f64)
        unless loss >= 0, do: raise(ArgumentError, "invalid last loss")
        loss
      end

    %{
      parameters: circuit.parameters,
      moments: moments,
      replay: replay,
      last_loss: loss,
      observations: observations,
      step: data["step"]
    }
  end

  defp decode_event(learner, data) do
    names = for {name, %{role: :input}} <- learner.circuit.sources, do: name

    event =
      if learner.options.loss == :pairwise do
        keys!(data, ~w(winner loser))
        %{winner: decode_named(data["winner"], names), loser: decode_named(data["loser"], names)}
      else
        keys!(data, ~w(inputs target))
        %{inputs: decode_named(data["inputs"], names), target: data["target"]}
      end

    Learner.validate_event(learner, event)
  end

  defp decode_named(data, names) do
    keys!(data, Enum.map(names, &Atom.to_string/1))
    Map.new(names, &{&1, Map.fetch!(data, Atom.to_string(&1))})
  end

  defp keys!(data, keys) do
    unless is_map(data) and Enum.sort(Map.keys(data)) == Enum.sort(keys),
      do: raise(ArgumentError, "unexpected snapshot fields")
  end
end
