defmodule Handwired.Machine do
  @moduledoc """
  An explicit, immutable state machine around a circuit with map outputs.

  State keys name input cells and matching output fields. `step/2` combines the
  retained state with an event's inputs and returns `{output, next_machine}`.
  A machine can be stored in a GenServer, but needs no process to execute.
  Gradients of individual circuits are available; backpropagation through an
  entire event history is not implemented in this first milestone.
  """
  alias Handwired.{Circuit, Evaluator, Number}
  defstruct [:circuit, :state]

  def new(circuit, initial_state) do
    state = Circuit.named(initial_state)

    outputs =
      case circuit.output do
        {:map, fields} -> fields
        _ -> raise ArgumentError, "a stateful circuit must return a map"
      end

    if map_size(state) == 0, do: raise(ArgumentError, "a machine needs at least one state cell")

    state =
      Map.new(state, fn {name, value} ->
        unless match?(%{role: :input}, circuit.sources[name]) and Map.has_key?(outputs, name),
          do: raise(ArgumentError, "state #{inspect(name)} needs matching input and output cells")

        spec = circuit.sources[name]
        value = Number.validate(value, spec.range, circuit.precision, name)
        Handwired.Domain.validate_grid(value, Map.get(spec, :grid), name)
        {name, value}
      end)

    %__MODULE__{circuit: circuit, state: state}
  end

  def step(%__MODULE__{} = machine, event) do
    event = Circuit.named(event)

    if Enum.any?(Map.keys(event), &Map.has_key?(machine.state, &1)),
      do: raise(ArgumentError, "event inputs cannot overwrite stored state")

    output = Evaluator.run(machine.circuit, Map.merge(machine.state, event))
    next_state = Map.take(output, Map.keys(machine.state))

    Enum.each(next_state, fn {name, value} ->
      spec = machine.circuit.sources[name]
      Number.validate(value, spec.range, machine.circuit.precision, name)
      Handwired.Domain.validate_grid(value, Map.get(spec, :grid), name)
    end)

    {output, %{machine | state: next_state}}
  end

  def run(machine, events) do
    Enum.map_reduce(events, machine, fn event, machine -> step(machine, event) end)
  end
end
