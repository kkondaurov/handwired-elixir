defmodule Handwired.MachineTest do
  use ExUnit.Case, async: true
  alias Handwired.Machine
  alias Handwired.TestPrograms, as: P

  defp controller do
    Handwired.compile(&P.controller/6,
      inputs: [position: {-100, 100}, velocity: {-100, 100}, target: {-10, 10}, dt: {0, 0.05}],
      parameters: [stiffness: 40, damping: 12]
    )
  end

  test "a retained-state controller agrees with native Elixir at every event and settles" do
    machine = Machine.new(controller(), position: 0, velocity: 0)
    events = for i <- 1..1000, do: %{target: if(i <= 500, do: 1.0, else: -0.5), dt: 0.01}
    {outputs, final} = Machine.run(machine, events)

    {expected, _} =
      Enum.map_reduce(events, %{position: 0.0, velocity: 0.0}, fn event, state ->
        next = P.controller(state.position, state.velocity, event.target, event.dt, 40.0, 12.0)
        {next, next}
      end)

    for {output, reference} <- Enum.zip(outputs, expected) do
      assert_in_delta output.position, reference.position, 1.0e-12
      assert_in_delta output.velocity, reference.velocity, 1.0e-12
    end

    assert_in_delta final.state.position, -0.5, 1.0e-6
    assert_in_delta final.state.velocity, 0.0, 1.0e-6
    assert machine.state == %{position: 0, velocity: 0}
  end

  test "state persists and independent machines never share it" do
    circuit = Handwired.compile(&P.accumulator/2, inputs: [total: {0, 10}, amount: {0, 10}])
    first = Machine.new(circuit, total: 0)
    second = Machine.new(circuit, total: 5)
    {_, first} = Machine.step(first, amount: 2)
    {_, first} = Machine.step(first, amount: 3)
    assert first.state.total == 5
    assert second.state.total == 5
    {output, _} = Machine.step(second, amount: 1)
    assert output.total == 6
    assert_raise ArgumentError, ~r/overwrite/, fn -> Machine.step(first, total: 0, amount: 1) end
    assert_raise ArgumentError, ~r/within/, fn -> Machine.step(first, amount: 10) end
  end

  test "machine construction validates both sides of the state contract" do
    assert_raise ArgumentError, ~r/matching input/, fn ->
      Machine.new(controller(), stiffness: 1)
    end

    assert_raise ArgumentError, ~r/within/, fn ->
      Machine.new(controller(), position: 1000, velocity: 0)
    end

    scalar = Handwired.compile(&P.square/1, inputs: [x: -1..1])
    assert_raise ArgumentError, ~r/return a map/, fn -> Machine.new(scalar, x: 0) end
  end

  test "f32 state on a rounded decimal bound can be fed back unchanged" do
    circuit =
      Handwired.compile(&P.accumulator/2,
        inputs: [total: {0, 0.1}, amount: {0, 0.1}],
        precision: :f32
      )

    machine = Machine.new(circuit, total: 0)
    {first, machine} = Machine.step(machine, amount: 0.1)
    {second, _} = Machine.step(machine, amount: 0)
    assert first == second
    assert first.total == circuit.sources.total.range |> elem(1)
  end
end
