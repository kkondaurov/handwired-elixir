defmodule ControllerExample do
  use Handwired

  defwire step(position, velocity, target, dt, stiffness, damping) do
    acceleration = stiffness * (target - position) - damping * velocity
    next_velocity = velocity + acceleration * dt
    next_position = position + next_velocity * dt
    %{position: next_position, velocity: next_velocity}
  end
end

circuit =
  Handwired.compile(&ControllerExample.step/6,
    inputs: [position: {-100, 100}, velocity: {-100, 100}, target: {-10, 10}, dt: {0, 0.05}],
    parameters: [stiffness: 40, damping: 12]
  )
  |> Handwired.lower()

IO.inspect(Handwired.stats(circuit), label: "Controller primitive network")

machine = Handwired.Machine.new(circuit, position: 0, velocity: 0)
events = for index <- 1..1000, do: %{target: if(index <= 500, do: 1.0, else: -0.5), dt: 0.01}
{outputs, final_machine} = Handwired.Machine.run(machine, events)

# Execute the original function as an independent reference at every step.
{references, _} =
  Enum.map_reduce(events, %{position: 0.0, velocity: 0.0}, fn event, state ->
    next =
      ControllerExample.step(state.position, state.velocity, event.target, event.dt, 40.0, 12.0)

    {next, next}
  end)

max_error =
  Enum.zip(outputs, references)
  |> Enum.map(fn {actual, expected} ->
    max(abs(actual.position - expected.position), abs(actual.velocity - expected.velocity))
  end)
  |> Enum.max()

IO.puts("Stateful controller: one compiled graph, 1,000 successive events")

for index <- [0, 49, 99, 499, 549, 599, 999] do
  IO.inspect(%{
    step: index + 1,
    target: Enum.at(events, index).target,
    state: Enum.at(outputs, index)
  })
end

IO.inspect(max_error, label: "Largest error against native Elixir")
IO.inspect(final_machine.state, label: "Final retained state")

if max_error > 1.0e-12 or abs(final_machine.state.position + 0.5) > 1.0e-6 or
     abs(final_machine.state.velocity) > 1.0e-6,
   do: raise("controller verification failed")

IO.puts(
  "PASS: every transition matches native Elixir and the controller settles at the new target."
)
