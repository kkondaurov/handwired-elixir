defmodule DeliveryScore do
  use Handwired

  defwire score(cost, minutes, cost_weight, time_weight) do
    -cost_weight * cost / 16 - time_weight * minutes / 64
  end
end

alias Handwired.Learner

circuit =
  Handwired.compile(&DeliveryScore.score/4,
    inputs: [cost: {0, 16}, minutes: {0, 64}],
    parameters: [cost_weight: 1, time_weight: 1],
    parameter_ranges: [cost_weight: {0.03, 4}, time_weight: {0.03, 4}]
  )
  |> Handwired.lower()

# A customer chooses the quicker option despite its higher price.
learner = Learner.new(circuit, loss: :pairwise, steps: 50)
learner = Learner.prefer(learner, %{cost: 12, minutes: 10}, %{cost: 4, minutes: 50})

# Test the inferred trade-off on a different, unseen pair.
quick = %{cost: 10, minutes: 15}
cheap = %{cost: 5, minutes: 40}
true = Learner.predict(learner, quick) > Learner.predict(learner, cheap)
IO.inspect(learner.circuit.parameters, label: "Learned trade-offs")

# Persist this binary in your storage. The original circuit is supplied on load.
bytes = Learner.dump(learner)
restored = Learner.load(circuit, bytes)
true = restored == learner
continued = Learner.prefer(restored, quick, cheap)
true = continued.step == 100
true = Learner.undo(continued) == restored
true = Learner.reset(continued).circuit == circuit
IO.puts("Held-out ranking, save/restore, continued learning, undo and reset passed.")
