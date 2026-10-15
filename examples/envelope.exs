defmodule EnvelopeExample do
  use Handwired

  defwire amplitude(t, peak, decay) do
    max(0.0, peak - decay * t)
  end
end

circuit =
  Handwired.compile(&EnvelopeExample.amplitude/3,
    inputs: [t: {0.0, 1.0}],
    parameters: [peak: 0.6, decay: 0.5],
    parameter_ranges: [peak: {0.0, 2.0}, decay: {0.0, 5.0}]
  )
  |> Handwired.lower()

IO.inspect(Handwired.stats(circuit), label: "Envelope primitive network")

# A desired envelope, sampled at eleven times. Its controls are unknown to fit/3.
examples = for i <- 0..10, t = i * 0.05, do: {%{t: t}, max(0.0, 1.0 - 2.0 * t)}
fit = Handwired.fit(circuit, examples, steps: 2000, tolerance: 1.0e-12)

IO.puts("Envelope: authored Elixir -> affine/ReGLU network -> gradients -> fitted parameters")
IO.inspect(circuit.parameters, label: "Initial controls")
IO.inspect(fit.circuit.parameters, label: "Learned controls")

IO.inspect(%{before: fit.initial_loss, after: fit.loss, steps: fit.steps},
  label: "Mean squared error"
)

for t <- [0.025, 0.173, 0.319, 0.477, 0.7] do
  predicted = Handwired.run(fit.circuit, t: t)
  target = max(0.0, 1.0 - 2.0 * t)
  IO.inspect(%{t: t, predicted: predicted, target: target}, label: "Unseen point")
  if abs(predicted - target) > 1.0e-4, do: raise("held-out prediction failed")
end

if fit.loss > 1.0e-10, do: raise("envelope fitting failed")
IO.puts("PASS: the graph learned the controls and predicts held-out samples.")
