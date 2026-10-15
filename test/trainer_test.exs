defmodule Handwired.TrainerTest do
  use ExUnit.Case, async: true
  alias Handwired.TestPrograms, as: P

  defp envelope(precision \\ :f64) do
    Handwired.compile(&P.envelope/3,
      inputs: [t: {0, 1}],
      parameters: [peak: 0.6, decay: 0.5],
      parameter_ranges: [peak: {0, 2}, decay: {0, 5}],
      precision: precision
    )
  end

  test "Adam recovers envelope parameters and predicts unseen times" do
    circuit = envelope()
    examples = for i <- 0..10, t = i * 0.05, do: {%{t: t}, max(0.0, 1.0 - 2.0 * t)}
    fit = Handwired.fit(circuit, examples, steps: 2000, tolerance: 1.0e-12)
    assert fit.loss < 1.0e-10
    assert fit.loss < fit.initial_loss / 1_000_000
    assert_in_delta fit.circuit.parameters.peak, 1.0, 1.0e-4
    assert_in_delta fit.circuit.parameters.decay, 2.0, 1.0e-4

    for t <- [0.025, 0.173, 0.319, 0.477, 0.7] do
      assert_in_delta Handwired.run(fit.circuit, t: t), max(0, 1 - 2 * t), 1.0e-4
    end

    assert circuit.parameters == %{peak: 0.6, decay: 0.5}
    assert circuit.nodes == fit.circuit.nodes
    assert length(fit.history) == fit.steps + 1
    assert List.last(fit.history) == {fit.steps, fit.loss}
  end

  test "the envelope also trains with explicit f32 arithmetic" do
    fit =
      Handwired.fit(envelope(:f32), [{%{t: 0}, 1}, {%{t: 0.2}, 0.6}, {%{t: 0.4}, 0.2}],
        steps: 2000
      )

    assert fit.loss < 1.0e-8
    assert_in_delta fit.circuit.parameters.decay, 2.0, 0.001
  end

  test "SGD, frozen parameters, and parameter range projection" do
    circuit = envelope() |> Handwired.put_parameters(peak: 1)

    fit =
      Handwired.fit(circuit, [{%{t: 0.1}, 0.8}, {%{t: 0.3}, 0.4}],
        trainable: [:decay],
        optimizer: :sgd,
        learning_rate: 1.0,
        steps: 500
      )

    assert fit.circuit.parameters.peak == 1.0
    assert fit.loss < 1.0e-8

    bounded =
      Handwired.compile(&P.offset/2,
        inputs: [x: -1..1],
        parameters: [bias: 0],
        parameter_ranges: [bias: {0, 1}]
      )

    fit = Handwired.fit(bounded, [{%{x: 0}, 10}], steps: 100)
    assert fit.circuit.parameters.bias == 1.0
    assert fit.loss == 81.0
  end

  test "zero-step fits report the initial loss and runs are reproducible" do
    examples = [{%{t: 0.2}, 0.6}]
    fit = Handwired.fit(envelope(), examples, steps: 0)
    assert fit.steps == 0
    assert fit.loss == fit.initial_loss

    assert Handwired.fit(envelope(), examples, steps: 10) ==
             Handwired.fit(envelope(), examples, steps: 10)
  end

  test "invalid training requests fail early" do
    assert_raise ArgumentError, ~r/nonempty/, fn -> Handwired.fit(envelope(), []) end

    for opts <- [
          [steps: -1],
          [learning_rate: 0],
          [optimizer: :unknown],
          [trainable: [:t]],
          [tolerance: -1]
        ] do
      assert_raise ArgumentError, fn -> Handwired.fit(envelope(), [{%{t: 0}, 1}], opts) end
    end
  end
end
