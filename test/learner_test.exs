defmodule Handwired.LearnerPrograms do
  use Handwired
  defwire(line(x, slope, bias), do: slope * x + bias)
  defwire(ranking(a, b, first, second), do: a * first + b * second)
end

defmodule Handwired.LearnerTest do
  use ExUnit.Case, async: true
  alias Handwired.Learner
  alias Handwired.LearnerPrograms, as: P

  def line(precision \\ :f64) do
    Handwired.compile(&P.line/3,
      inputs: [x: {-2000, 2000}],
      parameters: [slope: 1.0, bias: 0.0],
      parameter_ranges: [slope: {0, 4}, bias: {-2, 2}],
      precision: precision
    )
    |> Handwired.lower()
  end

  def ranking do
    Handwired.compile(&P.ranking/4,
      inputs: [a: {0, 1}, b: {0, 1}],
      parameters: [first: 1.0, second: 1.0],
      parameter_ranges: [first: {0.03, 4}, second: {0.03, 4}]
    )
    |> Handwired.lower()
  end

  test "regression improves on held-out inputs while preserving frozen parameters" do
    base = Learner.new(line(), trainable: [:slope], steps: 4)

    learner =
      Enum.reduce(1..60, base, fn n, acc ->
        Learner.observe(acc, %{x: rem(n, 3) + 1}, 3 * (rem(n, 3) + 1))
      end)

    assert abs(Learner.predict(learner, x: 1.5) - 4.5) < 0.02
    assert learner.circuit.parameters.bias == 0
    assert learner.step == 240
    assert learner.observations == 60
    assert length(learner.replay) == 60
    assert length(learner.history) == 32
    assert base.circuit.parameters.slope == 1.0
  end

  test "binary outcomes move a logit in the right direction and extreme logits stay finite" do
    for target <- [0, 1] do
      base = Learner.new(line(), loss: :binary_cross_entropy, trainable: [:bias])
      learner = Learner.observe(base, %{x: 0}, target)

      assert if target == 1,
               do: learner.circuit.parameters.bias > 0,
               else: learner.circuit.parameters.bias < 0

      for x <- [-1000, 1000] do
        learner = Learner.observe(base, %{x: x}, target)
        assert is_float(learner.last_loss)
        assert learner.last_loss >= 0
      end
    end
  end

  test "opposite pairwise feedback creates opposite held-out rankings" do
    original = Learner.new(ranking(), loss: :pairwise, steps: 30, pairwise_scale: 5)
    left = Learner.prefer(original, %{a: 1, b: 0}, %{a: 0, b: 1})
    right = Learner.prefer(original, %{a: 0, b: 1}, %{a: 1, b: 0})
    a = %{a: 0.8, b: 0.2}
    b = %{a: 0.3, b: 0.7}
    assert Learner.predict(left, a) > Learner.predict(left, b)
    assert Learner.predict(right, a) < Learner.predict(right, b)
    assert Learner.predict(original, a) == Learner.predict(original, b)
  end

  test "all three loss gradients including recency and anchor agree with central differences" do
    for loss <- [:mse, :binary_cross_entropy, :pairwise] do
      base = Learner.new(ranking(), loss: loss, anchor: 0.07, decay: 0.6, pairwise_scale: 3)

      learner =
        if loss == :pairwise do
          base
          |> Learner.prefer(%{a: 1, b: 0}, %{a: 0, b: 1})
          |> Learner.prefer(%{a: 0.2, b: 0.7}, %{a: 0.8, b: 0.4})
        else
          base |> Learner.observe(%{a: 1, b: 0.2}, 1) |> Learner.observe(%{a: 0.3, b: 0.9}, 0)
        end

      {_, gradient} = Learner.loss_and_grad(learner)

      for name <- [:first, :second] do
        eps = 1.0e-5
        value = learner.circuit.parameters[name]

        plus = %{
          learner
          | circuit: Handwired.put_parameters(learner.circuit, %{name => value + eps})
        }

        minus = %{
          learner
          | circuit: Handwired.put_parameters(learner.circuit, %{name => value - eps})
        }

        {lp, _} = Learner.loss_and_grad(plus)
        {lm, _} = Learner.loss_and_grad(minus)
        assert_in_delta gradient[name], (lp - lm) / (2 * eps), 1.0e-7
      end
    end
  end

  test "bounded replay forgets old feedback and decay gives more weight to recent feedback" do
    base =
      Learner.new(line(),
        optimizer: :sgd,
        trainable: [:bias],
        learning_rate: 0.01,
        replay_capacity: 2,
        undo_limit: 2
      )

    a = Learner.observe(base, %{x: 0}, 1)
    b = Learner.observe(a, %{x: 0}, -1)
    # Compare gradients at the same parameter point, changing only replay weighting.
    same = %{b | circuit: line()}
    {_, flat} = Learner.loss_and_grad(same)
    {_, recent} = Learner.loss_and_grad(%{same | options: %{same.options | decay: 0.1}})
    assert_in_delta flat.bias, 0, 1.0e-12
    assert recent.bias > 1
    c = Learner.observe(b, %{x: 0}, 0)
    assert Enum.map(c.replay, & &1.target) == [-1.0, 0.0]
    assert Learner.undo(c) == %{b | history: Enum.take(b.history, 1)}
    assert Learner.undo(Learner.undo(c)) == %{a | history: []}
    assert_raise ArgumentError, fn -> c |> Learner.undo() |> Learner.undo() |> Learner.undo() end
    assert Learner.reset(c) == base
  end

  test "restoring Adam state and continuing is exactly equal to uninterrupted training" do
    base = Learner.new(line(), steps: 3, replay_capacity: 3, undo_limit: 3)
    prefix = Enum.reduce(1..5, base, fn n, acc -> Learner.observe(acc, %{x: n}, 2 * n) end)
    restored = Learner.load(line(), Learner.dump(prefix))
    assert restored == prefix
    assert Learner.restore(line(), Learner.snapshot(prefix)) == prefix

    for target <- [0, 2, 4] do
      assert Learner.observe(restored, %{x: 2}, target) ==
               Learner.observe(prefix, %{x: 2}, target)
    end

    assert Learner.undo(restored) == Learner.undo(prefix)
    assert Learner.load(line(), Learner.dump(base)) == base
  end

  test "snapshots validate code identity, bounds, frozen parameters, buffers, and optimizer state" do
    base = Learner.new(line(), trainable: [:slope]) |> Learner.observe(%{x: 2}, 4)
    snapshot = Learner.snapshot(base)

    for invalid <- [
          Map.put(snapshot, "version", 9),
          Map.put(snapshot, "extra", true),
          put_in(snapshot, ["state", "parameters", "slope"], 999),
          put_in(snapshot, ["state", "parameters", "bias"], 1),
          put_in(snapshot, ["state", "moments", "slope"], [0, -1]),
          put_in(snapshot, ["state", "replay"], []),
          put_in(snapshot, ["state", "step"], 0),
          put_in(snapshot, ["options", "trainable"], ["unknown_atom"]),
          put_in(snapshot, ["state", "replay"], [%{"inputs" => %{"x" => "bad"}, "target" => 1}])
        ] do
      assert_raise ArgumentError, fn -> Learner.restore(line(), invalid) end
    end

    changed = Handwired.put_parameters(line(), slope: 2)

    assert_raise ArgumentError, ~r/different circuit/, fn ->
      Learner.restore(changed, snapshot)
    end

    assert_raise ArgumentError, fn -> Learner.restore(line(:f32), snapshot) end
    assert_raise ArgumentError, fn -> Learner.load(line(), Learner.dump(base) <> "extra") end

    assert_raise ArgumentError, fn ->
      Learner.load(line(), :erlang.term_to_binary(snapshot, [:compressed]))
    end

    assert_raise ArgumentError, fn -> Learner.load(line(), "corrupt") end
  end

  test "invalid feedback and options fail before returning a changed learner" do
    base = Learner.new(line())

    for opts <- [
          [decay: 0],
          [decay: 2],
          [steps: 0],
          [undo_limit: -1],
          [replay_capacity: 0],
          [trainable: [:nope]],
          [loss: :unknown],
          [learning_rate: 0],
          [anchor: -1],
          [foo: 2],
          [steps: 1, steps: 2]
        ] do
      assert_raise ArgumentError, fn -> Learner.new(line(), opts) end
    end

    for inputs <- [%{}, %{x: 2, bias: 1}, %{x: 9000}, %{x: "bad"}] do
      assert_raise ArgumentError, fn -> Learner.observe(base, inputs, 1) end
    end

    assert_raise ArgumentError, fn -> Learner.prefer(base, %{x: 1}, %{x: 2}) end

    assert_raise ArgumentError, fn ->
      Learner.observe(Learner.new(line(), loss: :binary_cross_entropy), %{x: 1}, 0.5)
    end

    assert base.observations == 0
  end

  test "parameter bounds and f32 rounding survive updates, snapshots, and undo" do
    base =
      Learner.new(line(:f32),
        optimizer: :sgd,
        learning_rate: 10,
        replay_capacity: 1,
        undo_limit: 1
      )

    learned = Learner.observe(base, %{x: 1}, 1.0e6)
    assert learned.circuit.parameters == %{slope: 4.0, bias: 2.0}
    assert Learner.restore(line(:f32), Learner.snapshot(learned)) == learned
    assert Learner.undo(learned) == base
  end
end
