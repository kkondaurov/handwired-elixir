defmodule Handwired.GradientTest do
  use ExUnit.Case, async: true
  alias Handwired.TestPrograms, as: P

  test "reverse differentiation agrees with central differences across mixed arithmetic" do
    circuit =
      Handwired.compile(&P.formula/3, inputs: [x: -10..10, y: -10..10], parameters: [scale: 2.3])

    epsilon = 1.0e-5

    for x <- [-3.4, -0.7, 0.8, 3.2], y <- [-2.1, 0.3, 2.4] do
      inputs = %{x: x, y: y}
      gradients = Handwired.grad(circuit, inputs, wrt: :all)

      for name <- [:x, :y] do
        plus = Handwired.run(circuit, Map.update!(inputs, name, &(&1 + epsilon)))
        minus = Handwired.run(circuit, Map.update!(inputs, name, &(&1 - epsilon)))
        assert_in_delta gradients[name], (plus - minus) / (2 * epsilon), 1.0e-7
      end

      plus = Handwired.run(Handwired.put_parameters(circuit, scale: 2.3 + epsilon), inputs)
      minus = Handwired.run(Handwired.put_parameters(circuit, scale: 2.3 - epsilon), inputs)
      assert_in_delta gradients.scale, (plus - minus) / (2 * epsilon), 1.0e-7
    end
  end

  test "shared wire contributions add and multiplication at zero differentiates correctly" do
    square = Handwired.compile(&P.square/1, inputs: [x: -10..10])
    assert Handwired.value_and_grad(square, [x: 3], wrt: :inputs) == {9.0, %{x: 6.0}}
    product = Handwired.compile(&P.product/2, parameters: [x: 0, y: 3])
    assert Handwired.grad(product, %{}) == %{x: 3.0, y: 0.0}
  end

  test "only the selected branch contributes gradients" do
    circuit = Handwired.compile(&P.branch/2, inputs: [x: -10..10], parameters: [scale: 3])
    assert Handwired.grad(circuit, [x: 2], wrt: :all) == %{x: 3.0, scale: 2.0}
    assert Handwired.grad(circuit, [x: -2], wrt: :all) == %{x: -3.0, scale: 2.0}
    inverse = Handwired.compile(&P.safe_inverse/1, inputs: [x: -10..10])
    assert Handwired.grad(inverse, [x: 0], wrt: :inputs) == %{x: 0.0}
    assert Handwired.grad(inverse, [x: 2], wrt: :inputs) == %{x: -0.25}
  end

  test "decision thresholds have zero gradient with no implicit surrogate" do
    circuit = Handwired.compile(&P.comparison/2, inputs: [x: -10..10], parameters: [threshold: 4])

    for x <- [3.9, 4, 4.1],
        do: assert(Handwired.grad(circuit, [x: x], wrt: :all) == %{x: 0.0, threshold: 0.0})
  end

  test "documented ReLU corner and min/max tie derivatives" do
    relu = Handwired.compile(&P.rectified/1, inputs: [x: -1..1])
    assert Handwired.grad(relu, [x: 0], wrt: :inputs) == %{x: 0.0}
    assert Handwired.grad(relu, [x: 1], wrt: :inputs) == %{x: 1.0}

    for fun <- [&P.minimum/2, &P.maximum/2] do
      circuit = Handwired.compile(fun, parameters: [x: 1, y: 1])
      assert Handwired.grad(circuit, %{}) == %{x: 1.0, y: 0.0}
    end
  end

  test "f32 backward pass works on the rounded intermediates" do
    circuit = Handwired.compile(&P.product/2, parameters: [x: 0, y: 3], precision: :f32)
    assert Handwired.grad(circuit, %{}) == %{x: 3.0, y: 0.0}
  end

  test "structured outputs need an explicitly authored scalar objective" do
    circuit = Handwired.compile(&P.structured/1, inputs: [x: -10..10])
    assert_raise ArgumentError, ~r/scalar numeric output/, fn -> Handwired.grad(circuit, x: 1) end
  end

  test "an input unused by the output has a zero gradient" do
    circuit = Handwired.compile(&P.constant/1, inputs: [x: -10..10])
    assert Handwired.grad(circuit, [x: 3], wrt: :all) == %{x: 0.0}

    assert_raise ArgumentError, ~r/unknown gradient/, fn ->
      Handwired.grad(circuit, [x: 3], wrt: [:missing])
    end
  end
end
