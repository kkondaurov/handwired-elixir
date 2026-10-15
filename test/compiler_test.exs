defmodule Handwired.CompilerTest do
  use ExUnit.Case, async: true
  alias Handwired.TestPrograms, as: P

  test "compiled arithmetic agrees with independently executed native Elixir" do
    circuit =
      Handwired.compile(&P.formula/3,
        inputs: [x: {-10, 10}, y: {-10, 10}],
        parameters: [scale: 2.3]
      )

    for i <- -20..20, j <- -10..10 do
      x = i / 3.0
      y = j / 2.0
      assert_in_delta Handwired.run(circuit, %{x: x, y: y}), P.formula(x, y, 2.3), 1.0e-12
    end
  end

  test "each branch agrees with native Elixir including the boundary" do
    circuit = Handwired.compile(&P.branch/2, inputs: [x: -10..10], parameters: [scale: 3])

    for x <- [-2.5, -0.01, 0.0, 0.01, 4.0] do
      assert Handwired.run(circuit, x: x) == P.branch(x, 3.0)
    end
  end

  test "binding versions and branch-local bindings preserve Elixir scope" do
    for {function, native} <- [{&P.rebound/1, &P.rebound/1}, {&P.scoped/1, &P.scoped/1}] do
      circuit = Handwired.compile(function, inputs: [x: -10..10])
      for x <- [-3.0, 0.0, 3.0], do: assert(Handwired.run(circuit, x: x) == native.(x))
    end
  end

  test "inactive branches and short-circuit operands do not execute" do
    for function <- [&P.safe_inverse/1, &P.safe_boolean/1, &P.safe_or/1] do
      circuit = Handwired.compile(function, inputs: [x: -10..10])
      for x <- [0.0, -2.0, 2.0], do: assert(Handwired.run(circuit, x: x) == function.(x))
    end
  end

  test "numeric zero is truthy as in Elixir" do
    circuit = Handwired.compile(&P.truthiness/1, inputs: [x: -1..1])
    assert Handwired.run(circuit, x: 0) == 10
  end

  test "fixed nested map, tuple and list results work in branches" do
    circuit = Handwired.compile(&P.structured/1, inputs: [x: -10..10])
    for x <- [-3.0, 0.0, 3.0], do: assert(Handwired.run(circuit, x: x) == P.structured(x))
  end

  test "f32 rounds inputs and each arithmetic operation" do
    f32 = Handwired.compile(&P.rounded/1, inputs: [x: {0, 20_000_000}], precision: :f32)
    f64 = Handwired.compile(&P.rounded/1, inputs: [x: {0, 20_000_000}])
    assert Handwired.run(f32, x: 16_777_216) == 0.0
    assert Handwired.run(f64, x: 16_777_216) == 1.0
    # Independently round through IEEE binary storage at each step.
    round32 = fn value ->
      <<v::float-32>> = <<value * 1.0::float-32>>
      v
    end

    for x <- [0.1, 1.2, 16_777_215, 16_777_217] do
      rounded_x = round32.(x)
      expected = round32.(round32.(rounded_x + round32.(1)) - rounded_x)
      assert Handwired.run(f32, x: x) == expected
    end
  end

  test "finite numeric ranges and input schemas are enforced" do
    circuit = Handwired.compile(&P.square/1, inputs: [x: -2..2])
    assert_raise ArgumentError, ~r/within/, fn -> Handwired.run(circuit, x: 3) end
    assert_raise ArgumentError, ~r/finite/, fn -> Handwired.run(circuit, x: "2") end
    assert_raise ArgumentError, ~r/expected inputs/, fn -> Handwired.run(circuit, %{}) end

    assert_raise ArgumentError, ~r/expected inputs/, fn ->
      Handwired.run(circuit, x: 1, extra: 2)
    end

    assert_raise ArgumentError, ~r/unique/, fn -> Handwired.run(circuit, x: 1, x: 2) end
  end

  test "nonfinite arithmetic fails with the offending graph node" do
    circuit = Handwired.compile(&P.overflow/1, inputs: [x: {0, 1.0e30}], precision: :f32)
    assert_raise ArgumentError, ~r/node.*multiply/, fn -> Handwired.run(circuit, x: 1.0e30) end
  end

  test "unsupported syntax and output structure changes give compile errors" do
    assert_raise ArgumentError, ~r/unsupported expression/, fn ->
      Handwired.compile(&P.unsupported/1, inputs: [x: -1..1])
    end

    assert_raise ArgumentError, ~r/matching types/, fn ->
      Handwired.compile(&P.mismatch/1, inputs: [x: -1..1])
    end
  end

  test "declarations must cover the arguments exactly" do
    assert_raise ArgumentError, ~r/declare every/, fn ->
      Handwired.compile(&P.square/1, inputs: [])
    end

    assert_raise ArgumentError, ~r/overlap/, fn ->
      Handwired.compile(&P.square/1, inputs: [x: -1..1], parameters: [x: 0])
    end

    assert_raise ArgumentError, ~r/unknown parameter/, fn ->
      Handwired.compile(&P.square/1, inputs: [x: -1..1], parameter_ranges: [z: {0, 1}])
    end

    assert_raise ArgumentError, ~r/precision/, fn ->
      Handwired.compile(&P.square/1, inputs: [x: -1..1], precision: :other)
    end

    assert_raise ArgumentError, ~r/external capture/, fn ->
      Handwired.compile(fn x -> x end, inputs: [x: -1..1])
    end
  end

  test "parameter updates validate ranges and retain the graph" do
    circuit =
      Handwired.compile(&P.offset/2,
        inputs: [x: -2..2],
        parameters: [bias: 0],
        parameter_ranges: [bias: -1..1]
      )

    next = Handwired.put_parameters(circuit, bias: 1)
    assert next.nodes == circuit.nodes
    assert Handwired.run(next, x: 1) == 2
    assert Handwired.run(circuit, x: 1) == 1
    assert_raise ArgumentError, ~r/within/, fn -> Handwired.put_parameters(circuit, bias: 2) end

    assert_raise ArgumentError, ~r/unknown parameter/, fn ->
      Handwired.put_parameters(circuit, other: 1)
    end
  end
end
