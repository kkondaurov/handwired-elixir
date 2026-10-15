defmodule Handwired.LoweringPrograms do
  use Handwired
  defwire(headline(x), do: if(x > 4, do: 10, else: 12))

  defwire comparisons(a, b) do
    {a > b, a >= b, a < b, a <= b, a == b, a != b, a === b, a !== b}
  end

  defwire(extrema(a, b), do: {min(a, b), max(a, b), abs(a), relu(b)})

  defwire logic(a, b) do
    %{both: a > 0 and b > 0, either: a > 0 or b > 0, negated: not (a == b)}
  end

  defwire mixed(x, y, scale) do
    product = (x + y) * (x - y)
    if x > y, do: max(product, y) * scale, else: min(product, x) * scale
  end

  defwire(half(x), do: x / 2)
  defwire(third(x), do: x / 3)
  defwire(divided(x, y), do: x / y)
  defwire(train_zero(x, weight), do: x * weight)
  defwire(zero_max(x), do: max(x, 0))
  defwire(zero_min(x), do: min(x, 0))
  defwire(numeric_condition(x), do: if(x, do: x + 1, else: x - 1))
  defwire(extreme_select(x), do: if(x > 0, do: 1.0e30, else: 1.0e-30))
  defwire(inferred_grid(x, y), do: if((x + y) * 0.5 > 4, do: x, else: y))
end

defmodule Handwired.LoweringTest do
  use ExUnit.Case, async: true
  alias Handwired.LoweringPrograms, as: L
  alias Handwired.TestPrograms, as: P

  defp compile(fun, inputs, opts \\ []) do
    Handwired.compile(fun, Keyword.merge([inputs: inputs], opts))
  end

  defp primitive_only!(circuit) do
    assert circuit.backend == :primitives

    for node <- Tuple.to_list(circuit.nodes) do
      assert elem(node, 0) in [:source, :affine, :reglu]
    end

    assigned = circuit.layers |> List.flatten() |> Enum.sort()

    expected =
      circuit.nodes
      |> Tuple.to_list()
      |> Enum.with_index()
      |> Enum.reject(fn {op, _} -> elem(op, 0) == :source end)
      |> Enum.map(&elem(&1, 1))

    assert assigned == expected

    depths =
      circuit.layers
      |> Enum.with_index(1)
      |> Enum.flat_map(fn {ids, depth} -> Enum.map(ids, &{&1, depth}) end)
      |> Map.new()

    for {op, wire} <- Enum.with_index(Tuple.to_list(circuit.nodes)) do
      terms =
        case op do
          {:source, _} -> []
          {:affine, terms, _} -> terms
          {:reglu, terms, _, gates, _} -> terms ++ gates
        end

      for {parent, _} <- terms do
        assert parent < wire
        assert Map.get(depths, parent, 0) < depths[wire]
      end
    end
  end

  test "headline program becomes real affine/ReGLU gates with a two-gate integer comparison" do
    semantic = compile(&L.headline/1, x: -20..20)
    circuit = Handwired.lower(semantic, grids: [x: 1])
    primitive_only!(circuit)
    assert Handwired.stats(circuit).comparisons == %{grid: 1, cascade: 0}
    assert Handwired.stats(circuit).reglu == 4
    for x <- -20..20, do: assert(Handwired.run(circuit, x: x) == L.headline(x))
    assert Handwired.lower(circuit) == circuit
    assert_raise ArgumentError, ~r/grid/, fn -> Handwired.run(circuit, x: 4.5) end
  end

  test "floating comparisons distinguish fractions around the headline threshold" do
    circuit = compile(&L.headline/1, x: -20..20) |> Handwired.lower()
    assert Handwired.stats(circuit).comparisons == %{grid: 0, cascade: 1}

    for x <- [3.9999999999999996, 4.0, 4.000000000000001, 4.25, 4.5],
        do: assert(Handwired.run(circuit, x: x) == L.headline(x))
  end

  test "f32 sign cascades cover every exponent class including subnormals" do
    circuit =
      compile(&L.comparisons/2, [a: {-1.0e38, 1.0e38}, b: {-1.0e38, 1.0e38}], precision: :f32)
      |> Handwired.lower()

    primitive_only!(circuit)

    for exponent <- -149..126 do
      x = :math.pow(2.0, exponent)

      for {a, b} <- [{x, 0.0}, {-x, 0.0}, {0.0, x}, {x, x}, {-x, x}] do
        assert Handwired.run(circuit, a: a, b: b) == L.comparisons(a, b),
               "exponent #{exponent}, #{inspect({a, b})}"
      end
    end
  end

  test "f64 sign cascades resolve the smallest subnormal through extreme finite values" do
    circuit =
      compile(&L.comparisons/2, a: {-8.0e307, 8.0e307}, b: {-8.0e307, 8.0e307})
      |> Handwired.lower()

    exponents =
      Enum.uniq(
        [-1074, -1073, -1022, -53, -52, -1, 0, 52, 53, 1022] ++ Enum.to_list(-1074..1022//53)
      )

    for exponent <- exponents do
      x = :math.pow(2.0, exponent)

      for {a, b} <- [{x, 0.0}, {-x, 0.0}, {0.0, x}, {x, x}, {-x, x}],
          do: assert(Handwired.run(circuit, a: a, b: b) == L.comparisons(a, b))
    end

    for {a, b} <- [{1.0, 1.0000000000000002}, {8.0e307, 7.999999999999999e307}],
        do: assert(Handwired.run(circuit, a: a, b: b) == L.comparisons(a, b))
  end

  test "grid comparisons cover fractional grids, equality, and all relations" do
    circuit =
      compile(&L.comparisons/2, a: -10..10, b: -10..10)
      |> Handwired.lower(grids: [a: 1 / 64, b: 1 / 64])

    assert Handwired.stats(circuit).comparisons == %{grid: 8, cascade: 0}

    for i <- -10..10, j <- -10..10 do
      a = i / 64
      b = j / 64
      assert Handwired.run(circuit, a: a, b: b) == L.comparisons(a, b)
    end
  end

  test "grids propagate through sums and products and oversized grids fall back to cascades" do
    circuit =
      compile(&L.inferred_grid/2, x: -20..20, y: -20..20)
      |> Handwired.lower(grids: [x: 1, y: 1])

    assert Handwired.stats(circuit).comparisons == %{grid: 1, cascade: 0}

    for x <- -10..10,
        y <- -10..10,
        do: assert(Handwired.run(circuit, x: x, y: y) == L.inferred_grid(x, y))

    wide =
      compile(&L.comparisons/2, [a: {-1.0e20, 1.0e20}, b: {-1.0e20, 1.0e20}], precision: :f32)
      |> Handwired.lower(grids: [a: 1, b: 1])

    assert Handwired.stats(wide).comparisons == %{grid: 0, cascade: 8}

    assert Handwired.run(wide, a: 16_777_216, b: 16_777_218) ==
             L.comparisons(16_777_216.0, 16_777_218.0)
  end

  test "gated selection preserves a small operand beside a vastly larger one" do
    for precision <- [:f32, :f64] do
      original = compile(&L.extreme_select/1, [x: -1..1], precision: precision)
      circuit = Handwired.lower(original)

      for x <- [-1, 0, 1] do
        assert Handwired.run(circuit, x: x) == Handwired.run(original, x: x)
        assert Handwired.grad(circuit, [x: x], wrt: :inputs) == %{x: 0.0}
      end
    end
  end

  test "cascades handle the largest finite magnitude without overflowing a ReLU gate" do
    <<max32::float-32>> = <<0x7F7FFFFF::32>>
    <<max64::float-64>> = <<0x7FEFFFFFFFFFFFFF::64>>

    for {precision, largest} <- [{:f32, max32}, {:f64, max64}], sign <- [-1, 1] do
      range = if sign == 1, do: {0, largest}, else: {-largest, 0}

      circuit =
        compile(&L.comparisons/2, [a: range, b: {0, 0}], precision: precision)
        |> Handwired.lower()

      a = sign * largest
      assert Handwired.run(circuit, a: a, b: 0) == L.comparisons(a, 0.0)
    end
  end

  test "signed products, exact extrema, and structured branches agree in both precisions" do
    for precision <- [:f32, :f64],
        fun <- [&P.structured/1, &P.rebound/1, &L.numeric_condition/1] do
      original = compile(fun, [x: -10..10], precision: precision)
      lowered = Handwired.lower(original)
      primitive_only!(lowered)

      for x <- [-8.25, -0.25, 0, 0.25, 8.25],
          do: assert(Handwired.run(lowered, x: x) == Handwired.run(original, x: x))
    end

    for precision <- [:f32, :f64], fun <- [&P.product/2, &L.extrema/2, &L.logic/2] do
      names = if fun == (&P.product/2), do: [:x, :y], else: [:a, :b]

      original =
        compile(fun, Map.new(names, &{&1, {-20_000_000, 20_000_000}}), precision: precision)

      lowered = Handwired.lower(original)

      for a <- [-16_777_216.0, -3.25, 0.0, 0.3, 1000.37], b <- [-1.0, 0.0, 0.3, 3.5] do
        inputs = Enum.zip(names, [a, b])
        assert Handwired.run(lowered, inputs) == Handwired.run(original, inputs)
      end
    end
  end

  test "primitive backwards matches semantic gradients including zero multiplication and ties" do
    for precision <- [:f32, :f64], {a, b} <- [{0, 3}, {3, 0}, {0, 0}, {-2, 4}, {-3, -2}] do
      original = compile(&P.product/2, [], parameters: [x: a, y: b], precision: precision)
      lowered = Handwired.lower(original)
      assert Handwired.value_and_grad(lowered, []) == Handwired.value_and_grad(original, [])
    end

    for fun <- [&P.minimum/2, &P.maximum/2] do
      original =
        compile(fun, [], parameters: [x: 1, y: 1], parameter_ranges: [x: -10..10, y: -10..10])

      assert Handwired.grad(Handwired.lower(original), []) == %{x: 1.0, y: 0.0}
    end

    for fun <- [&L.zero_max/1, &L.zero_min/1] do
      original = compile(fun, x: -10..10)
      assert Handwired.grad(Handwired.lower(original), [x: 0], wrt: :inputs) == %{x: 1.0}
    end
  end

  test "mixed lowered objective gradients match finite differences and the semantic oracle" do
    original = compile(&L.mixed/3, [x: -10..10, y: -10..10], parameters: [scale: 1.3])
    lowered = Handwired.lower(original)

    for x <- [-3.5, -0.75, 1.5, 4.0], y <- [-2.25, 0.25, 3.25] do
      inputs = %{x: x, y: y}

      assert Handwired.value_and_grad(lowered, inputs, wrt: :all) ==
               Handwired.value_and_grad(original, inputs, wrt: :all)

      gradient = Handwired.grad(lowered, inputs, wrt: :inputs)

      for name <- [:x, :y] do
        plus = Handwired.run(lowered, Map.update!(inputs, name, &(&1 + 1.0e-5)))
        minus = Handwired.run(lowered, Map.update!(inputs, name, &(&1 - 1.0e-5)))
        assert_in_delta gradient[name], (plus - minus) / 2.0e-5, 1.0e-7
      end
    end
  end

  test "lowered parameters remain live and fitting can escape a zero initial product" do
    original = compile(&L.train_zero/2, [x: -10..10], parameters: [weight: 0.0])
    lowered = Handwired.lower(original)
    fit = Handwired.fit(lowered, [{[x: 1], 3}, {[x: 2], 6}], steps: 1500)
    assert fit.loss < 1.0e-9
    assert_in_delta Handwired.run(fit.circuit, x: 4), 12, 1.0e-4
    assert fit.circuit.nodes == lowered.nodes
    assert fit.circuit.backend == :primitives
    assert lowered.parameters.weight == 0.0
  end

  test "parameter grids are checked on compilation and updates" do
    original = compile(&P.offset/2, [x: -10..10], parameters: [bias: 1])
    lowered = Handwired.lower(original, grids: [bias: 1])
    assert_raise ArgumentError, ~r/grid/, fn -> Handwired.put_parameters(lowered, bias: 1.5) end
    invalid = Handwired.put_parameters(original, bias: 1.5)
    assert_raise ArgumentError, ~r/grid/, fn -> Handwired.lower(invalid, grids: [bias: 1]) end

    assert_raise ArgumentError, ~r/power of two/, fn ->
      Handwired.lower(original, grids: [x: 0.1])
    end
  end

  test "the envelope learns on primitive graphs in both precisions" do
    for precision <- [:f32, :f64] do
      original =
        compile(&P.envelope/3, [t: {0, 1}],
          parameters: [peak: 0.6, decay: 0.5],
          parameter_ranges: [peak: {0, 2}, decay: {0, 5}],
          precision: precision
        )

      circuit = Handwired.lower(original)
      primitive_only!(circuit)
      samples = for i <- 0..10, t = i * 0.05, do: {[t: t], max(0.0, 1.0 - 2 * t)}
      fit = Handwired.fit(circuit, samples, steps: 2000, tolerance: 1.0e-12)
      assert fit.loss < 1.0e-10

      for t <- [0.025, 0.173, 0.319, 0.477, 0.7],
          do: assert_in_delta(Handwired.run(fit.circuit, t: t), max(0.0, 1 - 2 * t), 1.0e-4)
    end
  end

  test "a state grid is checked before invalid state can be retained" do
    circuit =
      compile(&P.accumulator/2, total: -10..10, amount: -10..10)
      |> Handwired.lower(grids: [total: 1])

    assert_raise ArgumentError, ~r/grid/, fn -> Handwired.Machine.new(circuit, total: 0.5) end
    machine = Handwired.Machine.new(circuit, total: 0)
    assert_raise ArgumentError, ~r/grid/, fn -> Handwired.Machine.step(machine, amount: 0.5) end
    {value, _} = Handwired.Machine.step(machine, amount: 1)
    assert value.total == 1
  end

  test "constant division is explicit about its rounding contract" do
    original = compile(&L.half/1, x: -10..10)
    lowered = Handwired.lower(original)

    for x <- [-3.1, 0, 1.7],
        do: assert(Handwired.run(lowered, x: x) == Handwired.run(original, x: x))

    thirds = compile(&L.third/1, x: -10..10)
    assert_raise ArgumentError, ~r/power-of-two/, fn -> Handwired.lower(thirds) end
    reciprocal = Handwired.lower(thirds, division: :reciprocal)
    assert_in_delta Handwired.run(reciprocal, x: 1.7), 1.7 / 3, 1.0e-15
    runtime = compile(&L.divided/2, x: -10..10, y: 1..10)
    assert_raise ArgumentError, ~r/compile-time constant/, fn -> Handwired.lower(runtime) end
  end

  test "unbounded or overflowing comparison domains fail during lowering without fallback" do
    for range <- [nil, {-1.0e308, 1.0e308}] do
      original = compile(&L.comparisons/2, a: range, b: range)
      assert_raise ArgumentError, ~r/finite operand ranges/, fn -> Handwired.lower(original) end
    end
  end

  test "lowered state machine matches the reference for 1000 transitions" do
    original =
      compile(
        &P.controller/6,
        [position: -100..100, velocity: -100..100, target: -10..10, dt: {0, 0.05}],
        parameters: [stiffness: 40, damping: 12]
      )

    lowered = Handwired.lower(original)
    events = for i <- 1..1000, do: [target: if(i <= 500, do: 1, else: -0.5), dt: 0.01]

    {reference, _} =
      Handwired.Machine.run(Handwired.Machine.new(original, position: 0, velocity: 0), events)

    {actual, machine} =
      Handwired.Machine.run(Handwired.Machine.new(lowered, position: 0, velocity: 0), events)

    assert reference == actual
    assert machine.circuit.backend == :primitives
    assert_in_delta machine.state.position, -0.5, 1.0e-6
  end
end
