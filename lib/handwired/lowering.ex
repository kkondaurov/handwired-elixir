defmodule Handwired.Lowering do
  @moduledoc """
  Lower a semantic circuit to sources, ordered sparse affine forms, and ReGLU
  products. Comparisons expand to integer-grid bits or exact floating-point sign
  cascades. The forward program contains no host comparison or selection opcode.

  `grids: [name: step]` declares checked power-of-two source grids.
  `division: :reciprocal` opts into rounded reciprocal multiplication for literal
  divisors; the default `:exact` accepts only power-of-two literal divisors.
  """
  alias Handwired.{Circuit, Domain, Number}

  def lower(circuit, opts \\ [])
  def lower(%Circuit{backend: :primitives} = circuit, []), do: circuit

  def lower(%Circuit{backend: :semantic} = circuit, opts) do
    unknown = Keyword.keys(opts) -- [:grids, :division]
    if unknown != [], do: raise(ArgumentError, "unknown lowering options: #{inspect(unknown)}")
    grids = opts |> Keyword.get(:grids, []) |> Circuit.named()

    if Map.keys(grids) -- Map.keys(circuit.sources) != [],
      do: raise(ArgumentError, "grid declaration names an unknown source")

    division = Keyword.get(opts, :division, :exact)
    unless division in [:exact, :reciprocal], do: raise(ArgumentError, "invalid division mode")

    state = %{
      nodes: [],
      next: 0,
      info: %{},
      mapped: %{},
      macros: %{},
      constants: %{},
      precision: circuit.precision,
      division: division,
      comparisons: %{grid: 0, cascade: 0}
    }

    {sources, state} =
      circuit.sources
      |> Enum.sort_by(fn {_, spec} -> spec.wire end)
      |> Enum.map_reduce(state, fn {name, spec}, state ->
        grid =
          if Map.has_key?(grids, name), do: Domain.declared_grid(grids[name], circuit.precision)

        info = %{range: spec.range, grid: grid, constant: nil, type: :number}
        {wire, state} = emit({:source, name}, info, state)
        state = %{state | mapped: Map.put(state.mapped, spec.wire, wire)}
        {{name, Map.merge(spec, %{wire: wire, grid: grid})}, state}
      end)

    needed = reachable(circuit.nodes, leaves(circuit.output), MapSet.new())

    state =
      needed
      |> Enum.sort()
      |> Enum.reduce(state, fn id, state ->
        if Map.has_key?(state.mapped, id) do
          state
        else
          operation = elem(circuit.nodes, id)
          {wire, state} = lower_operation(operation, state)
          %{state | mapped: Map.put(state.mapped, id, wire)}
        end
      end)

    nodes = state.nodes |> Enum.reverse() |> List.to_tuple()

    result = %Circuit{
      circuit
      | nodes: nodes,
        sources: Map.new(sources),
        output: remap_output(circuit.output, state.mapped),
        backend: :primitives,
        macros: state.macros,
        layers: layers(nodes),
        lowering: %{
          comparisons: state.comparisons,
          division: division,
          semantic_nodes: tuple_size(circuit.nodes),
          wire_map: state.mapped
        }
    }

    Circuit.put_parameters(result, result.parameters)
  end

  def lower(_, _),
    do:
      raise(
        ArgumentError,
        "lower expects a semantic circuit (or an already lowered circuit without options)"
      )

  @doc "Counts of instructions, sparse weights, sources, and dependency layers."
  def stats(%Circuit{backend: :primitives} = circuit) do
    counts = circuit.nodes |> Tuple.to_list() |> Enum.frequencies_by(&elem(&1, 0))

    weights =
      circuit.nodes
      |> Tuple.to_list()
      |> Enum.reduce(0, fn
        {:source, _}, n -> n
        {:affine, terms, _}, n -> n + length(terms) + 1
        {:reglu, terms, _, gates, _}, n -> n + length(terms) + length(gates) + 2
      end)

    %{
      sources: Map.get(counts, :source, 0),
      affine: Map.get(counts, :affine, 0),
      reglu: Map.get(counts, :reglu, 0),
      layers: length(circuit.layers),
      weights: weights,
      comparisons: circuit.lowering.comparisons
    }
  end

  defp leaves({:wire, id, _}), do: [id]
  defp leaves({:map, fields}), do: Enum.flat_map(fields, fn {_, shape} -> leaves(shape) end)
  defp leaves({_, entries}), do: Enum.flat_map(entries, &leaves/1)

  defp reachable(_, [], seen), do: seen

  defp reachable(nodes, [id | rest], seen) do
    if MapSet.member?(seen, id),
      do: reachable(nodes, rest, seen),
      else: reachable(nodes, parents(elem(nodes, id)) ++ rest, MapSet.put(seen, id))
  end

  defp parents({:source, _}), do: []
  defp parents({:constant, _}), do: []
  defp parents(operation), do: operation |> Tuple.to_list() |> tl()

  defp remap_output({:wire, id, type}, mapped), do: {:wire, Map.fetch!(mapped, id), type}

  defp remap_output({:map, fields}, mapped),
    do: {:map, Map.new(fields, fn {key, shape} -> {key, remap_output(shape, mapped)} end)}

  defp remap_output({kind, entries}, mapped),
    do: {kind, Enum.map(entries, &remap_output(&1, mapped))}

  defp emit(operation, info, state) do
    id = state.next

    {id,
     %{
       state
       | next: id + 1,
         nodes: [operation | state.nodes],
         info: Map.put(state.info, id, info)
     }}
  end

  defp tag({wire, state}, updates),
    do: {wire, %{state | info: Map.update!(state.info, wire, &Map.merge(&1, updates))}}

  defp macro_rule({wire, state}, rule),
    do: {wire, %{state | macros: Map.put(state.macros, wire, rule)}}

  defp form(terms, bias, state) do
    bias = Number.cast(bias, state.precision)
    terms = Enum.map(terms, fn {wire, weight} -> {wire, Number.cast(weight, state.precision)} end)

    products =
      Enum.map(terms, fn {wire, weight} ->
        info = state.info[wire]

        %{
          range: Domain.multiply(info.range, {weight, weight}, state.precision),
          grid: Domain.multiply_grid(info.grid, Domain.grid(weight)),
          constant:
            if(info.constant != nil,
              do: Domain.safe(fn -> Number.cast(info.constant * weight, state.precision) end)
            )
        }
      end)

    grid = Enum.reduce(products, Domain.grid(bias), &Domain.combine(&1.grid, &2))

    {range, grid, constant} =
      Enum.reduce(products, {{bias, bias}, grid, bias}, fn product, {range, grid, constant} ->
        range = Domain.add(range, product.range, state.precision)

        grid =
          if Domain.within_grid?(product.range, grid, state.precision) and
               Domain.within_grid?(range, grid, state.precision),
             do: grid

        constant =
          if constant != nil and product.constant != nil,
            do: Domain.safe(fn -> Number.cast(constant + product.constant, state.precision) end)

        {range, grid, constant}
      end)

    {terms, bias, %{range: range, grid: grid, constant: constant, type: :number}}
  end

  defp affine(terms, bias, state) do
    {terms, bias, info} = form(terms, bias, state)
    emit({:affine, terms, bias}, info, state)
  end

  defp reglu(terms, bias, gates, gate_bias, state) do
    {terms, bias, left} = form(terms, bias, state)
    {gates, gate_bias, gate} = form(gates, gate_bias, state)
    range = Domain.multiply(left.range, Domain.relu(gate.range), state.precision)
    grid = Domain.multiply_grid(left.grid, gate.grid)
    grid = if Domain.within_grid?(range, grid, state.precision), do: grid

    constant =
      if left.constant != nil and gate.constant != nil,
        do:
          Domain.safe(fn ->
            Number.cast(left.constant * max(0.0, gate.constant), state.precision)
          end)

    emit(
      {:reglu, terms, bias, gates, gate_bias},
      %{range: range, grid: grid, constant: constant, type: :number},
      state
    )
  end

  defp relu(terms, bias, state), do: reglu([], 1.0, terms, bias, state)

  defp constant(value, state) do
    value = Number.cast(value, state.precision)

    case Map.fetch(state.constants, value) do
      {:ok, id} ->
        {id, state}

      :error ->
        {id, state} = affine([], value, state)
        {id, %{state | constants: Map.put(state.constants, value, id)}}
    end
  end

  defp lower_operation({:constant, value}, state) when is_number(value),
    do: constant(value, state)

  defp lower_operation({:constant, value}, state) when is_boolean(value),
    do: tag(affine([], if(value, do: 1.0, else: 0.0), state), %{type: :boolean})

  defp lower_operation({:constant, nil}, _),
    do:
      raise(
        ArgumentError,
        "primitive circuits require numeric or boolean leaves; use explicit if/else values instead of nil"
      )

  defp lower_operation({:select, c, yes, no}, state) do
    {bit, state} = truth_bit(state.mapped[c], state)
    select(bit, state.mapped[yes], state.mapped[no], state)
  end

  defp lower_operation({op, a}, state) do
    a = state.mapped[a]

    case op do
      :negate ->
        affine([{a, -1.0}], 0.0, state)

      :relu ->
        relu([{a, 1.0}], 0.0, state)

      :abs ->
        {positive, state} = relu([{a, 1.0}], 0.0, state)
        {negative, state} = relu([{a, -1.0}], 0.0, state)
        affine([{positive, 1.0}, {negative, 1.0}], 0.0, state)

      :not ->
        tag(affine([{a, -1.0}], 1.0, state), %{type: :boolean, range: {0.0, 1.0}, grid: 1.0})
    end
  end

  defp lower_operation({op, a, b}, state) do
    a = state.mapped[a]
    b = state.mapped[b]

    case op do
      :add ->
        affine([{a, 1.0}, {b, 1.0}], 0.0, state)

      :subtract ->
        affine([{a, 1.0}, {b, -1.0}], 0.0, state)

      :multiply ->
        multiply(a, b, state)

      :divide ->
        divide(a, b, state)

      :and ->
        tag(relu([{a, 1.0}, {b, 1.0}], -1.0, state), %{
          type: :boolean,
          range: {0.0, 1.0},
          grid: 1.0
        })

      :or ->
        {neither, state} = relu([{a, -1.0}, {b, -1.0}], 1.0, state)

        tag(affine([{neither, -1.0}], 1.0, state), %{type: :boolean, range: {0.0, 1.0}, grid: 1.0})

      op when op in [:min, :max] ->
        extremum(op, a, b, state)

      op when op in [:>, :>=, :<, :<=, :==, :!=, :===, :!==] ->
        compare(op, a, b, state)
    end
  end

  defp multiply(a, b, state) do
    result =
      cond do
        Domain.nonnegative?(state.info[b].range) ->
          reglu([{a, 1.0}], 0.0, [{b, 1.0}], 0.0, state)

        Domain.nonnegative?(state.info[a].range) ->
          reglu([{b, 1.0}], 0.0, [{a, 1.0}], 0.0, state)

        true ->
          {positive, state} = reglu([{a, 1.0}], 0.0, [{b, 1.0}], 0.0, state)
          {negative, state} = reglu([{a, 1.0}], 0.0, [{b, -1.0}], 0.0, state)
          affine([{positive, 1.0}, {negative, -1.0}], 0.0, state)
      end

    result
    |> tag(%{range: Domain.multiply(state.info[a].range, state.info[b].range, state.precision)})
    |> macro_rule({:multiply, a, b})
  end

  defp divide(a, b, state) do
    value = state.info[b].constant

    if value == nil or value == 0,
      do:
        raise(
          ArgumentError,
          "primitive division requires a nonzero compile-time constant divisor"
        )

    if state.division == :exact and Domain.grid(value) != abs(value),
      do:
        raise(
          ArgumentError,
          "exact primitive division requires a power-of-two divisor; use division: :reciprocal to opt into rounded reciprocal multiplication"
        )

    affine([{a, Number.cast(1.0 / value, state.precision)}], 0.0, state)
  end

  defp truth_bit(wire, state) do
    if state.info[wire].type == :boolean, do: {wire, state}, else: constant(1.0, state)
  end

  defp select(bit, yes, no, state) do
    {on, next} = reglu([{yes, 1.0}], 0.0, [{bit, 1.0}], 0.0, state)
    {off, next} = reglu([{no, 1.0}], 0.0, [{bit, -1.0}], 1.0, next)

    affine([{on, 1.0}, {off, 1.0}], 0.0, next)
    |> tag(%{
      type: state.info[yes].type,
      range: Domain.union(state.info[yes].range, state.info[no].range),
      grid: Domain.combine(state.info[yes].grid, state.info[no].grid)
    })
    |> macro_rule({:select, bit, yes, no})
  end

  defp extremum(kind, a, b, state) do
    # A zero operand gives an exact one-gate construction even for open ranges.
    result =
      cond do
        state.info[a].constant == 0.0 or state.info[b].constant == 0.0 ->
          x = if state.info[a].constant == 0.0, do: b, else: a

          if kind == :max do
            relu([{x, 1.0}], 0.0, state)
          else
            {negative, state} = relu([{x, -1.0}], 0.0, state)
            affine([{negative, -1.0}], 0.0, state)
          end

        true ->
          {bit, next} = compare(if(kind == :max, do: :>=, else: :<=), a, b, state)
          select(bit, a, b, next)
      end

    macro_rule(result, {:extremum, kind, a, b})
  end

  defp compare(op, a, b, state) do
    {test, a, b, invert} =
      case op do
        :> -> {:greater, a, b, false}
        :>= -> {:greater_equal, a, b, false}
        :< -> {:greater, b, a, false}
        :<= -> {:greater_equal, b, a, false}
        op when op in [:==, :===] -> {:equal, a, b, false}
        _ -> {:equal, a, b, true}
      end

    left = state.info[a]
    right = state.info[b]

    bound =
      Domain.add(
        left.range,
        Domain.multiply(right.range, {-1.0, -1.0}, state.precision),
        state.precision
      )

    if bound == nil,
      do:
        raise(
          ArgumentError,
          "primitive comparisons require finite operand ranges with a finite difference; declare input and parameter ranges"
        )

    grid = Domain.combine(left.grid, right.grid)
    grid = if grid == :zero, do: 1.0, else: grid

    safe_grid =
      grid != nil and
        Domain.safe(fn ->
          (Domain.max_abs(left.range) + Domain.max_abs(right.range)) / grid + 1 <=
            Domain.power(Domain.bits(state.precision)) and
            Number.cast(1.0 / grid, state.precision) > 0
        end) == true

    {bit, state} =
      if safe_grid do
        state = %{state | comparisons: Map.update!(state.comparisons, :grid, &(&1 + 1))}
        integer_bit(test, [{a, 1.0 / grid}, {b, -1.0 / grid}], state)
      else
        state = %{state | comparisons: Map.update!(state.comparisons, :cascade, &(&1 + 1))}
        {difference, state} = affine([{a, 1.0}, {b, -1.0}], 0.0, state)
        cascade(test, difference, Domain.max_abs(bound), state)
      end

    result = if invert, do: affine([{bit, -1.0}], 1.0, state), else: {bit, state}

    result
    |> tag(%{type: :boolean, range: {0.0, 1.0}, grid: 1.0, constant: nil})
    |> macro_rule(:stop)
  end

  defp integer_bit(test, terms, state) do
    negative = Enum.map(terms, fn {wire, weight} -> {wire, -weight} end)

    case test do
      :greater ->
        {ramp, state} = relu(negative, 1.0, state)
        relu([{ramp, -1.0}], 1.0, state)

      :greater_equal ->
        {ramp, state} = relu(negative, 0.0, state)
        relu([{ramp, -1.0}], 1.0, state)

      :equal ->
        {pos, state} = relu(terms, 0.0, state)
        {neg, state} = relu(negative, 0.0, state)
        relu([{pos, -1.0}, {neg, -1.0}], 1.0, state)
    end
  end

  # Adapted from Cheng Lou's planSignCascade / emitCascadeCompare (MIT).
  # Each clamp stage preserves tiny differences while shrinking large ones;
  # scaling down first would erase subnormals. See LICENSE-UPSTREAM.
  defp cascade(test, difference, bound, state) do
    exponent = Domain.ceiling_exponent(max(bound, Domain.power(-Domain.lift(state.precision))))
    stages = plan(exponent, 0, state.precision, [])
    {clamps, [{scale, _limit}]} = Enum.split(stages, -1)

    if clamps == [] do
      integer_bit(test, [{difference, scale}], state)
    else
      case test do
        :greater ->
          {positive, state} = clamp_stages(difference, 1.0, clamps, state)
          {not_greater, state} = relu([{positive, -scale}], 1.0, state)
          relu([{not_greater, -1.0}], 1.0, state)

        :greater_equal ->
          {negative, state} = clamp_stages(difference, -1.0, clamps, state)
          relu([{negative, -scale}], 1.0, state)

        :equal ->
          {negative, state} = clamp_stages(difference, -1.0, clamps, state)
          {positive, state} = clamp_stages(difference, 1.0, clamps, state)
          relu([{negative, -scale}, {positive, -scale}], 1.0, state)
      end
    end
  end

  defp plan(limit, gain, precision, stages) do
    if stages != [] and limit <= 0 and gain >= Domain.lift(precision) do
      Enum.reverse(stages)
    else
      next_limit = max(0, limit - Domain.bits(precision))
      room = min(Domain.max_exponent(precision), Domain.bits(precision) + next_limit - limit)
      scale = min(room, max(0, Domain.lift(precision) - gain))

      plan(next_limit, gain + scale, precision, [
        {Domain.power(scale), Domain.power(next_limit)} | stages
      ])
    end
  end

  defp clamp_stages(difference, sign, stages, state) do
    stages
    |> Enum.with_index()
    |> Enum.reduce({difference, state}, fn {{scale, limit}, index}, {previous, state} ->
      terms = [{previous, scale * if(index == 0, do: sign, else: 1.0)}]

      {scaled_terms, state} =
        if index == 0 do
          {positive, state} = relu(terms, 0.0, state)
          {[{positive, 1.0}], state}
        else
          {terms, state}
        end

      # Clamp the already positive first projection. Subtracting a limit from
      # a negative value near -max_float could overflow before ReLU clips it.
      excess_terms = if index == 0, do: scaled_terms, else: terms
      {excess, state} = relu(excess_terms, -limit, state)
      tag(affine(scaled_terms ++ [{excess, -1.0}], 0.0, state), %{range: {0.0, limit}})
    end)
  end

  defp layers(nodes) do
    {_, layers} =
      nodes
      |> Tuple.to_list()
      |> Enum.with_index()
      |> Enum.reduce({%{}, %{}}, fn {op, wire}, {depths, layers} ->
        case op do
          {:source, _} ->
            {Map.put(depths, wire, 0), layers}

          _ ->
            terms =
              case op do
                {:affine, terms, _} -> terms
                {:reglu, terms, _, gates, _} -> terms ++ gates
              end

            depth =
              1 +
                Enum.reduce(terms, 0, fn {parent, _}, depth ->
                  max(depth, Map.fetch!(depths, parent))
                end)

            {Map.put(depths, wire, depth), Map.update(layers, depth, [wire], &[wire | &1])}
        end
      end)

    layers |> Enum.sort() |> Enum.map(fn {_, wires} -> Enum.reverse(wires) end)
  end
end
