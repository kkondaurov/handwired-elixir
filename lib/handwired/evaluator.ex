defmodule Handwired.Evaluator do
  @moduledoc false
  alias Handwired.{Circuit, Number}

  def run(%Circuit{} = circuit, inputs) do
    {result, _context} = forward(circuit, inputs)
    result
  end

  def value_and_grad(circuit, inputs, opts \\ []) do
    if Keyword.keys(opts) -- [:wrt] != [], do: raise(ArgumentError, "unknown gradient option")

    output =
      case circuit.output do
        {:wire, wire, :number} -> wire
        _ -> raise ArgumentError, "gradients require a scalar numeric output"
      end

    names = gradient_names(circuit, Keyword.get(opts, :wrt, :parameters))
    {value, context} = forward(circuit, inputs)

    gradients =
      Enum.reduce(context.tape, %{output => 1.0}, fn wire, gradients ->
        upstream = Map.get(gradients, wire, 0.0)

        if upstream == 0.0 do
          gradients
        else
          operation = elem(circuit.nodes, wire)

          Enum.reduce(derivatives(wire, operation, context), gradients, fn {parent, derivative},
                                                                           gradients ->
            contribution = Number.cast(upstream * derivative, circuit.precision)

            Map.update(
              gradients,
              parent,
              contribution,
              &Number.cast(&1 + contribution, circuit.precision)
            )
          end)
        end
      end)

    {value,
     Map.new(names, fn name -> {name, Map.get(gradients, circuit.sources[name].wire, 0.0)} end)}
  end

  defp gradient_names(circuit, :parameters), do: Map.keys(circuit.parameters)

  defp gradient_names(circuit, :inputs),
    do: for({name, %{role: :input}} <- circuit.sources, do: name)

  defp gradient_names(circuit, :all), do: Map.keys(circuit.sources)

  defp gradient_names(circuit, names) when is_list(names) do
    if names -- Map.keys(circuit.sources) != [],
      do: raise(ArgumentError, "unknown gradient source")

    Enum.uniq(names)
  end

  defp gradient_names(_, _),
    do: raise(ArgumentError, "wrt must be :parameters, :inputs, :all, or source names")

  defp forward(circuit, inputs) do
    inputs = Circuit.named(inputs)
    required = for {name, %{role: :input}} <- circuit.sources, do: name

    if Enum.sort(Map.keys(inputs)) != Enum.sort(required),
      do:
        raise(
          ArgumentError,
          "expected inputs #{inspect(Enum.sort(required))}, got #{inspect(Map.keys(inputs))}"
        )

    sources =
      Enum.reduce(inputs, circuit.parameters, fn {name, value}, sources ->
        spec = circuit.sources[name]
        value = Number.validate(value, spec.range, circuit.precision, name)
        Handwired.Domain.validate_grid(value, Map.get(spec, :grid), name)
        Map.put(sources, name, value)
      end)

    context = %{circuit: circuit, sources: sources, values: %{}, tape: []}
    # A lowered network executes its arithmetic instructions eagerly. There are
    # no semantic branch callbacks: both arms become ordinary gate dependencies.
    context =
      if circuit.backend == :primitives do
        circuit.nodes
        |> Tuple.to_list()
        |> Enum.with_index()
        |> Enum.reduce(context, fn {_, wire}, context -> elem(evaluate(wire, context), 1) end)
      else
        context
      end

    output(circuit.output, context)
  end

  defp output({:wire, wire, :boolean}, %{circuit: %{backend: :primitives}} = context) do
    {value, context} = evaluate(wire, context)

    unless value == 0.0 or value == 1.0,
      do: raise(ArgumentError, "primitive boolean output is not an exact bit: #{value}")

    {value == 1.0, context}
  end

  defp output({:wire, wire, _type}, context), do: evaluate(wire, context)

  defp output({kind, entries}, context) when kind in [:tuple, :list] do
    {values, context} = Enum.map_reduce(entries, context, &output/2)
    {if(kind == :tuple, do: List.to_tuple(values), else: values), context}
  end

  defp output({:map, entries}, context) do
    {values, context} =
      Enum.map_reduce(Enum.sort(entries), context, fn {key, shape}, context ->
        {value, context} = output(shape, context)
        {{key, value}, context}
      end)

    {Map.new(values), context}
  end

  defp evaluate(wire, context) do
    case Map.fetch(context.values, wire) do
      {:ok, value} ->
        {value, context}

      :error ->
        operation = elem(context.circuit.nodes, wire)
        {value, context} = operation(operation, context)

        value =
          if is_number(value), do: Number.cast(value, context.circuit.precision), else: value

        {value,
         %{context | values: Map.put(context.values, wire, value), tape: [wire | context.tape]}}
    end
  rescue
    error in [ArithmeticError, ArgumentError] ->
      raise ArgumentError,
            "node #{wire} (#{inspect(elem(context.circuit.nodes, wire))}): #{Exception.message(error)}"
  end

  defp operation({:source, name}, context), do: {Map.fetch!(context.sources, name), context}
  defp operation({:constant, value}, context), do: {value, context}

  defp operation({:affine, terms, bias}, context),
    do: {affine_value(terms, bias, context), context}

  defp operation({:reglu, terms, bias, gates, gate_bias}, context) do
    left = affine_value(terms, bias, context)
    gate = affine_value(gates, gate_bias, context)
    {left * max(0.0, gate), context}
  end

  defp operation({:select, condition, yes, no}, context) do
    {condition, context} = evaluate(condition, context)
    # Preserve branch laziness: inactive divisions and other invalid arithmetic
    # never execute, and their wires are absent from the differentiation tape.
    evaluate(if(truthy?(condition), do: yes, else: no), context)
  end

  defp operation({operator, a, b}, context) when operator in [:and, :or] do
    {a, context} = evaluate(a, context)

    case {operator, a} do
      {:and, false} -> {false, context}
      {:or, true} -> {true, context}
      {_, value} when is_boolean(value) -> evaluate(b, context)
      _ -> raise ArgumentError, "boolean operation received a non-boolean value"
    end
  end

  defp operation({operator, a, b}, context) do
    {a, context} = evaluate(a, context)
    {b, context} = evaluate(b, context)

    value =
      case operator do
        :add -> a + b
        :subtract -> a - b
        :multiply -> a * b
        :divide -> a / b
        :min -> min(a, b)
        :max -> max(a, b)
        :> -> a > b
        :>= -> a >= b
        :< -> a < b
        :<= -> a <= b
        :== -> a == b
        :!= -> a != b
        :=== -> a === b
        :!== -> a !== b
      end

    {value, context}
  end

  defp operation({operator, a}, context) do
    {a, context} = evaluate(a, context)

    value =
      case operator do
        :negate -> -a
        :abs -> abs(a)
        :relu -> max(0.0, a)
        :not -> not a
      end

    {value, context}
  end

  defp affine_value(terms, bias, context) do
    precision = context.circuit.precision

    Enum.reduce(terms, bias, fn {wire, coefficient}, value ->
      product = Number.cast(coefficient * Map.fetch!(context.values, wire), precision)
      Number.cast(value + product, precision)
    end)
  end

  defp derivatives(wire, operation, context) do
    values = context.values

    case Map.get(context.circuit.macros, wire) do
      :stop ->
        []

      {:multiply, a, b} ->
        [{a, values[b]}, {b, values[a]}]

      {:select, bit, yes, no} ->
        [{if(values[bit] == 1.0, do: yes, else: no), 1.0}]

      {:extremum, kind, a, b} ->
        first = if kind == :max, do: values[a] >= values[b], else: values[a] <= values[b]
        [{if(first, do: a, else: b), 1.0}]

      nil ->
        primitive_or_semantic_partials(operation, context)
    end
  end

  defp primitive_or_semantic_partials({:affine, terms, _}, _), do: terms

  defp primitive_or_semantic_partials({:reglu, terms, bias, gates, gate_bias}, context) do
    left = affine_value(terms, bias, context)
    gate = affine_value(gates, gate_bias, context)

    Enum.map(terms, fn {wire, coefficient} -> {wire, coefficient * max(0.0, gate)} end) ++
      Enum.map(gates, fn {wire, coefficient} ->
        {wire, if(gate > 0.0, do: coefficient * left, else: 0.0)}
      end)
  end

  defp primitive_or_semantic_partials(operation, context), do: partials(operation, context.values)

  defp partials({:add, a, b}, _), do: [{a, 1.0}, {b, 1.0}]
  defp partials({:subtract, a, b}, _), do: [{a, 1.0}, {b, -1.0}]
  defp partials({:multiply, a, b}, values), do: [{a, values[b]}, {b, values[a]}]

  defp partials({:divide, a, b}, values),
    do: [{a, 1.0 / values[b]}, {b, -values[a] / (values[b] * values[b])}]

  defp partials({:negate, a}, _), do: [{a, -1.0}]

  defp partials({:abs, a}, values),
    do: [
      {a,
       cond do
         values[a] > 0 -> 1.0
         values[a] < 0 -> -1.0
         true -> 0.0
       end}
    ]

  defp partials({:relu, a}, values), do: [{a, if(values[a] > 0.0, do: 1.0, else: 0.0)}]
  defp partials({:min, a, b}, values), do: [{if(values[a] <= values[b], do: a, else: b), 1.0}]
  defp partials({:max, a, b}, values), do: [{if(values[a] >= values[b], do: a, else: b), 1.0}]

  defp partials({:select, condition, yes, no}, values),
    do: [{if(truthy?(values[condition]), do: yes, else: no), 1.0}]

  # Sources, constants and discrete decisions do not have differentiable parents.
  defp partials(_, _), do: []
  defp truthy?(value), do: value != false and value != nil
end
