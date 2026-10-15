defmodule Handwired.Compiler do
  @moduledoc false
  alias Handwired.{Circuit, Number}

  def compile(fun, opts) when is_function(fun) do
    info = Map.new(Function.info(fun))

    unless info.type == :external and Code.ensure_loaded?(info.module) and
             function_exported?(info.module, :__handwired_definition__, 2),
           do:
             raise(
               ArgumentError,
               "compile expects an external capture such as &MyModule.my_defwire/2"
             )

    unknown = Keyword.keys(opts) -- [:inputs, :parameters, :parameter_ranges, :precision]
    if unknown != [], do: raise(ArgumentError, "unknown compile options: #{inspect(unknown)}")
    {args, body} = info.module.__handwired_definition__(info.name, info.arity)
    inputs = opts |> Keyword.get(:inputs, []) |> Circuit.named()
    parameters = opts |> Keyword.get(:parameters, []) |> Circuit.named()
    ranges = opts |> Keyword.get(:parameter_ranges, []) |> Circuit.named()
    precision = Keyword.get(opts, :precision, :f64)
    unless precision in [:f32, :f64], do: raise(ArgumentError, "precision must be :f32 or :f64")

    overlap = Enum.filter(Map.keys(inputs), &Map.has_key?(parameters, &1))

    if overlap != [],
      do: raise(ArgumentError, "inputs and parameters overlap: #{inspect(overlap)}")

    supplied = Map.keys(inputs) ++ Map.keys(parameters)

    if Enum.sort(supplied) != Enum.sort(args),
      do:
        raise(
          ArgumentError,
          "declare every argument exactly once; expected #{inspect(args)}, got #{inspect(supplied)}"
        )

    if Map.keys(ranges) -- Map.keys(parameters) != [],
      do: raise(ArgumentError, "parameter_ranges contains an unknown parameter")

    state = %{nodes: [], next: 0, env: %{}, precision: precision}

    {state, sources} =
      Enum.reduce(args, {state, %{}}, fn name, {state, sources} ->
        role = if Map.has_key?(inputs, name), do: :input, else: :parameter

        range =
          Number.range(if role == :input, do: inputs[name], else: ranges[name])
          |> Number.quantize_range(precision)

        {wire, state} = node({:source, name}, :number, state)
        spec = %{role: role, range: range, wire: elem(wire, 1)}
        {%{state | env: Map.put(state.env, name, wire)}, Map.put(sources, name, spec)}
      end)

    {output, state} = expr(body, state)

    circuit = %Circuit{
      name: {info.module, info.name, info.arity},
      nodes: state.nodes |> Enum.reverse() |> List.to_tuple(),
      output: output,
      sources: sources,
      parameters: parameters,
      precision: precision
    }

    Circuit.put_parameters(circuit, parameters)
  end

  defp node(operation, type, state) do
    {{:wire, state.next, type}, %{state | nodes: [operation | state.nodes], next: state.next + 1}}
  end

  defp expr(value, state) when is_number(value),
    do: node({:constant, Number.cast(value, state.precision)}, :number, state)

  defp expr(value, state) when is_boolean(value) or is_nil(value),
    do: node({:constant, value}, :boolean, state)

  defp expr({:__block__, _, expressions}, state) do
    Enum.reduce(expressions, {nil, state}, fn expression, {_, state} ->
      expr(expression, state)
    end)
  end

  defp expr({:=, _, [{name, _, context}, value]}, state)
       when is_atom(name) and is_atom(context) do
    {wire, state} = expr(value, state)
    {wire, %{state | env: Map.put(state.env, name, wire)}}
  end

  defp expr({name, _, context} = ast, state) when is_atom(name) and is_atom(context) do
    case Map.fetch(state.env, name) do
      {:ok, wire} -> {wire, state}
      :error -> unsupported(ast, "unbound variable")
    end
  end

  defp expr({:if, _, [condition, clauses]}, state) do
    {condition, state} = expr(condition, state)
    condition = leaf(condition)
    env = state.env
    {yes, state} = expr(Keyword.fetch!(clauses, :do), state)
    {no, state} = expr(Keyword.get(clauses, :else), %{state | env: env})
    select(condition, yes, no, %{state | env: env})
  end

  defp expr({operator, _, [a, b]}, state)
       when operator in [:+, :-, :*, :/, :min, :max, :>, :>=, :<, :<=, :==, :!=, :===, :!==] do
    {a, state} = expr(a, state)
    {b, state} = expr(b, state)

    operation =
      case operator do
        :+ -> :add
        :- -> :subtract
        :* -> :multiply
        :/ -> :divide
        other -> other
      end

    type = if operator in [:>, :>=, :<, :<=, :==, :!=, :===, :!==], do: :boolean, else: :number
    node({operation, numeric(a), numeric(b)}, type, state)
  end

  defp expr({operator, _, [a]}, state) when operator in [:-, :+, :abs, :relu] do
    {a, state} = expr(a, state)
    numeric(a)

    case operator do
      :+ -> {a, state}
      :- -> node({:negate, numeric(a)}, :number, state)
      other -> node({other, numeric(a)}, :number, state)
    end
  end

  defp expr({operator, _, [a, b]}, state) when operator in [:and, :or] do
    {a, state} = expr(a, state)
    {b, state} = expr(b, state)
    node({operator, boolean(a), boolean(b)}, :boolean, state)
  end

  defp expr({:not, _, [a]}, state) do
    {a, state} = expr(a, state)
    node({:not, boolean(a)}, :boolean, state)
  end

  defp expr({:{}, _, entries}, state), do: container(:tuple, entries, state)
  defp expr({a, b}, state), do: container(:tuple, [a, b], state)

  defp expr({:%{}, _, entries} = ast, state) do
    unless Enum.all?(entries, &match?({key, _} when is_atom(key), &1)),
      do: unsupported(ast, "maps must have fixed atom keys")

    {fields, state} =
      Enum.map_reduce(entries, state, fn {key, value}, state ->
        {value, state} = expr(value, state)
        {{key, value}, state}
      end)

    {{:map, Map.new(fields)}, state}
  end

  defp expr(entries, state) when is_list(entries), do: container(:list, entries, state)
  defp expr(ast, _state), do: unsupported(ast, "unsupported expression")

  defp container(kind, entries, state) do
    {entries, state} = Enum.map_reduce(entries, state, &expr/2)
    {{kind, entries}, state}
  end

  defp select(condition, {:wire, a, type}, {:wire, b, type}, state),
    do: node({:select, condition, a, b}, type, state)

  defp select(condition, {kind, a}, {kind, b}, state)
       when kind in [:tuple, :list] and length(a) == length(b) do
    {entries, state} =
      Enum.map_reduce(Enum.zip(a, b), state, fn {a, b}, state ->
        select(condition, a, b, state)
      end)

    {{kind, entries}, state}
  end

  defp select(condition, {:map, a}, {:map, b}, state) when map_size(a) == map_size(b) do
    unless Enum.sort(Map.keys(a)) == Enum.sort(Map.keys(b)),
      do: raise(ArgumentError, "if branches must have the same output structure")

    {entries, state} =
      Enum.map_reduce(Enum.sort(Map.keys(a)), state, fn key, state ->
        {entry, state} = select(condition, a[key], b[key], state)
        {{key, entry}, state}
      end)

    {{:map, Map.new(entries)}, state}
  end

  defp select(_, _, _, _),
    do: raise(ArgumentError, "if branches must have matching types and output structures")

  defp leaf({:wire, id, _}), do: id
  defp leaf(_), do: raise(ArgumentError, "expected a scalar expression")
  defp numeric({:wire, id, :number}), do: id
  defp numeric(_), do: raise(ArgumentError, "expected a numeric scalar")
  defp boolean({:wire, id, :boolean}), do: id
  defp boolean(_), do: raise(ArgumentError, "and/or/not require boolean expressions")
  defp unsupported(ast, reason), do: raise(ArgumentError, "#{reason}: #{Macro.to_string(ast)}")
end
