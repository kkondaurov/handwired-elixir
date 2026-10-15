defmodule Handwired.Circuit do
  @moduledoc """
  An immutable numeric program. `nodes` is a topologically ordered tuple;
  `output` describes scalar, tuple, list, or atom-keyed map results.
  Parameters are stored separately so fitting never rebuilds the graph.
  """
  defstruct [
    :name,
    :nodes,
    :output,
    :sources,
    :parameters,
    precision: :f64,
    backend: :semantic,
    macros: %{},
    layers: [],
    lowering: %{}
  ]

  def put_parameters(circuit, values) do
    values = named(values)

    Enum.each(Map.keys(values), fn name ->
      unless Map.has_key?(circuit.parameters, name),
        do: raise(ArgumentError, "unknown parameter: #{inspect(name)}")
    end)

    parameters =
      Enum.reduce(values, circuit.parameters, fn {name, value}, acc ->
        spec = Map.fetch!(circuit.sources, name)
        number = Handwired.Number.validate(value, spec.range, circuit.precision, name)
        Handwired.Domain.validate_grid(number, Map.get(spec, :grid), name)
        Map.put(acc, name, number)
      end)

    %{circuit | parameters: parameters}
  end

  def named(values) when is_map(values), do: values

  def named(values) when is_list(values) do
    unless Keyword.keyword?(values) and
             length(Keyword.keys(values)) == length(Enum.uniq(Keyword.keys(values))),
           do: raise(ArgumentError, "expected unique keyword keys")

    Map.new(values)
  end

  def named(_), do: raise(ArgumentError, "expected a map or keyword list")
end
