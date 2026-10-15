defmodule Handwired do
  @moduledoc """
  Author numeric functions, compile them to an inspectable graph, and fit their
  parameters. Everything executes in Elixir; there are no runtime dependencies.

  `use Handwired` imports `defwire/2` and `relu/1`. A `defwire` definition also
  creates an ordinary Elixir function, useful as an independent reference.
  Compile with an external function capture, explicit input ranges, and initial
  parameter values. For incremental feedback, persistent optimizer state and
  replay/undo, see `Handwired.Learner`. See the README and executable examples
  for complete uses.
  """

  defmacro __using__(_opts) do
    quote do
      import Handwired, only: [defwire: 2, relu: 1]
      Module.register_attribute(__MODULE__, :handwired_definitions, accumulate: true)
      @before_compile Handwired
    end
  end

  defmacro defwire({name, _, args} = head, do: body) when is_atom(name) do
    args = args || []

    unless Enum.all?(args, &match?({var, _, context} when is_atom(var) and is_atom(context), &1)) do
      raise ArgumentError, "defwire arguments must be simple named variables"
    end

    names = Enum.map(args, &elem(&1, 0))

    if length(Enum.uniq(names)) != length(names) or :_ in names do
      raise ArgumentError, "defwire arguments must have unique, non-placeholder names"
    end

    quote do
      @handwired_definitions unquote(Macro.escape({name, length(args), names, body}))
      def unquote(head), do: unquote(body)
    end
  end

  defmacro defwire(_, _) do
    raise ArgumentError, "defwire expects a single function clause with named arguments"
  end

  defmacro __before_compile__(env) do
    definitions = Module.get_attribute(env.module, :handwired_definitions)
    keys = Enum.map(definitions, fn {name, arity, _, _} -> {name, arity} end)

    if length(keys) != length(Enum.uniq(keys)) do
      raise ArgumentError, "multiple defwire clauses for the same function are not supported yet"
    end

    clauses =
      Enum.map(definitions, fn {name, arity, args, body} ->
        quote do
          def __handwired_definition__(unquote(name), unquote(arity)),
            do: unquote(Macro.escape({args, body}))
        end
      end)

    quote do
      unquote_splicing(clauses)
      def __handwired_functions__, do: unquote(Macro.escape(Enum.reverse(keys)))

      def __handwired_definition__(name, arity),
        do: raise(ArgumentError, "no defwire definition for #{name}/#{arity}")
    end
  end

  @doc "ReLU, also callable in ordinary Elixir. Its derivative at zero is zero."
  def relu(value) when is_number(value), do: max(0.0, value)

  @doc "Compile a defwire function capture. Input ranges are `{min, max}` or integer ranges."
  defdelegate compile(fun, opts), to: Handwired.Compiler

  @doc "Lower a compiled circuit into affine/ReGLU primitives. Options: :grids and :division."
  defdelegate lower(circuit, opts \\ []), to: Handwired.Lowering

  @doc "Inspect primitive instruction, weight, layer, and comparison counts."
  defdelegate stats(circuit), to: Handwired.Lowering

  @doc "Run a circuit with a map or keyword list of named numeric inputs."
  defdelegate run(circuit, inputs), to: Handwired.Evaluator

  @doc "Return `{scalar_output, gradients}`. `:wrt` defaults to `:parameters`."
  defdelegate value_and_grad(circuit, inputs, opts \\ []), to: Handwired.Evaluator

  @doc "Return gradients of a scalar output with respect to parameters or named sources."
  def grad(circuit, inputs, opts \\ []), do: elem(value_and_grad(circuit, inputs, opts), 1)

  @doc "Fit scalar outputs to `{inputs, target}` examples using mean squared error."
  defdelegate fit(circuit, examples, opts \\ []), to: Handwired.Trainer

  @doc "Replace named parameter values, validating their ranges and precision."
  defdelegate put_parameters(circuit, values), to: Handwired.Circuit
end
