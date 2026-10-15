defmodule PrimitiveExample do
  use Handwired
  defwire(decision(x), do: if(x > 4, do: 10, else: 12))
end

semantic = Handwired.compile(&PrimitiveExample.decision/1, inputs: [x: -20..20], precision: :f32)
integer_net = Handwired.lower(semantic, grids: [x: 1])
float_net = Handwired.lower(semantic)

IO.puts("Elixir if x > 4 -> an arithmetic network, with no comparison/branch instructions")
IO.inspect(Handwired.stats(integer_net), label: "Integer grid network")

for {operation, wire} <- Enum.with_index(Tuple.to_list(integer_net.nodes)) do
  unless elem(operation, 0) in [:source, :affine, :reglu], do: raise("unexpected primitive")
  IO.inspect(operation, label: "wire #{wire}")
end

for x <- [3, 4, 5, 6] do
  result = Handwired.run(integer_net, x: x)
  if result != PrimitiveExample.decision(x), do: raise("grid decision failed")
  IO.inspect(%{x: x, result: result}, label: "Integer decision")
end

IO.inspect(Handwired.stats(float_net), label: "Floating-point network")

for x <- [3.999999523162842, 4.0, 4.000000476837158, 4.5] do
  result = Handwired.run(float_net, x: x)
  if result != PrimitiveExample.decision(x), do: raise("floating decision failed")
  IO.inspect(%{x: x, result: result}, label: "Floating decision")
end

Code.require_file("envelope.exs", __DIR__)
Code.require_file("controller.exs", __DIR__)
IO.puts("PASS: primitive-only decisions, parameter learning, and retained state.")
