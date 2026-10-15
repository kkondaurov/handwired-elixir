# Handwired Elixir

Author numeric behavior in Elixir, compile it into an inspectable circuit, run it,
differentiate it, and fit selected parameters to examples.

This is a native implementation of the programming model and affine/ReGLU
arithmetic networks of
[Cheng Lou's Handwired](https://github.com/chenglou/handwired). The compiler,
lowering pass, evaluator, reverse-mode differentiation, Adam/SGD training, and state-machine
runner are implemented in Elixir. **No JavaScript, Node, Bun, Nx, NIF, or package
dependencies are required.** It runs on Elixir 1.14+ / Erlang OTP 25+.

## Install from GitHub

Add this dependency to your application's `mix.exs`:

```elixir
{:handwired, github: "kkondaurov/handwired-elixir", tag: "v0.2.1"}
```

Then run `mix deps.get`. The library has no Hex dependencies.

## Run it

```sh
git clone https://github.com/kkondaurov/handwired-elixir.git
cd handwired-elixir
mix test --warnings-as-errors
mix run examples/primitives.exs
# Or run the two applications separately:
mix run examples/envelope.exs
mix run examples/controller.exs
```

The primitive demo prints a network for `if x > 4`, checks integer and fractional
decisions, then runs the envelope and controller on primitive graphs.
All examples contain executable assertions and exit with an error if their
end-to-end checks fail. The envelope learns peak and decay from samples and
checks unseen times. The controller retains position and velocity across 1,000
events, changes its target halfway through, and checks every transition against
the original Elixir function.

## Write, compile, run, learn

```elixir
defmodule Envelope do
  use Handwired

  defwire amplitude(t, peak, decay) do
    max(0.0, peak - decay * t)
  end
end

circuit = Handwired.compile(&Envelope.amplitude/3,
  inputs: [t: {0.0, 1.0}],
  parameters: [peak: 0.6, decay: 0.5],
  parameter_ranges: [peak: {0.0, 2.0}, decay: {0.0, 5.0}])
  |> Handwired.lower()

Handwired.run(circuit, %{t: 0.2})
# => 0.5

Handwired.value_and_grad(circuit, %{t: 0.2})
# => {0.5, %{peak: 1.0, decay: -0.2}}

Handwired.grad(circuit, %{t: 0.2}, wrt: :all)
# => %{t: -0.5, peak: 1.0, decay: -0.2}

examples = [
  {%{t: 0.0}, 1.0},
  {%{t: 0.2}, 0.6},
  {%{t: 0.4}, 0.2}
]

fit = Handwired.fit(circuit, examples, steps: 2000)
fit.circuit.parameters
# approximately %{peak: 1.0, decay: 2.0}

Handwired.run(fit.circuit, %{t: 0.3})
# approximately 0.4
```

`defwire` also defines an ordinary Elixir function, so
`Envelope.amplitude(0.2, 0.6, 0.5)` is a separately executable reference.
The compiler reads the stored quoted syntax; it does not sample inputs or trace
one execution. Both conditional branches are represented in the graph.

Each argument must be declared exactly once as an input or a parameter. Inputs
and parameter values can be maps or keyword lists. Ranges are inclusive
`{minimum, maximum}` tuples or integer ranges, such as `-10..10`. Use tuples for
floating-point endpoints; Elixir's range syntax accepts integer endpoints.
An input range of `nil`, or an omitted parameter range, means any finite value
representable in the circuit's precision.

The compiler accepts an external capture such as `&Envelope.amplitude/3`.
Parameter updates keep the same graph:

```elixir
changed = Handwired.put_parameters(circuit, decay: 1.5)
```

## Supported frontend language

- One public `defwire` clause with simple named arguments.
- Numeric literals, local bindings and rebinding, and expression blocks.
- `+`, `-`, `*`, `/`, unary signs, `min`, `max`, `abs`, and imported `relu`.
- Numeric comparisons: `<`, `<=`, `>`, `>=`, `==`, `!=`, `===`, `!==`.
- Boolean literals, `and`, `or`, `not`, and `if ... do ... else ... end`.
- Fixed tuples, lists, and atom-keyed maps as intermediate/output structures.
- Conditional structured outputs when both branches have matching structure
  and leaf types. Numeric inputs are scalar in this milestone.

Unknown syntax is rejected during compilation. Helper calls, remote calls,
general macro expansion, `case`, multi-clause dispatch, recursion, dynamic
collections, pattern destructuring, and IO are future compiler work. As with
normal Elixir, numeric zero is truthy. Boolean `and`/`or` require booleans.

## What the compiler produces

`%Handwired.Circuit{}` holds:

- `nodes`: a topologically ordered tuple of operations referring to earlier
  nodes by integer ID;
- `sources`: named inputs and parameters, their ranges, and source node IDs;
- `parameters`: the current trainable values, separate from the graph;
- `output`: a scalar or nested output description;
- `precision`: `:f64` or `:f32`;
- `backend`: `:semantic` after `compile/2`, `:primitives` after `lower/2`;
- `macros`: backward-only rules preserving authored derivatives at corners;
- `layers`: groups of primitive instruction IDs at each dependency depth;
- `lowering`: comparison strategy counts, division policy, and the semantic-to-primitive wire map.

For example, inspect `Tuple.to_list(circuit.nodes)` in IEx. The semantic
evaluator memoizes nodes on each invocation. It evaluates only the selected arm
of a conditional and only the needed operand of a short-circuit expression.
An inactive `1 / 0` branch therefore causes no error. Its differentiation tape
contains only the operations evaluated for that call.
Execution follows dependencies from the outputs: unused bindings are not
evaluated, including arithmetic errors in those unused expressions.

`compile/2` preserves that semantic graph as a reference. `lower/2` builds a new
graph with exactly three node shapes:

```elixir
{:source, name}
{:affine, [{wire, coefficient}, ...], bias}
{:reglu, left_terms, left_bias, gate_terms, gate_bias}
```

An affine form is an ordered sparse weighted sum. ReGLU computes
`left_form * max(0, gate_form)`. A ReLU is a ReGLU with constant left form 1.
Constants are affine forms with no terms. All forward instructions in a lowered
network use these primitives, including the implementation of comparisons and
conditionals. There is no forward call back into the semantic evaluator.

## Lowering to arithmetic gates

```elixir
semantic = Handwired.compile(&MyProgram.decision/1,
  inputs: [x: -20..20], precision: :f32)
network = Handwired.lower(semantic, grids: [x: 1])
Handwired.stats(network)
```

`grids` is optional. Each supplied step must be a representable positive power of
two. A step of 1 declares integers; `1 / 64` declares multiples of 1/64. Sources
are checked after conversion to the circuit's precision, on every run, parameter
update, and state transition. Leave continuous trainable parameters off the grid
list: the optimizer does not quantize updates onto a discrete grid.

The lowerer propagates conservative ranges and provable grids through arithmetic.
For comparisons, both operands need finite ranges whose difference fits the
precision. These include parameter ranges when a parameter controls a decision.

| Authored operation | Primitive construction |
| --- | --- |
| Addition, subtraction, negation | Ordered affine form |
| Signed `a * b` | `a * relu(b) - a * relu(-b)`; one gate when an operand is provably nonnegative |
| `abs(x)` | `relu(x) + relu(-x)` |
| Boolean AND / OR / NOT | `relu(a+b-1)` / `1-relu(1-a-b)` / `1-a` on exact bits |
| `if bit, do: yes, else: no` | `yes * relu(bit) + no * relu(1-bit)` |
| `min` / `max` | Exact comparison bit and gated operand selection; comparison with zero uses a direct ReLU construction |
| Comparison on a proven grid | Two or three ReLU gates when all partial sums fit the precision's exact-integer budget |
| General floating-point comparison | Rounded difference followed by a sign-amplifying, saturating ReLU cascade |

For an integer `x`, the bit for `x > 4` can be written as
`relu(1 - relu(5 - x))`. That bit gates the two result values. The demo produces
**4 affine instructions and 4 ReGLU instructions**, plus one source, for the
whole `if x > 4, do: 10, else: 12` program. For fractional inputs the cascade
resolves the sign of the difference instead of interpolating across the threshold.

The cascade is adapted from upstream's comparison construction, extended to both
precisions: it resolves nonzero differences down to `2^-149` for FP32 and
`2^-1074` for FP64. Each stage scales by a power of two and clamps before the
next stage. It preserves tiny differences while bringing large magnitudes into
range. The CPU evaluator preserves subnormals; the tests cover every FP32
exponent class and FP64 subnormal, adjacent-value, and extreme-magnitude cases.

All primitive arithmetic executes eagerly in topological order, so both arms of
a conditional must produce finite arithmetic. Numeric zero remains truthy in
Elixir: a numeric `if` condition lowers to an always-on gate. Boolean leaves use
0/1 internally and are decoded to Elixir booleans at the output boundary.
Numeric and Boolean leaves are supported; replace `nil` branches with explicit
numeric or Boolean results before lowering.

Division by a compile-time power-of-two constant lowers exactly to an affine
coefficient when its reciprocal is representable. Other constant divisors can be
enabled with `Handwired.lower(circuit, division: :reciprocal)`, which explicitly
chooses rounded reciprocal multiplication and may differ from direct division.
Runtime-variable division is rejected during lowering, including in branches;
it is still available in the semantic evaluator. There is no hidden fallback.

`Handwired.stats/1` reports source, affine, ReGLU, layer, and stored-weight counts,
including bias slots. Parameters remain source cells, so fitting and
`put_parameters/2` change their values without rebuilding the network. Layers
describe dependency depth; the current CPU executor walks instructions in stored
order. Affine fusion, packed storage, and parallel layer execution can build on
this representation.

## Gradients and numeric policy

The default is **`:f64`**, using BEAM floating-point arithmetic. All numeric
sources and literals are converted to floats. This is a declared numeric domain,
not arbitrary-precision Elixir integer arithmetic: `===` compares the converted
numbers, so numeric integer/float distinctions do not survive compilation.

Use `precision: :f32` to round each numeric source, literal, operation result,
gradient contribution, and gradient accumulation through IEEE-754 single
precision. Range endpoints are rounded to the same domain before validation,
so a rounded value on a boundary can be fed back as state. Boolean values are
kept as booleans by the semantic evaluator and represented as bits inside the
primitive network. Each primitive affine form rounds its bias, coefficient
products, and running sums in stored order; ReGLU rounds its final product.
The lowerer preserves rounding points instead of fusing neighboring sums.
Non-finite inputs/results and out-of-range sources raise an
`ArgumentError`; execution errors identify graph node IDs and operations.

`value_and_grad/3` returns `{value, gradients}` for a **scalar numeric output**.
`:wrt` accepts `:parameters` (default), `:inputs`, `:all`, or a list of source
names. To train a structured-output program, first author a scalar objective
for the aspect you want to optimize.

Backward execution differentiates real arithmetic using the rounded forward
values. It does not differentiate the discontinuities of floating-point rounding.
ReLU and `abs` have derivative zero at zero. At a `min`/`max` tie the left operand
gets the derivative. A conditional differentiates its selected branch;
comparisons and Boolean logic have zero derivatives. There are no implicit
straight-through estimators. Learning a hard threshold needs a separately
chosen relaxation or search method.

The primitive backward pass differentiates affine and ReGLU forms directly.
Backward-only macro rules preserve multiplication derivatives when an operand is
zero, min/max's left-operand tie convention, and the derivative of a selected
branch. Comparison outputs stop gradients. These rules do not participate in
forward evaluation. Without the multiplication rule, differentiating a pair of
ReLUs with zero slope at their corners would incorrectly stop learning at zero.

## Fitting

`Handwired.fit/3` uses full-batch mean squared error and returns a
`%Handwired.FitResult{circuit, initial_loss, loss, steps, history}`. History is a
list of `{step, loss}` entries including the initial and final points. The
original circuit remains unchanged. Training is deterministic.

| Option | Default | Meaning |
| --- | --- | --- |
| `steps` | `1000` | Maximum updates; zero evaluates the initial objective |
| `learning_rate` | `0.03` | Positive update rate |
| `optimizer` | `:adam` | `:adam` or `:sgd` |
| `tolerance` | `1.0e-10` | Stop when MSE is at or below this value |
| `trainable` | All parameters | Names of the parameters permitted to change |

Adam uses beta1 `0.9`, beta2 `0.999`, epsilon `1.0e-8`, and bias correction.
Updates are projected into declared parameter ranges, then rounded to the
circuit's precision. To keep an approved peak while fitting the decay:

```elixir
circuit = Handwired.put_parameters(circuit, peak: 1.0)
fit = Handwired.fit(circuit, examples, trainable: [:decay])
```

Loss is not guaranteed to decrease on every Adam/SGD step. These are local
optimizers, with inspectable objective history, rather than global solvers.

## Stateful programs

A controller returns a map with fields matching its state input names:

```elixir
defmodule Controller do
  use Handwired

  defwire step(position, velocity, target, dt, stiffness, damping) do
    acceleration = stiffness * (target - position) - damping * velocity
    next_velocity = velocity + acceleration * dt
    %{position: position + next_velocity * dt, velocity: next_velocity}
  end
end

circuit = Handwired.compile(&Controller.step/6,
  inputs: [position: {-100, 100}, velocity: {-100, 100},
           target: {-10, 10}, dt: {0, 0.05}],
  parameters: [stiffness: 40, damping: 12])
  |> Handwired.lower()

machine = Handwired.Machine.new(circuit, position: 0, velocity: 0)
{output, machine} = Handwired.Machine.step(machine, target: 1, dt: 0.01)
# output => %{position: 0.004, velocity: 0.4}
```

`Machine.run/2` returns `{outputs, final_machine}` for an enumerable of events.
State is explicit and immutable, and next-state values are checked before being
retained. Events cannot overwrite state fields. The machine can live inside a
GenServer without making every arithmetic operation a process.

This milestone differentiates individual circuit evaluations. Training through
a whole retained-state history (backpropagation through time) is future work.
The controller example verifies stateful execution; the envelope example verifies
learning.

## Validation

```sh
mix format --check-formatted
mix compile --warnings-as-errors
mix test --warnings-as-errors
mix run examples/primitives.exs
```

The tests cover independent native-Elixir comparisons, central-difference
gradient checks, inactive branches, shared-wire gradients, zero-valued products,
decision boundaries, explicit rounding, held-out predictions, frozen parameters,
range projection, retained state, and 1,000-step controller traces.
Primitive tests also verify the instruction whitelist and layer dependencies,
compare both backends, check grid declarations, exercise exact comparison
cascades, and train both an envelope and a product initialized at zero.

The core is deliberately a readable reference implementation. It uses ordinary
Elixir tuples and maps, with no claim of optimized numerical throughput. Natural
next layers are graph simplification, richer Elixir syntax, unrolled recurrent
training, and an optional Nx backend, with these semantics and tests as the
reference.

## Source map

| File | Responsibility |
| --- | --- |
| `lib/handwired.ex` | `defwire` macro and public API |
| `lib/handwired/compiler.ex` | Quoted Elixir to operation graph |
| `lib/handwired/lowering.ex` | Semantic operations to affine/ReGLU networks |
| `lib/handwired/domain.ex` | Conservative ranges, binary grids, precision budgets |
| `lib/handwired/circuit.ex` | Immutable graph and parameter updates |
| `lib/handwired/evaluator.ex` | Forward execution and reverse differentiation |
| `lib/handwired/number.ex` | Floating-point conversion and range checks |
| `lib/handwired/trainer.ex` | MSE, Adam/SGD, fit results |
| `lib/handwired/machine.ex` | Explicit retained state and event execution |

MIT licensed. See `NOTICE` and `LICENSE-UPSTREAM` for upstream attribution.
