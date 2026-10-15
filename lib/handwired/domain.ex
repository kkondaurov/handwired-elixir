defmodule Handwired.Domain do
  @moduledoc false
  import Bitwise
  alias Handwired.Number

  def bits(:f32), do: 24
  def bits(:f64), do: 53
  def lift(:f32), do: 149
  def lift(:f64), do: 1074
  def max_exponent(:f32), do: 127
  def max_exponent(:f64), do: 1023
  def power(exponent), do: :math.pow(2.0, exponent)

  # Largest power of two dividing an exact binary floating-point value.
  def grid(value) when value == 0, do: :zero

  def grid(value) do
    <<_::1, exponent::11, fraction::52>> = <<value * 1.0::float-64>>

    {mantissa, shift} =
      if exponent == 0, do: {fraction, -1074}, else: {fraction + (1 <<< 52), exponent - 1075}

    power(shift + trailing(mantissa))
  end

  defp trailing(n) when rem(n, 2) == 0, do: 1 + trailing(div(n, 2))
  defp trailing(_), do: 0

  def declared_grid(value, precision) do
    number = Number.cast(value, :f64)

    unless number > 0 and grid(number) == number and Number.cast(number, precision) == number,
      do: raise(ArgumentError, "grid must be a positive representable power of two")

    number
  end

  def validate_grid(_, nil, _), do: :ok

  def validate_grid(value, step, name) do
    actual = grid(value)

    unless actual == :zero or actual >= step,
      do: raise(ArgumentError, "#{name} must lie on the declared #{step} grid, got #{value}")

    :ok
  end

  def combine(nil, _), do: nil
  def combine(_, nil), do: nil
  def combine(:zero, b), do: b
  def combine(a, :zero), do: a
  def combine(a, b), do: min(a, b)

  def multiply_grid(:zero, _), do: :zero
  def multiply_grid(_, :zero), do: :zero
  def multiply_grid(nil, _), do: nil
  def multiply_grid(_, nil), do: nil

  def multiply_grid(a, b) do
    result = safe(fn -> a * b end)
    if result == 0.0, do: nil, else: result
  end

  def max_abs(nil), do: nil
  def max_abs({low, high}), do: max(abs(low), abs(high))
  def nonnegative?({low, _}), do: low >= 0
  def nonnegative?(_), do: false

  def add(nil, _, _), do: nil
  def add(_, nil, _), do: nil

  def add({a, b}, {c, d}, precision),
    do: safe(fn -> {Number.cast(a + c, precision), Number.cast(b + d, precision)} end)

  def multiply(nil, _, _), do: nil
  def multiply(_, nil, _), do: nil

  def multiply({a, b}, {c, d}, precision) do
    safe(fn ->
      values = for x <- [a, b], y <- [c, d], do: Number.cast(x * y, precision)
      Enum.min_max(values)
    end)
  end

  def relu(nil), do: nil
  def relu({a, b}), do: {max(0.0, a), max(0.0, b)}
  def union(nil, _), do: nil
  def union(_, nil), do: nil
  def union({a, b}, {c, d}), do: {min(a, c), max(b, d)}

  def within_grid?(_, nil, _), do: false
  def within_grid?(nil, _, _), do: false
  def within_grid?({a, b}, :zero, _), do: a == 0 and b == 0

  def within_grid?(range, step, precision) do
    safe(fn -> max_abs(range) / step <= power(bits(precision)) end) == true
  end

  def safe(fun) do
    fun.()
  rescue
    _ in [ArithmeticError, ArgumentError] -> nil
  end

  def ceiling_exponent(value) do
    exponent = trunc(:math.ceil(:math.log2(value)))
    adjust_exponent(value, exponent)
  end

  defp adjust_exponent(value, exponent) do
    cond do
      exponent < 1024 and power(exponent) < value -> adjust_exponent(value, exponent + 1)
      exponent > -1074 and power(exponent - 1) >= value -> adjust_exponent(value, exponent - 1)
      true -> exponent
    end
  end
end
