defmodule Handwired.Number do
  @moduledoc false

  def cast(value, precision) when is_number(value) and precision in [:f32, :f64] do
    try do
      number = value * 1.0

      case precision do
        :f64 ->
          number

        :f32 ->
          # Round through an IEEE-754 single. Decoding a non-finite result raises.
          <<rounded::float-32>> = <<number::float-32>>
          rounded
      end
    rescue
      _ -> raise ArgumentError, "#{inspect(value)} is not a finite #{precision} number"
    end
  end

  def cast(value, precision),
    do: raise(ArgumentError, "expected a finite #{precision} number, got: #{inspect(value)}")

  def range(nil), do: nil
  def range(%Range{first: first, last: last}), do: range({first, last})

  def range({low, high}) when is_number(low) and is_number(high) do
    low = cast(low, :f64)
    high = cast(high, :f64)
    if low > high, do: raise(ArgumentError, "range minimum must not exceed maximum")
    {low, high}
  end

  def range(other), do: raise(ArgumentError, "invalid numeric range: #{inspect(other)}")

  def quantize_range(nil, _precision), do: nil
  def quantize_range({low, high}, precision), do: {cast(low, precision), cast(high, precision)}

  def validate(value, range, precision, name) do
    # Both bounds and values belong to the declared floating-point domain. This
    # also allows a rounded state/parameter on a bound to be fed back unchanged.
    number = cast(value, precision)

    case range do
      {low, high} when number < low or number > high ->
        raise ArgumentError, "#{name} must be within #{low}..#{high}, got #{number}"

      _ ->
        number
    end
  end

  def clamp(value, nil), do: value
  def clamp(value, {low, high}), do: max(low, min(high, value))
end
