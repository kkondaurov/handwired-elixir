defmodule Handwired.TestPrograms do
  use Handwired

  defwire(envelope(t, peak, decay), do: max(0.0, peak - decay * t))
  defwire(square(x), do: x * x)
  defwire(product(x, y), do: x * y)
  defwire(constant(x), do: x * 0 + 7)

  defwire formula(x, y, scale) do
    sum = x + y
    difference = x - y
    sum * difference / scale + abs(x) - min(x, y)
  end

  defwire branch(x, scale) do
    if x > 0 do
      scale * x
    else
      -scale * x + 1
    end
  end

  defwire(safe_inverse(x), do: if(x == 0, do: 0, else: 1 / x))
  defwire(safe_boolean(x), do: if(x != 0 and 1 / x > 0, do: 1, else: 0))
  defwire(safe_or(x), do: if(x == 0 or 1 / x > 0, do: 1, else: 0))
  defwire(truthiness(x), do: if(x, do: 10, else: 20))
  defwire(comparison(x, threshold), do: if(x >= threshold, do: 10, else: 12))
  defwire(rounded(x), do: x + 1 - x)
  defwire(overflow(x), do: x * x)
  defwire(rectified(x), do: relu(x))
  defwire(minimum(x, y), do: min(x, y))
  defwire(maximum(x, y), do: max(x, y))
  defwire(offset(x, bias), do: x + bias)

  defwire structured(x) do
    if not (x < 0) do
      %{point: {x, x * 2}, samples: [x, x + 1], positive: true}
    else
      %{point: {-x, -x * 2}, samples: [-x, -x + 1], positive: false}
    end
  end

  defwire rebound(x) do
    x = x + 1
    x = x * 2
    x - 3
  end

  defwire scoped(x) do
    y = 10

    selected =
      if x > 0 do
        y = x * 2
        y + 1
      else
        y = x * 3
        y - 1
      end

    selected + y
  end

  defwire controller(position, velocity, target, dt, stiffness, damping) do
    acceleration = stiffness * (target - position) - damping * velocity
    next_velocity = velocity + acceleration * dt
    next_position = position + next_velocity * dt
    %{position: next_position, velocity: next_velocity}
  end

  defwire(accumulator(total, amount), do: %{total: total + amount})
  defwire(unsupported(x), do: :math.sin(x))
  defwire(mismatch(x), do: if(x > 0, do: 1, else: {1, 2}))
end
