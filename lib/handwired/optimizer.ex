defmodule Handwired.Optimizer do
  @moduledoc false

  def update(:sgd, _, gradient, _, rate, moments), do: {rate * gradient, moments}

  def update(:adam, name, gradient, step, rate, moments) do
    {m, v} = Map.get(moments, name, {0.0, 0.0})
    m = 0.9 * m + 0.1 * gradient
    v = 0.999 * v + 0.001 * gradient * gradient
    corrected_m = m / (1.0 - :math.pow(0.9, step))
    corrected_v = v / (1.0 - :math.pow(0.999, step))
    delta = rate * corrected_m / (:math.sqrt(corrected_v) + 1.0e-8)
    {delta, Map.put(moments, name, {m, v})}
  end
end
