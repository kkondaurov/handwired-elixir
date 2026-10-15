defmodule Handwired.MixProject do
  use Mix.Project

  def project do
    [
      app: :handwired,
      version: "0.2.1",
      elixir: "~> 1.14",
      deps: [],
      start_permanent: Mix.env() == :prod,
      description: "Authored, differentiable numeric programs in pure Elixir",
      source_url: "https://github.com/kkondaurov/handwired-elixir",
      homepage_url: "https://github.com/kkondaurov/handwired-elixir",
      package: [
        licenses: ["MIT"],
        links: %{"GitHub" => "https://github.com/kkondaurov/handwired-elixir"}
      ]
    ]
  end

  def application, do: [extra_applications: [:logger, :crypto]]
end
