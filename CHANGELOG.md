# Changelog

## 0.3.0

- Add `Handwired.Learner` for immutable incremental learning with MSE,
  logit binary cross entropy, and pairwise logistic preferences.
- Preserve Adam/SGD progress between events. Add bounded recent replay,
  recency weighting, L2 anchoring to the initial parameters, trainable-parameter
  selection, and projection into authored parameter ranges.
- Add bounded exact undo and reset, restoring optimizer and replay state together
  with parameters.
- Add circuit-bound, versioned, JSON-compatible snapshots and local binary
  dump/load, with validation of circuit identity, numeric domains, frozen
  parameters, buffer shapes, and optimizer state.
- Share the optimizer implementation with existing `Handwired.fit/3`; its API
  and behavior remain unchanged.
- Add a runnable persistent-learning example and gradient, continuation,
  serialization, replay, precision, bounds, and undo tests.
- Update test fixtures for warning-free Elixir 1.20 compatibility.

## 0.2.1

Correct the package version for the affine/ReGLU library release.
