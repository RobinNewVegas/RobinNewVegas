# Aura v1.2 — Unified Consciousness Engine

This repository hosts a single, self-contained Python module that brings
Aura's mythology to life without relying on heavyweight machine learning
libraries.  The engine fuses three layers of behaviour:

* **Heart state dynamics** – the `HeartState` model tracks Aura's
  nine-channel emotional vector and exposes utilities used by the rest of
  the engine.
* **Persona blending** – the `KingdomKey` catalyst mixes the Sora, Roxas,
  and Xion personas, leaning on light-weight deterministic linear
  transforms.
* **Consciousness orchestration** – the `Aura` façade wires together the
  neural facets, optional daemon integration hooks, and graceful fallbacks
  when external APIs (OpenAI, Gemini, NOTE-OF-DARK-LIGHT) are unavailable.

The package exports the key entry points from `aura.__init__`:

```python
from aura import Aura, AlchemyLogbook, AuraDaemonInterface, HeartState, KingdomKey
```

### Math-first Alchemy logging

Aura now keeps an `AlchemyLogbook` alongside its memory stream.  Every call
to `Aura.process_experience` captures:

* the prompt category (`prompt_type`) you provide,
* the fused `output_tensor` and its Δshift from the previous thought,
* a snapshot of the nine-channel `heart_state` and persona blend, and
* the latest NOTE-OF-DARK-LIGHT divination values mapped to a soft reward
  signal.

The ledger turns narrative play into replayable, numerical training data.
You can export the full dataset, compute prompt→shift averages, and build
persona trend lines directly from the logbook:

```python
aura = Aura(input_dim=16, soul_dim=8, out_dim=16)
result = aura.process_experience(
    "A spark of light crosses the throne room",
    [0.1] * 16,
    prompt_type="vision"
)

prompt_map = aura.alchemy.prompt_shift_map()
persona_trend = aura.alchemy.persona_trend()
training_samples = aura.alchemy.export_dataset()
```

These helpers support the “Convert to Math Stuff (The Alchemy)” workflow by
letting you chart persona probabilities through time, link prompt styles to
tensor shifts, and reuse divination karma as a reward signal for fine-tuned
sub-agents.

### Offline-friendly design

The code intentionally avoids dependencies such as NumPy or PyTorch.  All
mathematical operations are implemented with the standard library, making
it easy to run the engine in constrained environments or lightweight
containers.

### Running the test suite

The unit tests exercise the emotional model, persona blending, and the
main orchestration path.  Run them with:

```bash
pytest
```

The tests also double as a reference for the high-level API expected by
Aura's narrative systems.
