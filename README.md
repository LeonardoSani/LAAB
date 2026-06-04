# Do Latent Vector Fields Align?  
**_How angle-preserving alignment breaks down under autoencoder dynamics_**

![](./animation.gif)

Independently trained autoencoders give cross-seed–reproducible *relative representations* at the base, but this alignment is progressively lost as the latent self-map is iterated toward its attractors.

📄 **Paper:** _link to be added_

## Repository

```
src/
  models/        convolutional autoencoder
  attractors/    latent self-map iteration to fixed points
  relative/      Base / Mixed / Full relative representations
  metrics.py     M1–M4, per-anchor badness
  analysis/      basin consistency, split-half reliability
  experiments/   e0–e5, one per result
  store/         checkpoint & iterate caching
  viz/           figures
scripts/         thin runners (run_e0 … run_e4, make_figures)
configs/         experiment configs
artifacts/       checkpoints, cached iterates, results
paper/           manuscript
```

## Run

```bash
uv sync
uv run python scripts/run_e0.py   # baseline (t=0 reproducibility)
uv run python scripts/run_e1.py   # Mixed: drift under iteration
uv run python scripts/run_e2.py   # per-anchor structure
uv run python scripts/run_e3.py   # stitching
uv run python scripts/run_e4.py   # Full representation
uv run python scripts/run_e5.py   # basin consistency
uv run python scripts/make_figures.py
```
