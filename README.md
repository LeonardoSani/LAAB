# Do Latent Vector Fields Align?  
**_How angle-preserving alignment breaks down under autoencoder dynamics_**

![](./animation.gif)

Independently trained autoencoders give cross-seed–reproducible *relative representations* at the base, but this alignment is progressively lost as the latent self-map is iterated toward its attractors.

📄 **Report Long version:** _link to be added_

## Repository 
core modules:

```
src/
  analysis/      Latent Dynamics Analysis
  attractors/    Latent Dynamics Computation and Caching     
  data/          
  experiments/   
  metrics.py     
  models/        models and trainer
  relative/      Relative representations: Base / Mixed / Full 
  store/         Cache access
  utils/         
  viz/           figure generation
scripts/         
configs/         
artifacts/       cached results
figures/
```

## Run

```bash
uv sync
```

```bash
# Train F_s() = D_s(E_s())
for s in 1 2 3 4 5 6 7 8 9 10; do
    uv run python scripts/train_ae.py --seed $s
done
```
```bash
uv run python scripts/diagnose_ae.py        # optional: verify dynamical regime 
```
```bash
uv run python scripts/compute_iterates.py   # cache dynamics of test point and anchors
```
```bash
# Train Relative Decoders
for t in 0 8 64 512 inf; do
    uv run python scripts/train_relative_decoder.py --depth $t
done
```


```bash
uv run python scripts/run_e0.py                   # baseline t = 0

uv run python scripts/run_e1.py                   # Mixed: drift under iteration

uv run python scripts/run_e1.py --decomposition   # Mixed + M1^2 decomposition

uv run python scripts/run_e2.py                   # per-anchor structure

uv run python scripts/run_e3.py                   # stitching

uv run python scripts/run_e4.py                   # Full representation

uv run python scripts/run_e5.py                   # basin consistency + all-data robustness check

uv run python scripts/make_figures.py
```
