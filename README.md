# GratingCouplerOpt

Inverse design and optimization of integrated grating couplers, plus statistical
(yield) analysis of how the optimized designs hold up under fabrication variation.

Simulations are run with [Tidy3D](https://www.flexcompute.com/tidy3d/) FDTD and
optimized with adjoint gradients (`autograd` + `optax`/Adam). Two device families
are studied:

- **`GC_4um_2D/`** — 2D InP grating coupler for a 4 µm mode-field-diameter fiber.
  The main workhorse: initial design, gradient-based optimization, stochastic
  (fabrication-robust) optimization, and sensitivity analysis.
- **`GC_3D_4um_Si/`** — 3D silicon grating coupler for a 4 µm fiber, seeded from
  the 2D results and refined in full 3D.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Running Tidy3D simulations requires a Flexcompute account and an API key
configured via `tidy3d configure`.

## Main files

### `GC_4um_2D/` — 2D InP grating coupler
| File | Purpose |
| --- | --- |
| `main.py` | Core library. Builds the parametrized grating geometry (`make_grating_structure`), assembles the Tidy3D simulation (`make_sim`), apodization helpers (`apodized_to_widths`, `get_centers`), the tanh parameter projection/bounds (`projection_builder`), the Adam optimization loop (`run_adam`), and the coupling-efficiency figure of merit (`get_coupling_efficiency`). |
| `playground.ipynb` | Scratch notebook for setting up and visually checking the simulation. |
| `bayesianOpt.ipynb` | Bayesian optimization to find good initial design parameters (apodization rate `R`, `r0`, fill factor, etch depth). |
| `initial_opt.ipynb` | Adjoint/Adam gradient optimization starting from the found initial parameters. |
| `stochastic_opt.ipynb` | Stochastic gradient descent that samples fabrication errors (etch depth, alignment, over/under-etch ~ N(0, 5 nm)) each step for a fabrication-robust design. |
| `adjoint_sensitivity.ipynb` | Adjoint-based sensitivity analysis of the final design to the same fabrication perturbations. |
| `random.ipynb` | Quick scratch notebook for plotting/comparing saved optimization runs. |
| `data/*.json` | Saved optimization histories (initial vs. stochastic, 50 nm / 100 nm grids, 6-tooth variant). |

### `GC_3D_4um_Si/` — 3D Si grating coupler
| File | Purpose |
| --- | --- |
| `main.py` | 3D counterpart of the 2D core library: geometry, simulation setup, optimization, and FOM for the silicon device. |
| `device.ipynb` | Description and visualization of the 4 µm 3D grating coupler device. |
| `opt.ipynb` | 3D gradient-based optimization of the grating coupler (seeded from the 2D result). |
| `playground.ipynb` | Scratch notebook for building/checking the 3D simulation. |
| `analysis.ipynb` | Performance analysis of the optimized device (incident angle, misalignment sweeps). |
| `data/*.pkl` | Saved optimization states (2D-seed and 3D runs). |

## Notes

`fdve-*.hdf5` and `batch.hdf5` files are Tidy3D server-side run caches and are
git-ignored (they are regenerated when simulations run).
