# PINN
A Physics-Informed Neural Network (PINN) that models 1D carbon diffusion during steel carburization and solves an inverse problem to identify hidden internal carbon sources from sparse concentration data.


# Physics-Informed Neural Network for 1D Carbon Transport in Carburization

A PINN-based framework that models transient carbon diffusion during carburization and solves an **inverse problem** to identify hidden internal carbon sources (e.g., carbide dissolution) from sparse concentration measurements.

> **B.Tech Mini Project** — Metallurgical and Materials Engineering, VNIT Nagpur (2025–26)  
> Authors: Sarthak Bondre · Yatharth Nema  
> Guide: Prof. Dr. Abhinav Arya

---

## What the project does

Carburization diffuses carbon from a carbon-rich furnace atmosphere into the surface of steel components. The carbon concentration profile controls the final hardness gradient. Conventional models (FDM/FEM) solve this as a *forward* problem — given the source, predict the concentration.

This project goes further: it treats the source itself as **unknown** and recovers it from a handful of sparse concentration measurements, using a neural network that is simultaneously constrained to obey the diffusion PDE.

The governing equation solved is:

$$\frac{\partial C}{\partial t} = D(t)\frac{\partial^2 C}{\partial x^2} + S(x,t;\,x_0,\,A)$$

where the source term $S$ (representing carbide dissolution) is parameterized as a Gaussian and its location $x_0$ and amplitude $A$ are **trainable parameters** learned during training.

---

## Repository structure

```
.
├── PINN_FINAL.ipynb       # Full implementation (FD solver + PINN + inverse identification)
├── Mini_project_S26.pdf   # Project report
└── README.md
```

---

## Key results

| Quantity | True (FD) | Recovered (PINN) |
|---|---|---|
| Source location $x_0$ | 0.40 | ≈ 0.40 |
| Concentration profiles | Reference | Strong agreement across all time steps |
| Error | — | Low across domain; slightly higher near boundaries |

The PINN correctly reconstructed the Gaussian carbide dissolution zone and matched the finite-difference reference solution over the full spatio-temporal domain.

---

## Method summary

### 1. Reference solution (Finite Difference)
An explicit FD solver generates the ground-truth concentration field $C_\text{FD}(x,t)$ on a 201×801 space–time grid. The source is fixed at $x_0 = 0.4$ with a Gaussian spatial profile that decays exponentially in time.

### 2. Sparse observations
500 points are randomly sampled from the FD solution, mimicking sparse experimental measurements (EPMA/SIMS).

### 3. PINN architecture
- **Inputs:** $(x, t)$
- **Hidden layers:** 4 × 64 neurons, `tanh` activation
- **Output:** $C_\text{PINN}(x, t)$
- **Extra trainable variables:** `x0_unb`, `A_unb` (mapped through sigmoid/softplus to enforce physical constraints)

### 4. Composite loss
$$\mathcal{L} = w_\text{PDE}\,\mathcal{L}_\text{PDE} + w_\text{IC}\,\mathcal{L}_\text{IC} + w_L\,\mathcal{L}_{BC,L} + w_R\,\mathcal{L}_{BC,R} + w_\text{data}\,\mathcal{L}_\text{data}$$

Weights: PDE=1, IC=10, BC=10, data=50.

### 5. Optimization
Adam optimizer, learning rate $10^{-3}$, 10,000 epochs. Gradients computed via TensorFlow automatic differentiation.

---

## Setup and usage

```bash
# Install dependencies
pip install tensorflow numpy matplotlib

# Run the notebook
jupyter notebook PINN_FINAL.ipynb
```

Tested with Python 3.9+, TensorFlow 2.x.

To change training duration or collocation point count, edit the top of the notebook:

```python
EPOCHS = 10000   # number of training epochs
Nf     = 8000    # number of PDE collocation points
```

---

## Physics parameters

| Parameter | Value | Description |
|---|---|---|
| $D_0$ | $2\times10^{-5}$ m²/s | Pre-exponential diffusivity |
| $E_a$ | 142,000 J/mol | Activation energy for C in austenite |
| $R$ | 8.314 J/mol·K | Universal gas constant |
| $T(t)$ | $1173 + 50\sin(0.5t)$ K | Oscillating furnace temperature |
| $x_0$ (true) | 0.4 | Source depth (normalized) |
| $\sigma_x$ | 0.1 | Spatial spread of source |

---

## Future directions

- Extension to 2D/3D diffusion geometry
- Incorporation of thermodynamic activity models for carbon
- Validation against real EPMA or SIMS depth profiles
- More complex multi-carbide dissolution kinetics

---

## References

1. Raissi, Perdikaris, Karniadakis — *Physics-informed neural networks*, J. Comput. Phys., 2019
2. Karniadakis et al. — *Physics-informed machine learning*, Nature Reviews Physics, 2021
3. Meng et al. — *PPINN: Parareal physics-informed neural network*, CMAME, 2020
