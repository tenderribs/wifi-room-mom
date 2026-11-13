# WiFi Room MoM

We simulate WiFi propagation in a 2D room using the **method of moments (MoM)** for 2D TMz scattering, where only $E_z$ is nonzero. The project starts from the course's MATLAB exercise (a plane wave hitting PEC objects), ports it to Python, and extends it towards concrete walls and a router modelled as a line source.

## Setup

Requires Python 3.14.

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Open `main.ipynb` and run all cells. Choose the source in the parameters cell:

- `SRC = "plane_wave"`: incident plane wave with direction `KPW`.
- `SRC = "router"`: line source at `rs` with current `I`.

The notebook plots $|E_z|$, $\mathrm{Re}\,E_z$ and the received power in dB.