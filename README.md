# Diffusion-Limited Aggregation with Thermal Bias

This project simulates a two-dimensional diffusion-limited aggregation (DLA) process on a square lattice. Particles undergo random walks until they stick to an aggregate, and localized heat sources bias the walkers toward specific regions, generating dendritic growth patterns. The simulation also estimates the fractal dimension of the resulting cluster via box-counting and radial-mass scaling analyses.

## Project Structure

- `app.py` – main Python script that runs the simulation, computes fractal metrics, and generates visualizations.
- `report.tex` / `report.txt` – LaTeX report describing the methodology, results, and full source listing.
- `dla_result.png` – sample output figure illustrating the cluster and fractal-dimension fit (generated when the simulation completes).
- `.gitignore` – excludes build artifacts, virtual environments, and generated files from version control.

## Requirements

- Python 3.9+
- Packages: `numpy`, `matplotlib`, `numba`

Install dependencies with:

```bash
python -m pip install numpy matplotlib numba
```

## Usage

Run the simulation from the project root:

```bash
python app.py
```

The script prints progress updates, reports fractal dimensions at the end, and saves the visualization as `dla_result.png` when running in a headless environment.

## Report Compilation

The LaTeX report (`report.tex`) contains a detailed description of the project and includes the full source code via `\lstinputlisting`. To compile it locally or on Overleaf, ensure the following files sit in the same directory/project:

- `report.tex` (rename `report.txt` if needed)
- `dla_result.png`
- `app.py`

Compile with any LaTeX engine (e.g., `pdflatex`). The resulting PDF summarizes the simulation’s background, methodology, results, and future work ideas.

## Customization

Adjust the parameters near the top of `app.py` to explore different behaviors:

- `N_PARTICLES` – total number of walkers
- `N_HEAT_SOURCES`, `HEAT_STRENGTH`, `HEAT_RADIUS`, `HEAT_BIAS_PROB` – thermal bias configuration
- `STEP_LIMIT` – maximum steps per walker

Increasing the particle count produces more detailed clusters but lengthens runtime. Consider using a GPU backend via Numba CUDA or CuPy for large-scale experiments.

## License

This project is provided as-is for educational and research purposes. Feel free to modify and extend it for your own explorations. Contributions and suggestions are welcome!

