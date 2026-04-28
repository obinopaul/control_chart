# Cost-Sensitive Online Learning for Control Chart Pattern Recognition

This repository contains the codebase for the paper **"Cost-Sensitive Online Learning for Control Chart Pattern Recognition"**. It implements novel Cost-Sensitive Passive-Aggressive (CSPA) algorithms for detecting abnormal patterns in control charts under class imbalance, where normal samples vastly outnumber abnormal ones (e.g., 900:100 ratio). The proposed algorithms incorporate cost-sensitive loss functions and class-specific regularization into the Passive-Aggressive online learning framework to improve detection of rare abnormal patterns. Comparisons are also provided against existing cost-sensitive online learning methods (CSRDA and CSTG).

## Algorithms

The table below maps each algorithm used in the paper to its implementation. The **Proposed** algorithms are the CSPA variants introduced in this work. The **Literature** algorithms are existing methods used for comparison.

### Proposed CSPA Algorithms

<table>
<thead>
<tr>
<th>Paper Name</th>
<th>File</th>
<th>Function</th>
<th>Loss Function</th>
<th>Step Size</th>
<th>Weight Update</th>
</tr>
</thead>
<tbody>

<tr>
<td>

$\text{CSPA}_1$

</td>
<td><code>algorithms/PA1_Csplit.py</code></td>
<td><code>PA1_Csplit</code></td>
<td>

$\ell^{\text{Hinge}} = \max(0, 1 - y_t \mathbf{\omega}_t^\top \mathbf{x}_t)$

</td>
<td>

$\tau_t = \min(C_t,\; \ell_t / \lVert\mathbf{x}_t\rVert^2)$

</td>
<td>

$\mathbf{\omega}_{t+1} = \mathbf{\omega}_t + \tau_t y_t \mathbf{x}_t$

</td>
</tr>

<tr>
<td>

$\text{CSPA}_2$

</td>
<td><code>algorithms/PA2_Csplit.py</code></td>
<td><code>PA2_Csplit</code></td>
<td>

$\ell^{\text{Hinge}} = \max(0, 1 - y_t \mathbf{\omega}_t^\top \mathbf{x}_t)$

</td>
<td>

$\tau_t = \ell_t / (\lVert\mathbf{x}_t\rVert^2 + 1/(2C_t))$

</td>
<td>

$\mathbf{\omega}_{t+1} = \mathbf{\omega}_t + \tau_t y_t \mathbf{x}_t$

</td>
</tr>

<tr>
<td>

$\text{CSPA-}\ell^{I}$

</td>
<td><code>algorithms/PA_L1.py</code></td>
<td><code>PA_L1</code></td>
<td>

$\ell^{I} = \max(0,\; (\rho \cdot \mathbb{I}_{y_t=1} + \mathbb{I}_{y_t=-1}) - y_t \mathbf{\omega}_t^\top \mathbf{x}_t)$

</td>
<td>

$\tau_t = \ell_t / \lVert\mathbf{x}_t\rVert^2$

</td>
<td>

$\mathbf{\omega}_{t+1} = \mathbf{\omega}_t + \tau_t y_t \mathbf{x}_t$

</td>
</tr>

<tr>
<td>

$\text{CSPA-}\ell^{II}$

</td>
<td><code>algorithms/PA_L2.py</code></td>
<td><code>PA_L2</code></td>
<td>

$\ell^{II} = (\rho \cdot \mathbb{I}_{y_t=1} + \mathbb{I}_{y_t=-1}) \cdot \max(0, 1 - y_t \mathbf{\omega}_t^\top \mathbf{x}_t)$

</td>
<td>

$\tau_t = (1 - y_t f_t) / (\bar{\rho}_t \lVert\mathbf{x}_t\rVert^2)$

</td>
<td>

$\mathbf{\omega}_{t+1} = \mathbf{\omega}_t + \tau_t \bar{\rho}_t y_t \mathbf{x}_t$

</td>
</tr>

<tr>
<td>

$\text{CSPA}_1\text{-}\ell^{I}$

</td>
<td><code>algorithms/PA1_L1.py</code></td>
<td><code>PA1_L1</code></td>
<td>

$\ell^{I} = \max(0,\; (\rho \cdot \mathbb{I}_{y_t=1} + \mathbb{I}_{y_t=-1}) - y_t \mathbf{\omega}_t^\top \mathbf{x}_t)$

</td>
<td>

$\tau_t = \min(C_t,\; \ell_t / \lVert\mathbf{x}_t\rVert^2)$

</td>
<td>

$\mathbf{\omega}_{t+1} = \mathbf{\omega}_t + \tau_t y_t \mathbf{x}_t$

</td>
</tr>

<tr>
<td>

$\text{CSPA}_1\text{-}\ell^{II}$

</td>
<td><code>algorithms/PA1_L2.py</code></td>
<td><code>PA1_L2</code></td>
<td>

$\ell^{II} = (\rho \cdot \mathbb{I}_{y_t=1} + \mathbb{I}_{y_t=-1}) \cdot \max(0, 1 - y_t \mathbf{\omega}_t^\top \mathbf{x}_t)$

</td>
<td>

$\tau_t = \min(C_t,\; \ell_t / (\bar{\rho}_t^2 \lVert\mathbf{x}_t\rVert^2))$

</td>
<td>

$\mathbf{\omega}_{t+1} = \mathbf{\omega}_t + \tau_t \bar{\rho}_t y_t \mathbf{x}_t$

</td>
</tr>

<tr>
<td>

$\text{CSPA}_2\text{-}\ell^{I}$

</td>
<td><code>algorithms/PA2_L1.py</code></td>
<td><code>PA2_L1</code></td>
<td>

$\ell^{I} = \max(0,\; (\rho \cdot \mathbb{I}_{y_t=1} + \mathbb{I}_{y_t=-1}) - y_t \mathbf{\omega}_t^\top \mathbf{x}_t)$

</td>
<td>

$\tau_t = \ell_t / (\lVert\mathbf{x}_t\rVert^2 + 1/(2C_t))$

</td>
<td>

$\mathbf{\omega}_{t+1} = \mathbf{\omega}_t + \tau_t y_t \mathbf{x}_t$

</td>
</tr>

<tr>
<td>

$\text{CSPA}_2\text{-}\ell^{II}$

</td>
<td><code>algorithms/PA2_L2.py</code></td>
<td><code>PA2_L2</code></td>
<td>

$\ell^{II} = (\rho \cdot \mathbb{I}_{y_t=1} + \mathbb{I}_{y_t=-1}) \cdot \max(0, 1 - y_t \mathbf{\omega}_t^\top \mathbf{x}_t)$

</td>
<td>

$\tau_t = \ell_t / (\bar{\rho}_t^2 \lVert\mathbf{x}_t\rVert^2 + 1/(2C_t))$

</td>
<td>

$\mathbf{\omega}_{t+1} = \mathbf{\omega}_t + \tau_t \bar{\rho}_t y_t \mathbf{x}_t$

</td>
</tr>

</tbody>
</table>

Where $\rho = (\eta_p T_n) / (\eta_n T_p)$ is the cost-sensitive parameter, $\bar{\rho}_t = \rho \cdot \mathbb{I}_{y_t=1} + \mathbb{I}_{y_t=-1}$, and $C_t = C^+$ if $y_t = +1$ or $C_t = C^-$ if $y_t = -1$ (class-specific regularization).

### Literature Benchmark Algorithms

| Paper Name    | File                    | Function  | Reference                                                                                     |
| ------------- | ----------------------- | --------- | --------------------------------------------------------------------------------------------- |
| CSRDA-I       | `algorithms/CSRDA_1.py` | `CSRDA_1` | Chen et al., "Cost-sensitive Regularized Dual Averaging" (IEEE ICBK, 2021)                    |
| CSRDA-II      | `algorithms/CSRDA_2.py` | `CSRDA_2` | Chen et al., "Cost-sensitive Regularized Dual Averaging" (IEEE ICBK, 2021)                    |
| CSTG-I        | `algorithms/CSTG_1.py`  | `CSTG_1`  | Chen et al., CSTG: An effective framework for cost-sensitive sparse online learning. (Society for Industrial and Applied Mathematics 2017).       |
| CSTG-II       | `algorithms/CSTG_2.py`  | `CSTG_2`  | CSTG: An effective framework for cost-sensitive sparse online learning. (Society for Industrial and Applied Mathematics 2017).        |
| $\text{PA}$            | `algorithms/PA.py`      | `PA`      | Crammer et al., 2006. Online passive-aggressive algorithms. Journal of Machine Learning Research, 7(Mar), pp.551-585. |
| $\text{PA}_1$ | `algorithms/PA1.py`     | `PA1`     | Crammer et al., 2006. Online passive-aggressive algorithms. Journal of Machine Learning Research, 7(Mar), pp.551-585. |
| $\text{PA}_2$ | `algorithms/PA2.py`     | `PA2`     | Crammer et al., 2006. Online passive-aggressive algorithms. Journal of Machine Learning Research, 7(Mar), pp.551-585. |

## Installation

```bash
pip install -r requirements.txt
```

## Data Generation

Each dataset contains samples of the form $(\mathbf{x}_t, y_t)$ with $y_t \in \{-1, +1\}$, where $+1$ indicates the abnormal class and $-1$ indicates the normal class. The synthetic control chart patterns are generated using the mathematical models below, following Bag et al. (2012):

| Pattern        | Mathematical Model                                      |
| -------------- | ------------------------------------------------------- |
| Normal         | $x_t = \mu + r_t \sigma$                                |
| Up-trend       | $x_t = \mu + r_t \sigma + g t$                          |
| Down-trend     | $x_t = \mu + r_t \sigma - g t$                          |
| Up-shift       | $x_t = \mu + r_t \sigma + k s$                          |
| Down-shift     | $x_t = \mu + r_t \sigma - k s$                          |
| Systematic     | $x_t = \mu + r_t \sigma + d(-1)^t$                      |
| Cyclic         | $x_t = \mu + r_t \sigma + \alpha \sin(2\pi t / \Omega)$ |
| Stratification | $x_t = \mu + r_t \sigma'$                               |

The parameter ranges used for data generation are:

| Symbol    | Pattern        | Parameter                  | Range                         |
| --------- | -------------- | -------------------------- | ----------------------------- |
| $m$       | All            | Window length              | $\{10, 15, 20, \ldots, 100\}$ |
| $\mu$     | All            | Process mean               | $0$                           |
| $\sigma$  | All            | Process standard deviation | $1$                           |
| $r_t$     | All            | Random noise               | $\mathcal{N}(0, 1)$           |
| $g$       | Trend          | Magnitude of gradient      | $[0.005\sigma, 0.605\sigma]$  |
| $k$       | Shift          | Shift position             | $m/2$                         |
| $s$       | Shift          | Shift magnitude            | $[0.005\sigma, 1.805\sigma]$  |
| $d$       | Systematic     | Magnitude                  | $[0.005\sigma, 1.805\sigma]$  |
| $\alpha$  | Cyclic         | Amplitude                  | $[0.005\sigma, 1.805\sigma]$  |
| $\Omega$  | Cyclic         | Period                     | $8$                           |
| $\sigma'$ | Stratification | Standard deviation         | $[0.005\sigma, 0.8\sigma]$    |

### Generate a Single Dataset

```bash
python data_generator.py -t bc -d binary_synthetic_data.libsvm -w 48 --t 0.5 -a 900 -b 100 -m 1 --abtype 1 --normalize_abnormal
```

**Parameters:**

- `-t bc`: Binary classification task
- `-d`: Output file path
- `-w 48`: Window length $m$
- `--t 0.5`: Abnormal parameter magnitude
- `-a 900`: Number of normal samples
- `-b 100`: Number of abnormal samples
- `--abtype 1`: Abnormal pattern type (1=up-trend, 2=down-trend, 3=up-shift, 4=down-shift, 5=systematic, 6=cyclic, 7=stratification)
- `--normalize_abnormal`: Normalize abnormal samples to unit norm

### Generate All Datasets for a Single Pattern Type

```bash
python generate_datasets.py 1  # Generate all datasets for up-trend (abtype 1)
```

This creates 19 window lengths $\times$ 40 parameter values = 760 datasets for the specified abnormal type.

### Generate All Datasets for All Pattern Types

```bash
python generate_all_datasets.py
```

This generates datasets for all 7 abnormal pattern types (abtype 1--7), totaling 5,320 datasets.

## Running the Algorithms

### Run a Single Algorithm on a Single Dataset

```bash
python run.py -t bc -a PA1_L1 -d data/abtype1/abtype1_w50_t0.5.libsvm -f libsvm -n 20
```

**Parameters:**

- `-t bc`: Binary classification task
- `-a PA1_L1`: Algorithm name (must match the function name from the algorithms table above)
- `-d`: Path to the dataset
- `-f libsvm`: File format
- `-n 20`: Number of independent runs

### Run All Algorithms on a Single Dataset

```bash
python compare.py -t bc -d data/abtype1/abtype1_w50_t0.5.libsvm -f libsvm -n 20 -s results/abtype1
```

### Run All Algorithms on All Datasets for a Pattern Type

```bash
python compare_all.py 1  # Run all algorithms on all abtype 1 (up-trend) datasets
```

Results are saved to `results/abtype{X}/`.

### Run on a SLURM Cluster

```bash
sbatch compare_all.sh
```

## Visualization

The following scripts generate the plots used in the paper.

| Script                | Description                                                                                |
| --------------------- | ------------------------------------------------------------------------------------------ |
| `plot_heatmaps.py`    | Performance heatmaps (window length vs. abnormal parameter) for all metrics and algorithms |
| `plot_gmean.py`       | G-Mean vs. abnormal parameter for each algorithm across window lengths                     |
| `plot_time.py`        | Execution time vs. abnormal parameter across window lengths                                |
| `plot_metrics.py`     | All performance metrics over training samples for each dataset                             |
| `plot_metrics_PA.py`  | Performance metrics focused on the PA algorithm variants                                   |
| `plot_cer.py`         | Cumulative error rate plots                                                                |
| `plot_collage.py`     | Interactive HTML collage for side-by-side algorithm comparison                             |
| `plot_performance.py` | Performance summary plots                                                                  |
| `plot_all.py`         | Runs multiple plotting scripts in parallel                                                 |

## Project Structure

```
CCPR_main/
├── algorithms/                         # Algorithm implementations
│   ├── PA1_Csplit.py                   # CSPA₁
│   ├── PA2_Csplit.py                   # CSPA₂
│   ├── PA_L1.py                        # CSPA-ℓᴵ
│   ├── PA_L2.py                        # CSPA-ℓᴵᴵ
│   ├── PA1_L1.py                       # CSPA₁-ℓᴵ
│   ├── PA1_L2.py                       # CSPA₁-ℓᴵᴵ
│   ├── PA2_L1.py                       # CSPA₂-ℓᴵ
│   ├── PA2_L2.py                       # CSPA₂-ℓᴵᴵ
│   ├── CSRDA_1.py                      # CSRDA-I (Chen et al., 2021)
│   ├── CSRDA_2.py                      # CSRDA-II (Chen et al., 2021)
│   ├── CSTG_1.py                       # CSTG-I (Chen et al., 2017)
│   ├── CSTG_2.py                       # CSTG-II (Chen et al., 2017)
│   └── ...                             # Additional algorithm variants
├── data/                               # Generated datasets
│   ├── abtype1/                        # Up-trend datasets
│   │   ├── abtype1_w10_t0.005.libsvm
│   │   ├── abtype1_w10_t0.02.libsvm
│   │   └── ...                         # 760 files (19 window lengths × 40 parameters)
│   ├── abtype2/                        # Down-trend datasets
│   ├── abtype3/                        # Up-shift datasets
│   ├── abtype4/                        # Down-shift datasets
│   ├── abtype5/                        # Systematic datasets
│   ├── abtype6/                        # Cyclic datasets
│   └── abtype7/                        # Stratification datasets
├── best_hyperparameters/               # Optimized hyperparameters per dataset
│   ├── abtype1/
│   │   ├── abtype1_w100_t0.005.csv
│   │   └── ...
│   ├── abtype2/
│   └── ...
├── results/                            # Experimental results (CSV per dataset)
│   ├── abtype1/
│   │   ├── abtype1_w100_t0.005.csv
│   │   └── ...
│   ├── abtype2/
│   └── ...
├── evaluate_model/                     # Model evaluation scripts
├── kernels/
│   └── Kernels.py                      # Kernel function implementations
├── regularizers/
│   └── Regularizer.py                  # Regularization functions
├── data_generator.py                   # Synthetic data generation
├── generate_datasets.py                # Batch dataset creation for one pattern type
├── generate_all_datasets.py            # Batch dataset creation for all pattern types
├── run.py                              # Run a single algorithm on a single dataset
├── compare.py                          # Run all algorithms on a single dataset
├── compare_all.py                      # Run all algorithms on all datasets for a pattern type
├── compare_all.sh                      # SLURM batch script for distributed computing
├── ol_train.py                         # Core training pipeline with hyperparameter optimization
├── CV_algorithm.py                     # Cross-validation and hyperparameter tuning
├── init_model.py                       # Model initialization
├── load_data.py                        # Data loading utilities
├── plot.py                             # Core plotting functions
├── plot_heatmaps.py                    # Heatmap generation
├── plot_gmean.py                       # G-Mean plots
├── plot_time.py                        # Execution time plots
├── plot_metrics.py                     # Performance metric plots
├── plot_metrics_PA.py                  # PA-focused metric plots
├── plot_cer.py                         # Cumulative error rate plots
├── plot_collage.py                     # Interactive HTML collage
├── plot_performance.py                 # Performance summary plots
├── plot_all.py                         # Parallel batch plotting
└── requirements.txt                    # Python dependencies
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{okafor2026cost,
  title={Cost-sensitive online learning for control chart pattern recognition},
  author={Okafor, Paul and Razzaghi, Talayeh},
  journal={Computers \& Industrial Engineering},
  pages={112030},
  year={2026},
  publisher={Elsevier}
}
```
