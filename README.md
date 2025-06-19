# Cost-Sensitive Control Chart Pattern Recognition (CCPR)

A comprehensive framework for cost-sensitive online learning algorithms applied to control chart pattern recognition, implementing novel variants of Passive-Aggressive (PA) and Online Gradient Descent (OGD) algorithms for highly imbalanced time-series classification.

## Project Overview

This project implements and evaluates cost-sensitive online learning algorithms specifically designed for Control Chart Pattern Recognition (CCPR) in manufacturing quality control. The framework addresses the challenge of detecting abnormal patterns in control charts where normal samples vastly outnumber abnormal ones (e.g., 900:100 ratio), making traditional accuracy-based metrics inadequate.

## Key Features

- **Cost-Sensitive Algorithms**: Novel implementations of CSPA (Cost-Sensitive Passive-Aggressive) and CSOGD (Cost-Sensitive Online Gradient Descent)
- **Multiple Loss Functions**: Support for ℓI and ℓII cost-sensitive loss functions in addition to standard hinge loss
- **Synthetic Data Generation**: Comprehensive time-series data generator with 7 abnormal pattern types
- **Performance Evaluation**: Extensive metrics including G-Mean, Cost-Sensitive Sum (CSS), sensitivity, and specificity
- **Visualization Tools**: Complete plotting framework for performance analysis and comparison
- **Hyperparameter Optimization**: Bayesian optimization using Optuna for automated parameter tuning


## Algorithm Analysis

Here's a breakdown of each Python file and their characteristics:

### **Passive-Aggressive (PA) Algorithms:**

#### **PA.py**
- **Loss Function**: Hinge Loss (`l_t = max(0,1-y_t*f_t)`)
- **Regularization**: No C parameter (basic PA variant)
- **Description**: Standard PA algorithm with unbounded step size

#### **PA1.py** 
- **Loss Function**: Hinge Loss
- **Regularization**: Single C parameter (`gamma_t = min(C, l_t / s_t)`)
- **Description**: PA-I variant with bounded step size using C

#### **PA2.py**
- **Loss Function**: Hinge Loss  
- **Regularization**: Single C parameter with quadratic penalty (`gamma_t = l_t / (s_t + (1 / (2 * C)))`)
- **Description**: PA-II variant with quadratic regularization term

#### **PA1_Csplit.py**
- **Loss Function**: Hinge Loss
- **Regularization**: C_positive and C_negative (`C_pos = C / num_positive`, `C_neg = C / num_negative`)
- **Description**: PA-I variant with class-specific regularization parameters

#### **PA2_Csplit.py**
- **Loss Function**: Hinge Loss
- **Regularization**: C_positive and C_negative with quadratic penalty
- **Description**: PA-II variant with class-specific regularization parameters

#### **PA_L1.py**
- **Loss Function**: Cost-Sensitive Hinge Loss Type I (`l_t = max(0, (rho if y_t == 1 else 1) - y_t * f_t)`)
- **Regularization**: No C parameter
- **Description**: Basic PA with L1-type cost-sensitive loss

#### **PA_L2.py**
- **Loss Function**: Cost-Sensitive Hinge Loss Type II (`l_t = (rho if y_t == 1 else 1) * max(0, 1 - y_t * f_t)`)
- **Regularization**: No C parameter
- **Description**: Basic PA with L2-type cost-sensitive loss

#### **PA1_L1.py**
- **Loss Function**: Cost-Sensitive Hinge Loss Type I (L1-type)
- **Regularization**: C_positive and C_negative
- **Description**: PA-I variant with L1-type cost-sensitive loss and class-specific C parameters

#### **PA1_L2.py**
- **Loss Function**: Cost-Sensitive Hinge Loss Type II (L2-type)
- **Regularization**: C_positive and C_negative
- **Description**: PA-I variant with L2-type cost-sensitive loss and class-specific C parameters

#### **PA2_L1.py**
- **Loss Function**: Cost-Sensitive Hinge Loss Type I (L1-type)
- **Regularization**: C_positive and C_negative with quadratic penalty
- **Description**: PA-II variant with L1-type cost-sensitive loss and class-specific C parameters

#### **PA2_L2.py**
- **Loss Function**: Cost-Sensitive Hinge Loss Type II (L2-type)
- **Regularization**: C_positive and C_negative with quadratic penalty
- **Description**: PA-II variant with L2-type cost-sensitive loss and class-specific C parameters

#### **PA_I_L1.py**
- **Loss Function**: Cost-Sensitive Hinge Loss Type I (L1-type)
- **Regularization**: Single C parameter
- **Description**: PA-I variant with L1-type cost-sensitive loss and standard C regularization

#### **PA_I_L2.py**
- **Loss Function**: Cost-Sensitive Hinge Loss Type II (L2-type)
- **Regularization**: Single C parameter
- **Description**: PA-I variant with L2-type cost-sensitive loss and standard C regularization

#### **PA_II_L1.py**
- **Loss Function**: Cost-Sensitive Hinge Loss Type I (L1-type)
- **Regularization**: Single C parameter with quadratic penalty
- **Description**: PA-II variant with L1-type cost-sensitive loss and quadratic regularization

#### **PA_II_L2.py**
- **Loss Function**: Cost-Sensitive Hinge Loss Type II (L2-type)
- **Regularization**: Single C parameter with quadratic penalty
- **Description**: PA-II variant with L2-type cost-sensitive loss and quadratic regularization

### **Online Gradient Descent (OGD) Algorithms:**

#### **OGD.py**
- **Loss Function**: Hinge Loss (when loss_type=1)
- **Regularization**: Single C parameter (used as learning rate)
- **Description**: Standard OGD with multiple loss type options (0-1, hinge, logistic, square)

#### **OGD_1.py**
- **Loss Function**: Cost-Sensitive Hinge Loss Type I (L1-type)
- **Regularization**: Single C parameter (used as learning rate)
- **Description**: OGD with L1-type cost-sensitive loss and multiple loss options

#### **OGD_2.py**
- **Loss Function**: Cost-Sensitive Hinge Loss Type II (L2-type)
- **Regularization**: Single C parameter (used as learning rate)
- **Description**: OGD with L2-type cost-sensitive loss and multiple loss options

### **Key Distinctions:**

**L1 vs L2 Loss Types:**
- **L1 (Type I)**: `l_t = max(0, (rho if y_t == 1 else 1) - y_t * f_t)`
- **L2 (Type II)**: `l_t = (rho if y_t == 1 else 1) * max(0, 1 - y_t * f_t)`

**PA Variant Types:**
- **PA**: No regularization bounds
- **PA-I**: Bounded step size with `min(C, l_t/s_t)`
- **PA-II**: Quadratic penalty term `1/(2*C)` in denominator

**C Parameter Types:**
- **Single C**: Standard regularization parameter
- **C_split**: Class-specific C_pos and C_neg parameters


## Installation

### Prerequisites

```bash
pip install -r requirements.txt
```

### Required Dependencies

```
numpy
pandas
joblib
optuna
matplotlib
pyswarm
scikit-learn
scipy
sqlalchemy
pytz
plotly
```

## Data Generation

The framework includes a sophisticated synthetic time-series data generator that creates control chart patterns with varying degrees of abnormality.

### Abnormal Pattern Types

1. **Uptrend** (abtype=1): Increasing slope k ∈ [0.005r, 0.605r]
2. **Downtrend** (abtype=2): Decreasing slope k ∈ [0.005r, 0.605r]
3. **Upshift** (abtype=3): Mean shift x ∈ [0.005r, 1.805r]
4. **Downshift** (abtype=4): Mean shift x ∈ [0.005r, 1.805r]
5. **Systematic** (abtype=5): Alternating pattern k ∈ [0.005r, 1.805r]
6. **Cyclic** (abtype=6): Sinusoidal pattern a ∈ [0.005r, 1.805r]
7. **Stratification** (abtype=7): Variance change σ ∈ [0.005r, 0.8r]

### Generate Single Dataset

```bash
python data_generator.py -t bc -d binary_synthetic_data.libsvm -w 48 --t 0.5 -a 900 -b 100 -m 1 --abtype 1 --normalize_abnormal
```

**Parameters:**
- `-t bc`: Binary classification
- `-d`: Output file path
- `-w 48`: Window length (time-series length)
- `--t 0.5`: Abnormal parameter magnitude
- `-a 900`: Number of normal samples
- `-b 100`: Number of abnormal samples
- `--abtype 1`: Abnormal pattern type (1-7)
- `--normalize_abnormal`: Normalize abnormal samples to unit norm

### Generate Multiple Datasets

```bash
python generate_datasets.py 1  # Generate all combinations for abtype 1
```

This creates datasets with:
- Window lengths: w ∈ [10, 15, 20, ..., 100] (19 values)
- Abnormal parameters: 40 unique values per abtype within specified ranges
- Total: 19 × 40 = 760 datasets per abnormal type

## Training and Evaluation

### Single Algorithm Execution

```bash
python run.py -t bc -a PA1_L1 -d data/abtype1/abtype1_w50_t0.5.libsvm -f libsvm -n 20
```

**Parameters:**
- `-t bc`: Task type (binary classification)
- `-a PA1_L1`: Algorithm name
- `-d`: Dataset path
- `-f libsvm`: File format
- `-n 20`: Number of independent runs

### Algorithm Comparison Framework

The project includes a comprehensive comparison system for evaluating all algorithms systematically:

#### Compare All Algorithms on Single Dataset

```bash
python compare.py -t bc -d data/abtype1/abtype1_w50_t0.5.libsvm -f libsvm -n 20 -s results/abtype1
```

This runs all 20+ algorithms (PA, PA1, PA2, OGD, OGD_1, OGD_2, PA1_L1, PA1_L2, PA2_L1, PA2_L2, PA1_Csplit, PA2_Csplit, PA_L1, PA_L2, PA_I_L1, PA_I_L2, PA_II_L1, PA_II_L2, Gaussian_Kernel_Perceptron, Gaussian_Kernel_OGD) on a single dataset.

#### Batch Comparison Across All Datasets

```bash
python compare_all.py 1  # Run all algorithms on all abtype1 datasets
```

**Features:**
- **Parallel Processing**: Uses ThreadPoolExecutor for concurrent execution
- **Chunked Processing**: Processes datasets in chunks of 5 for optimal resource usage
- **Automatic Results Organization**: Creates organized directory structure in `results/abtype{X}/`
- **Comprehensive Coverage**: Runs all 760 datasets per abnormal type (19 window lengths × 40 parameter values)

#### Distributed Computing Support

For large-scale experiments, the framework includes SLURM batch scripts:

```bash
sbatch compare_all.sh  # Submit batch job for distributed computing
```

### Using the Main Interface

```bash
python ol_train.py
```

The training pipeline includes:
1. **Data Loading**: Automatic dataset parsing and preprocessing
2. **Hyperparameter Optimization**: Bayesian optimization using Optuna
3. **Cross-Validation**: 5-fold CV for robust parameter selection
4. **Performance Tracking**: Real-time metrics computation at regular intervals
5. **Results Storage**: Automatic saving of model parameters and performance metrics

### Hyperparameter Optimization

The framework uses Optuna for efficient hyperparameter tuning:

```python
# Automatically optimized parameters:
C ∈ [2^-4, 2^7]           # Regularization parameter
eta_p, eta_n ∈ [0, 1]     # Cost-sensitive weights (eta_p + eta_n = 1)
```

## Performance Metrics

The framework evaluates algorithms using multiple metrics appropriate for imbalanced classification:

### Primary Metrics
- **Sensitivity (Recall)**: True Positive Rate
- **Specificity**: True Negative Rate  
- **G-Mean**: √(Sensitivity × Specificity)
- **Cost-Sensitive Sum (CSS)**: η_p × Sensitivity + η_n × Specificity

### Additional Metrics
- **Accuracy**: Overall classification accuracy
- **Precision**: Positive predictive value
- **Cohen's Kappa**: Inter-rater agreement
- **Matthews Correlation Coefficient (MCC)**: Balanced metric for binary classification
- **Cumulative Error Rate**: Running mistake rate
- **Execution Time**: Computational efficiency

## Visualization and Analysis

The framework provides a comprehensive suite of visualization tools for analyzing algorithm performance across different dimensions:

### Core Plotting Scripts

#### 1. Performance Metrics Over Time (`plot_metrics.py`)
```bash
python plot_metrics.py
```
**Output**: Individual performance plots for all 12 metrics across all datasets and algorithms
- **Metrics**: Captured Time, Mistakes, Updates, Time, Accuracies, Sensitivities, Specificities, Precisions, G-Means, Kappas, MCCs, Cumulative Errors
- **Organization**: `plots_metrics/abtype{X}/{metric}/{dataset}.png`
- **Features**: Algorithm comparison with error bands over training samples

#### 2. G-Mean Performance Analysis (`plot_gmean.py`)
```bash
python plot_gmean.py
```
**Output**: G-Mean performance across parameter space for each algorithm
- **Visualization**: Line plots showing G-Mean vs abnormal parameter for different window lengths
- **Coverage**: All 7 abnormal types × 14+ algorithms
- **Organization**: `plot_G-Means/abtype{X}/abtype{X}_{algorithm}_G-Mean.png`

#### 3. Execution Time Analysis (`plot_time.py`)
```bash
python plot_time.py
```
**Output**: Computational efficiency analysis
- **Visualization**: Execution time vs abnormal parameter with window length comparison
- **Features**: Algorithm scalability assessment across parameter ranges
- **Organization**: `plot_time/abtype{X}/abtype{X}_{algorithm}_Time.png`

#### 4. Comprehensive Heatmaps (`plot_heatmaps.py`)
```bash
python plot_heatmaps.py
```
**Output**: Performance heatmaps for all metrics and algorithms
- **Visualization**: Window Length (y-axis) vs Abnormal Parameter (x-axis) heatmaps
- **Metrics**: All 12 performance metrics with color-coded intensity
- **Coverage**: 7 abtypes × 18+ algorithms × 12 metrics = 1,500+ heatmaps
- **Organization**: `plots_HeatMap/abtype{X}/{algorithm}/{metric}.png`
- **Features**: 
  - Automatic scaling (0-1 for bounded metrics, auto for others)
  - High-resolution visualization with customizable colormaps
  - Statistical significance through color intensity

#### 5. Performance Collages (`plot_collage.py`)
```bash
python plot_collage.py
```
**Output**: Interactive HTML grids for algorithm comparison
- **Visualization**: Browser-based scrollable tables of heatmap images
- **Features**:
  - Side-by-side algorithm comparison
  - Interactive navigation through abnormal types
  - Exportable HTML format
  - Customizable image dimensions
- **Algorithm Mapping**: Automatic renaming (e.g., PA1_Csplit → CSPA-I)

### Batch Visualization Processing

#### Generate All Plots (`plot_all.py`)
```bash
python plot_all.py
```
**Execution**: Parallel processing of multiple visualization scripts
- **Scripts Included**:
  - `plot_metrics_PA.py`: Passive-Aggressive algorithm focus
  - `plot_time.py`: Execution time analysis  
  - `plot_gmean.py`: G-Mean performance
  - `plot_time2.py`: Extended time analysis
  - `plot_gmean2.py`: Secondary G-Mean analysis
- **Features**: Multi-threaded execution using CPU cores

### Specialized Plotting Tools

#### 1. Algorithm-Specific Performance (`plot_metrics_PA.py`)
**Focus**: Detailed analysis of Passive-Aggressive variants
**Coverage**: PA, PA1, PA2, and all cost-sensitive PA variants

#### 2. Extended Performance Analysis (`plot_gmean2.py`, `plot_time2.py`)
**Purpose**: Secondary analysis with different parameter configurations
**Features**: Alternative visualization styles and parameter ranges

### Visualization Output Structure

```
plots_metrics/           # Individual metric plots
├── abtype1/
│   ├── Accuracies/     # Accuracy plots for all datasets
│   ├── G-Mean/         # G-Mean plots for all datasets
│   └── ...             # All 12 metrics
plots_HeatMap/           # Comprehensive heatmaps
├── abtype1/
│   ├── PA/             # PA algorithm heatmaps
│   │   ├── G-Mean.png
│   │   ├── Time.png
│   │   └── ...         # All 12 metrics
│   ├── PA1_L1/         # CSPA-I-L1 heatmaps
│   └── ...             # All algorithms
plot_G-Means/           # G-Mean parameter analysis
plot_time/              # Execution time analysis
output.html             # Interactive collage viewer
```

### Usage Examples

```bash
# Complete visualization pipeline
python plot_all.py

# Individual analysis components
python plot_heatmaps.py     # Generate all heatmaps
python plot_gmean.py        # G-Mean parameter analysis
python plot_time.py         # Time complexity analysis
python plot_metrics.py      # All metric visualizations
python plot_collage.py      # Interactive comparison tool

# View interactive results
open output.html            # Browse algorithm comparison collages
```

### Visualization Features

1. **Statistical Rigor**: Error bars, confidence intervals, multiple run averaging
2. **Algorithm Comparison**: Side-by-side performance analysis
3. **Parameter Space Exploration**: Systematic coverage of window lengths and abnormal parameters
4. **Multi-Metric Analysis**: 12 performance metrics across all conditions
5. **Interactive Tools**: Browser-based exploration and comparison
6. **Publication Quality**: High-resolution plots with customizable styling
7. **Batch Processing**: Automated generation of thousands of plots
8. **Organized Output**: Hierarchical directory structure for easy navigation

## Project Structure

```
CCPR_project/
├── algorithms/                    # Algorithm implementations
│   ├── PA1_L1.py                 # CSPA-I with ℓI loss
│   ├── PA1_L2.py                 # CSPA-I with ℓII loss
│   ├── OGD_1.py                  # CSOGD-I
│   ├── OGD_2.py                  # CSOGD-II
│   └── ...                       # Other algorithm variants
├── data/                         # Generated datasets
│   ├── abtype1/                  # Uptrend patterns
│   ├── abtype2/                  # Downtrend patterns
│   └── ...                       # Other pattern types
├── best_hyperparameters/         # Optimized parameters per dataset
├── results/                      # Experimental results
├── evaluate_model/               # Model evaluation scripts
├── kernels/                      # Kernel function implementations
├── regularizers/                 # Regularization functions
├── sample_data/                  # Example datasets
├── data_generator.py             # Synthetic data generation
├── generate_datasets.py          # Batch dataset creation  
├── run.py                        # Single algorithm execution
├── compare.py                    # All algorithms on single dataset
├── compare_all.py                # Batch comparison across all datasets
├── ol_train.py                   # Core training pipeline
├── plot_*.py                     # Comprehensive visualization suite
│   ├── plot_all.py              # Parallel batch plotting
│   ├── plot_metrics.py          # Individual metric plots
│   ├── plot_gmean.py            # G-Mean parameter analysis
│   ├── plot_time.py             # Execution time analysis
│   ├── plot_heatmaps.py         # Performance heatmaps
│   ├── plot_collage.py          # Interactive HTML comparisons
│   └── plot_*.py                # Additional specialized plots
├── CV_algorithm.py               # Cross-validation and hyperparameter tuning
├── init_model.py                 # Model initialization
├── load_data.py                  # Data loading utilities
├── plot.py                       # Core plotting functions with statistical analysis
└── requirements.txt              # Python dependencies
```

## Algorithm Details

### Cost-Sensitive Loss Functions

**ℓI Loss Function:**
```
ℓI(ωt; (xt, yt)) = max(0, (ρ·I(yt=1) + I(yt=-1)) - yt(ωt·xt))
```

**ℓII Loss Function:**
```
ℓII(ωt; (xt, yt)) = (ρ·I(yt=1) + I(yt=-1)) × max(0, 1 - yt(ωt·xt))
```

Where ρ = (ηp×Tn)/(ηn×Tp) is the cost-sensitive parameter.

### Update Rules

**CSPA-I Update:**
```
τt = min(Ct, ℓ(ω; (xt, yt))/||xt||²)
ωt+1 = ωt + τt × yt × xt
```

**CSPA-II Update:**
```
τt = ℓ(ω; (xt, yt))/(||xt||² + 1/(2Ct))
ωt+1 = ωt + τt × yt × xt
```

## Research Context

This implementation is part of ongoing research in cost-sensitive online learning for quality control applications. The algorithms address key challenges in manufacturing:

- **Class Imbalance**: Normal samples heavily outnumber abnormal ones
- **Online Learning**: Models must adapt to streaming data in real-time
- **Cost Sensitivity**: Misclassifying abnormal patterns is more costly than false alarms
- **Temporal Dependencies**: Control chart data exhibits time-series characteristics

## Citation

If you use this code in your research, please cite:

```bibtex
@article{ccpr2024,
  title={Cost-Sensitive Online Learning for Control Chart Pattern Recognition},
  author={[Author Names]},
  journal={[Journal Name]},
  year={2024},
  note={Under Review}
}
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-algorithm`)
3. Commit your changes (`git commit -am 'Add new algorithm'`)
4. Push to the branch (`git push origin feature/new-algorithm`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on the LIBOL framework by stevenhoi
- Synthetic data generation inspired by Montgomery's control chart patterns
- Optuna library for hyperparameter optimization
- scikit-learn for baseline metrics and utilities

## Contact

For questions or collaboration opportunities, please contact [contact information].