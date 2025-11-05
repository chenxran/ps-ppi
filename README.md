# PS-PPI

This repository is the official repository of the paper "[A Unified Framework for Inference with General Missingness Patterns and Machine Learning Imputation](https://arxiv.org/pdf/2508.15162)".

## Prerequisites

### Environment Setup

The simulation code is based on Python. However, if you would like to run simulations with multiple imputation, you need to set up both a Python and R environment. If you are not familiar with R or Python/Conda, please refer to their documentations as follows:
- [R Installation Guide](https://www.r-project.org/)
- [Conda Installation Guide](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

To setup the environments for Python, we recommend using `conda` to create an environment:

```bash
conda env create -f environment.yml
conda activate ps-ppi
```

For R, ensure that you have the 3.16.0 verison of the `mice` package installed. You can do this by running the following commands in R.

```R
install.packages("mice")
```

### Other necessary packages

If you want to run the simulations with the official implementation of the PPI++ methods, you should install the `ppi_py` package thorugh [here](https://github.com/aangelopoulos/ppi_py). We also reimplemented the PPI++ methods based on the framework of PS-PPI, which is the method called `PPI++_reimp` in the code.

## Running Simulation

We provide the following example command to showcase how to run the proposed PS-PPI method on the linear regression settings:

```bash
python general_missing_linear_regression.py \
  --use_estimated_ps True \
  --prediction_noise 0.2 \
  --prediction_bias 0.1 \
  --methods PS-PPI
```

By running the above command, the script will run a simulation study with the specified settings and output the results into the `results/` directory.


### Key CLI Options

- `--methods` – comma-separated list of estimators to evaluate (`WCCA`, `PPI++`, `PS-PPI`, `CCA`, `MI`, `PPI++_reimp`, `PS-PPI_closedform`). Note that `PS-PPI_closedform` is a variant of PS-PPI that uses closed-form imputation variance estiamtion for linear regression to speed up the computation.
- `--n_simulations` – number of repeated simulations.
- `--prediction_noise` / `--prediction_bias` – perturbations injected into the surrogate predictors.
- `--use_estimated_ps` – enable propensity score estimation inside the simulator; paired with `--mispecified_ps_type` for misspecification experiments.
- `--missing_mechanism` – missingness type (`MAR` or `MCAR`).

Run `python general_missing_linear_regression.py --help` for the full list of command-line flags.

## Batch Job Submission

To enable running simulation in batch through a job scheduler (e.g., SLURM), we provide example shell scripts `submit_job_general_missing_linear_regression.sh` and `submit_job_general_missing_linear_regression_mispecified_ps.sh`. You can modify the parameters inside these scripts to run the simulations with different settings as needed.
