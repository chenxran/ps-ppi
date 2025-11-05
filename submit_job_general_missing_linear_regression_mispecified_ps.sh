#!/bin/bash

# Define an array of parameter values.
values=(0 0.2 0.4 0.6 0.8 1.0 1.5 2.0)
methods=(PS-PPI)

module load Rtidyverse/4.2.0

# Loop over each combination for jobs with --use_estimated_ps True.
for noise in "${values[@]}"; do
    for bias in "${values[@]}"; do
        for method in "${methods[@]}"; do
            echo "Submitting job with prediction_noise=${noise}, prediction_bias=${bias}, method=${method}, use_estimated_ps=True"
            sbatch run_general_missing_linear_regression_mispecified_ps.sh "$noise" "$bias" "True" "$method"
        done
    done
done

echo "Submitting job with method=WCCA, use_estimated_ps=True"
sbatch run_general_missing_linear_regression_mispecified_ps.sh "0" "0" "True" "WCCA"