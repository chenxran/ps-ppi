#!/bin/bash

# Define an array of parameter values.
values=(0 0.2 0.4 0.6 0.8 1.0 1.5 2.0)
methods=(CCA WCCA MI PPI++ PS-PPI)

module load Rtidyverse/4.2.0

# Loop over each combination for jobs with --use_estimated_ps True.
for noise in "${values[@]}"; do
    for bias in "${values[@]}"; do
        for method in "${methods[@]}"; do
            echo "Submitting job with prediction_noise=${noise}, prediction_bias=${bias}, method=${method}, use_estimated_ps=True"
            sbatch run_general_missing_linear_regression.sh "$noise" "$bias" "True" "$method"
        done
    done
done

# Loop over each combination for jobs without --use_estimated_ps flag.
for noise in "${values[@]}"; do
    for bias in "${values[@]}"; do
        for method in "${methods[@]}"; do
            echo "Submitting job with prediction_noise=${noise}, prediction_bias=${bias}, method=${method}, use_estimated_ps=False"
            sbatch run_general_missing_linear_regression.sh "$noise" "$bias" "False" "$method"
        done
    done
done

echo "Submitting job with method=CCA"
sbatch run_general_missing_linear_regression.sh "0" "0" "False" "CCA"

echo "Submitting job with method=WCCA, use_estimated_ps=True"
sbatch run_general_missing_linear_regression.sh "0" "0" "True" "WCCA"

echo "Submitting job with method=WCCA, use_estimated_ps=False"
sbatch run_general_missing_linear_regression.sh "0" "0" "False" "WCCA"

echo "Submitting job with method=MI"
sbatch run_general_missing_linear_regression.sh "0" "0" "False" "MI"