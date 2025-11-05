#!/bin/bash

noise=$1
bias=$2
use_estimated_ps=$3
method=$4

echo "Running with prediction_noise=${noise}, prediction_bias=${bias}, use_estimated_ps=${use_estimated_ps}, method=${method}"

module load Rtidyverse/4.2.0

PYTHON_BIN=${PYTHON_BIN:-python3}

"$PYTHON_BIN" general_missing_linear_regression.py --use_estimated_ps True --prediction_noise "$noise" --prediction_bias "$bias" --methods "$method" --mispecified_ps_type 1
