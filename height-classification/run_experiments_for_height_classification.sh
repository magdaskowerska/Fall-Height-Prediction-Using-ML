#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date +"%d%m%Y_%H%M%S")
LOG_FILE="$SCRIPT_DIR/logs/pipeline_$TIMESTAMP.log"
RESULTS_DIRECTORY="$SCRIPT_DIR/results/$TIMESTAMP"


python3 "$SCRIPT_DIR/scripts/run_model_experiments.py" \
        --model-config "$SCRIPT_DIR/configs/models_config.json" \
        --data-config "$SCRIPT_DIR/configs/data_config.json" \
        --log-file "$LOG_FILE" \
        --results-directory "$RESULTS_DIRECTORY" \
        --script-directory "$SCRIPT_DIR"