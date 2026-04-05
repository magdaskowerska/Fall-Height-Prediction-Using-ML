#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DATA_GENERATION=false
RUN_MODEL_EXPERIMENTS=false
TIMESTAMP=$(date +"%d%m%Y_%H%M%S")
LOG_FILE="$SCRIPT_DIR/logs/pipeline_$TIMESTAMP.log"
RESULTS_DIRECTORY="$SCRIPT_DIR/results/$TIMESTAMP"
DATA_DIRECTORY="$SCRIPT_DIR/results/$2"

case "$1" in
    --data-generation)
        RUN_DATA_GENERATION=true
        RUN_MODEL_EXPERIMENTS=false
        ;;
    --models-experiment)
        RUN_DATA_GENERATION=false
        RUN_MODEL_EXPERIMENTS=true
        ;;
    --both)
        RUN_DATA_GENERATION=true
        RUN_MODEL_EXPERIMENTS=true
        ;;
    *)
        echo "Usage: $0 [--data-generation | --models-experiment | --both]"
        exit 1
        ;;
esac

mkdir -p "$SCRIPT_DIR/logs"

if [[ "$RUN_DATA_GENERATION" == "true" ]]; then
    python3 "$SCRIPT_DIR/scripts/run_datasets_generation.py" \
        --data-config "$SCRIPT_DIR/configs/height_cutoff_config.json" \
        --log-file "$LOG_FILE" \
        --results-directory "$RESULTS_DIRECTORY"
fi

if [[ "$RUN_MODEL_EXPERIMENTS" == "true" ]]; then
    CMD="python3 $SCRIPT_DIR/scripts/run_model_experiments.py \
        --model-config $SCRIPT_DIR/configs/models_config.json \
        --log-file $LOG_FILE \
        --results-directory $RESULTS_DIRECTORY"

    if [[ "$DATA_DIRECTORY" != "$SCRIPT_DIR/results/" ]]; then
        CMD="$CMD --data-directory $DATA_DIRECTORY"
    fi
    
    eval $CMD
fi