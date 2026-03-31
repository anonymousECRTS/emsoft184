#!/usr/bin/env bash
set -u

PYTHON_BIN="python"
SCRIPT="final-code.py"
ROOT_DIR="./deadlock-000/"
OUTPUT_FILE="deadlock-000.txt"

: > "$OUTPUT_FILE"

find "$ROOT_DIR" -type f -name "benchmark.json" | sort | while IFS= read -r json_file; do
    echo "Running: $json_file"

    run_output="$("$PYTHON_BIN" "$SCRIPT" --json "$json_file" --quiet --no-plot --no-sim 2>&1)"

    solver_times="$(printf '%s\n' "$run_output" \
        | grep -oE 'Analysis execution time:[[:space:]]*[0-9]+\.[0-9]+[[:space:]]*s' \
        | sed -E 's/Analysis execution time:[[:space:]]*([0-9]+\.[0-9]+)[[:space:]]*s/\1/')"

    if [ -n "$solver_times" ]; then
        times_list="$(printf '%s\n' "$solver_times" | paste -sd ', ' -)"
        echo "$json_file: [$times_list]" >> "$OUTPUT_FILE"
    else
        echo "$json_file: []" >> "$OUTPUT_FILE"
    fi
done

echo "Done. Results saved in $OUTPUT_FILE"