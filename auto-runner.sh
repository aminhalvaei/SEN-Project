#!/bin/bash

# Parameter sequence
params=(0 0.0001 0.00025 0.0005 0.00075 0.001 0.0025 0.005 0.0075 0.01 0.025 0.05 0.075 0.1)

# Loop over parameters
for p in "${params[@]}"; do
    echo "=============================="
    echo " Running with Decay parameter: $p "
    echo "=============================="
    
    # Construct log filename
    log_file="logs/run_param_${p}.log"
    mkdir -p logs
    
    # Run command and save + show log
    bash run.sh train RotatE FB15k-237 0 0 1024 256 1000 9.0 1.0 0.00005 5000 16 1.0 "$p" -de 2>&1 | tee "$log_file"
    
    echo "Finished run with $p. Log saved to $log_file"
done
