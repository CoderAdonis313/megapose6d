#!/usr/bin/env bash

echo "Deleting outputs & visualizations folders"
rm -rf local_data/examples/"${1:-baygon}"/outputs
rm -rf local_data/examples/"${1:-baygon}"/visualizations

export PYTHONDONTWRITEBYTECODE=1
echo "Drawing input coordinates"
python -m megapose.scripts.run_inference_on_example "${1:-baygon}" --vis-detections
echo "Successful input coordinates"

sleep 1

echo "Running inference" 
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
python -m megapose.scripts.run_inference_on_example "${1:-baygon}" --run-inference --model megapose-1.0-RGB
echo "Successful inference"

sleep 1

echo "Drawing visualizations"
python -m megapose.scripts.run_inference_on_example "${1:-baygon}" --vis-outputs
echo "Successful Visualizations"
