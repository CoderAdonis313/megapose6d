# COMMANDS

### Visualize input detections

```bash
python -m megapose.scripts.run_inference_on_example baygon --vis-detections
```

### Run inference

```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
python -m megapose.scripts.run_inference_on_example baygon --run-inference --model megapose-1.0-RGB
```

### Visualize output detections

```bash
python -m megapose.scripts.run_inference_on_example baygon --vis-outputs
```