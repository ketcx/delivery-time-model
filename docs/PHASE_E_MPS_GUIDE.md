# Phase E: Mac M1 MPS Optimization Guide

## Overview

This guide covers:
1. MPS (Metal Performance Shaders) availability and usage
2. Data types (FP32, FP16, BF16) and their tradeoffs
3. Common Out-of-Memory (OOM) patterns and solutions
4. Device fallback strategy

---

## 1. MPS Validation and Availability

### Check MPS Support

```python
import torch

# Check if MPS is available
if torch.backends.mps.is_available():
    device = "mps"
    print("✓ MPS is available")
else:
    device = "cpu"
    print("⚠️  MPS not available, using CPU")
```

### In the Project

```python
from src.device import is_mps_available, get_device

# Simple check
if is_mps_available():
    device = "mps"
else:
    device = "cpu"

# With fallback
device = get_device(preferred_device="mps", fallback_on_error=True)
```

---

## 2. Data Types (Dtypes) on M1

### Memory Footprint

| Type | Bytes | Use Case |
|------|-------|----------|
| **FP32** (float32) | 4 bytes | Default, highest precision |
| **FP16** (float16) | 2 bytes | Mixed precision, faster |
| **BF16** (bfloat16) | 2 bytes | Lower precision, some MPS limitations |

### Usage in Project

```yaml
# In configs/train.yaml
precision: fp32  # Options: fp32, fp16, bf16
```

```python
# In code
from src.device import get_dtype_from_string

dtype = get_dtype_from_string("fp32")  # or "fp16", "bf16"

# With autocast (automatic mixed precision)
with torch.autocast(device_type="mps", dtype=torch.float16):
    output = model(input_batch)
    loss = criterion(output, target)
```

### Tradeoffs

**FP32 (Default)**
- ✓ Highest precision
- ✓ Most stable training
- ✗ More memory (4 bytes per value)
- ✗ Slower on some devices
- **Recommendation**: Use for initial training, validation

**FP16**
- ✓ 2x smaller memory
- ✓ Faster computation
- ✗ Risk of numerical instability
- ✗ Gradient underflow possible
- **Recommendation**: Use with automatic mixed precision (AMP)

**BF16**
- ✓ Wider range than FP16
- ✓ More stable than FP16
- ✗ Limited MPS support (check version compatibility)
- **Recommendation**: Experimental on M1, avoid if possible

### MPS Limitations with Dtypes

- **FP16**: Supported but may have precision issues
- **BF16**: Limited support depending on PyTorch/macOS version
- **Recommendation**: Stick with FP32 unless benchmarking shows clear benefits

---

## 3. Out-of-Memory (OOM) Patterns and Solutions

### Common OOM Causes

#### ❌ Pattern 1: Batch Size Too Large

```python
# BAD - Will cause OOM
batch_size = 1024  # Too large for M1

# GOOD - Start conservative
batch_size = 32  # Increase gradually

# Test batch size
for batch_size in [8, 16, 32, 64, 128]:
    try:
        batch = next(iter(dataloader))
        model(batch[0].to(device))
        print(f"✓ Batch size {batch_size} OK")
    except RuntimeError as e:
        print(f"✗ OOM at batch size {batch_size}")
        break
```

#### ❌ Pattern 2: Tensors Accidentally Kept in Memory

```python
# BAD - Keeps all outputs in memory
outputs = []
for batch in dataloader:
    output = model(batch)
    outputs.append(output)  # Accumulates in memory!
loss = criterion(torch.cat(outputs), targets)

# GOOD - Process and discard
total_loss = 0.0
for batch in dataloader:
    output = model(batch)
    loss = criterion(output, targets)
    total_loss += loss.item()  # Keep only scalar
    loss.backward()
    optimizer.step()
```

#### ❌ Pattern 3: Not Clearing Gradients/Cache

```python
# BAD - Gradients accumulate
for batch in dataloader:
    output = model(batch)
    loss = criterion(output, targets)
    loss.backward()  # Gradients accumulate!
    optimizer.step()

# GOOD - Clear before backward pass
for batch in dataloader:
    optimizer.zero_grad()  # Clear old gradients
    output = model(batch)
    loss = criterion(output, targets)
    loss.backward()
    optimizer.step()
    
# ALSO GOOD - Use no_grad() for inference
with torch.no_grad():  # Disables gradient tracking
    output = model(batch)  # No memory overhead for gradients
```

#### ❌ Pattern 4: Large Intermediate Tensors

```python
# BAD - Creates large intermediate
x = torch.randn(1000, 1000, 1000, device=device)
y = x * 2
z = y + 1

# GOOD - In-place operations
x = torch.randn(1000, 1000, 1000, device=device)
x.mul_(2)  # In-place multiplication
x.add_(1)  # In-place addition
```

### OOM Solutions

1. **Reduce batch size**: Most effective, simple to try
2. **Enable gradient checkpointing**: Trade compute for memory
3. **Use mixed precision (AMP)**: Reduce memory with FP16
4. **Reduce model size**: Fewer parameters = less memory
5. **Gradient accumulation**: Simulate larger batches with smaller device batches

### Checking Memory Usage

```python
# On MPS, memory info is limited
# Use torch profiler instead
import torch.profiler as profiler

with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    output = model(batch)
    loss = criterion(output, targets)

prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10)
```

---

## 4. Device Fallback Strategy

### Automatic Fallback

```python
from src.device import get_device

# If MPS fails, automatically use CPU
device = get_device(preferred_device="mps", fallback_on_error=True)
```

### Manual Fallback (For Stability)

```python
from src.device import is_mps_available

# Prefer CPU for stability if MPS has issues
use_mps = is_mps_available() and not KNOWN_STABILITY_ISSUES
device = "mps" if use_mps else "cpu"

print(f"Using device: {device}")
```

### In Config

```yaml
# configs/train.yaml
device: mps  # Will fallback to CPU if unavailable
precision: fp32  # Safest choice for M1
```

### During Training

```python
try:
    output = model(batch)
    loss = criterion(output, targets)
    loss.backward()
    optimizer.step()
except RuntimeError as e:
    if "out of memory" in str(e):
        print("OOM detected, falling back to CPU")
        device = "cpu"
        model = model.to(device)
        batch = batch[0].to(device), batch[1].to(device)
        # Retry with reduced batch size
        reduced_batch_size = current_batch_size // 2
        break
```

---

## 5. Benchmarking

Run the benchmark script:

```bash
cd pytorch-llm-phase0
python scripts/benchmark.py
```

This will:
- Compare MPS vs CPU performance
- Test with FP32 (safest for M1)
- Measure throughput (samples/sec)
- Show warmup impact
- Provide recommendations

### Interpreting Results

```
BENCHMARK COMPARISON
==================================================
Config           | Avg Step (ms) | Throughput
--------------------------------------------------
mps_fp32         |         15.23 |  125.4 samples/s
cpu_fp32         |         25.64 |   74.2 samples/s
==================================================

✓ MPS is 1.69x faster than CPU
```

---

## 6. Best Practices for M1

### DO ✓

- ✓ Use MPS when available (1.5-2x speedup typical)
- ✓ Stick with FP32 for stability
- ✓ Start with small batch sizes, increase gradually
- ✓ Monitor actual throughput with benchmarks
- ✓ Use `torch.no_grad()` for inference
- ✓ Clear gradients with `optimizer.zero_grad()`
- ✓ Fall back to CPU if MPS becomes unstable

### DON'T ✗

- ✗ Don't use FP16/BF16 without testing
- ✗ Don't expect LLM-scale training to work (M1 is for development)
- ✗ Don't accumulate large tensors in Python lists
- ✗ Don't forget `optimizer.zero_grad()`
- ✗ Don't ignore OOM errors - reduce batch size
- ✗ Don't blindly trust performance - benchmark your workload

---

## 7. Example: Stable Training Loop

```python
from src.device import get_device, check_mps_limitations

# Setup
device = get_device(preferred_device="mps", fallback_on_error=True)
dtype = torch.float32  # Safest choice

logger.info(f"Device: {device}, Dtype: {dtype}")

# Check limitations
check_mps_limitations(device, dtype, logger=logger)

# Training loop
for epoch in range(num_epochs):
    for batch_idx, (x, y) in enumerate(train_loader):
        try:
            x, y = x.to(device), y.to(device)
            
            # Forward
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(f"OOM at epoch {epoch}, batch {batch_idx}")
                # Could reduce batch size and continue
                raise
            else:
                raise
```

---

## References

- [PyTorch MPS Backend](https://pytorch.org/docs/stable/notes/mps.html)
- [Apple Metal Performance Shaders](https://developer.apple.com/metal/pytorch/)
- [Mixed Precision Training](https://pytorch.org/docs/stable/amp.html)
