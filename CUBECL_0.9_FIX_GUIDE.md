# CubeCL 0.9 Migration Fix Guide

## API Changes Summary

### 1. ABSOLUTE_POS Type Change
```rust
// OLD (broken)
let idx = ABSOLUTE_POS;  // Was u32
if idx < len { ... }     // len is u32 comptime param

// NEW (fixed)
let idx = ABSOLUTE_POS;        // Now returns usize
if idx < (len as usize) { ... } // Cast u32 to usize for comparison
```

### 2. Array Indexing
```rust
// OLD (broken)
output[idx as usize] = val;  // idx was u32, needed cast

// NEW (fixed)
output[idx] = val;  // idx is already usize, no cast needed
```

### 3. Sync Function
```rust
// OLD (broken)
sync_units();

// NEW (fixed)
sync_cube();
```

### 4. CubeDim Constructor
```rust
// OLD (broken)
let cube_dim = CubeDim::new(BLOCK_SIZE, 1, 1);

// NEW (fixed)
let cube_dim = CubeDim::new(&client, BLOCK_SIZE as usize);
// OR for 2D/3D:
let cube_dim = CubeDim::new_2d(x, y);
let cube_dim = CubeDim::new_3d(x, y, z);
```

### 5. Launch API (Safe Version)
```rust
// OLD (broken - used unsafe launch_unchecked)
unsafe {
    my_kernel::launch_unchecked::<i32, CudaRuntime>(
        &client,
        cube_count,
        cube_dim,
        ArrayArg::from_raw_parts(...),
        ...
    );
}

// NEW (fixed - safe launch)
my_kernel::launch::<i32, CudaRuntime>(
    &client,
    cube_count,
    cube_dim,
    ArrayArg::from_raw_parts(...),
    ...
);
// Note: No unsafe block needed!
```

### 6. For Loop Bounds
```rust
// OLD (broken - runtime bounds)
for i in start..end { ... }

// NEW (fixed - comptime bounds with runtime check)
for i in 0u32..comptime_max {
    if (i as usize) < end {
        // actual work
    }
}
```

### 7. Mixed usize/u32 Arithmetic
```rust
// OLD (broken)
let packed_idx = idx / 8u32;  // idx is usize
let offset = base + idx;       // base is u32, idx is usize

// NEW (fixed)
let packed_idx = idx / 8usize;           // Keep usize
let offset = (base as usize) + idx;      // Cast u32 to usize
// OR if result must be u32:
let offset = base + (idx as u32);        // Cast usize to u32
```

### 8. Shift Operations (need u32)
```rust
// OLD (broken)
let shifted = value << (idx * 4usize);  // idx is usize

// NEW (fixed)
let shifted = value << ((idx * 4usize) as u32);  // Cast to u32 for shift
```

### 9. CoreError Variants (Changed in CubeCL 0.9)
```rust
// OLD (broken)
CoreError::dimension_mismatch(...)
CoreError::kernel_error(...)

// NEW (fixed - check CubeCL 0.9 source for correct variants)
// These variants may have been removed or renamed
// Check: ~/.cargo/registry/.../cubecl-core-0.9.0/src/error.rs
```

## File-Specific Issues

### trit-vsa/src/gpu/kernels.rs (178 errors)
- 124x type mismatches (usize vs u32)
- 24x sync_units → sync_cube
- Multiple for loop bound issues

### trit-vsa/src/gpu/ops.rs (54 errors)
- 8x CubeDim::new() signature
- 8x unsafe blocks (remove unsafe, use launch not launch_unchecked)
- CoreError variant issues

### unsloth-rs/src/kernels/fused_rmsnorm_rope.rs (51 errors)
- Array index type mismatches
- Similar usize/u32 issues

### unsloth-rs/src/kernels/fused_swiglu.rs (35 errors)
- Array index type mismatches

### unsloth-rs/src/kernels/cubecl/kernel.rs (23 errors)
- Array index type mismatches

## Verification Commands

After fixing a file:
```bash
# Check specific crate
CUDA_COMPUTE_CAP=90 cargo check -p trit-vsa --features cuda 2>&1 | grep "filename.rs"
CUDA_COMPUTE_CAP=90 cargo check -p unsloth-rs --features cuda 2>&1 | grep "filename.rs"

# Full workspace check
CUDA_COMPUTE_CAP=90 cargo check --workspace --features cuda
```
