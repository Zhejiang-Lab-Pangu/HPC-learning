kernel 函数

```c++
template <typename Gemm_>
__global__ void gemm_kernel(typename Gemm_::Params params) {
  // Declare shared memory.
  __shared__ typename Gemm_::SharedStorage shared_storage;

  // Construct the GEMM object.
  Gemm_ gemm(params, shared_storage);
  // Run GEMM.
  gemm.multiply_add();
}
```



执行计算核心逻辑：

``` 
/// Do the GEMM.
CUTLASS_DEVICE void multiply_add() {
  // Swizzle the IDs of the block (to enable better cache behavior).
  typename Traits::BlockSwizzle block_swizzle;
  dim3 block = block_swizzle.swizzle();

  // Scale the id.
  block.x *= Traits::OutputTile::kW;
  block.y *= Traits::OutputTile::kH;

  // We may want to use shared memory to clear the registers.
  typedef typename Traits::ClearAccumulators ClearAccumulators;

  // The streams to read A/B from global memory to shared memory.
  typename Traits::GlobalLoadStream global_stream(params, shared_storage, block);

  // Create the accumulator clear.
  ClearAccumulators clear(shared_storage.main_loop.clear);

  /// Define the mainloop iteration size
  typedef typename Traits::MultiplyAdd MultiplyAdd;

  // By how much we unroll the main loop.
  Index const kUnroll = static_cast<Index>(MultiplyAdd::AccumulatorsPerWarp::kD);

  // If we do not have enough steps in the main loop, trigger the residue code.
  if (params.k < kUnroll) {
    global_stream.residue(params.k, true);
  }

  // Fetch the fragments for A and B from global memory.
  // 对全局内存的预取，预取到寄存器
  global_stream.copy();

  // Copy the elements to shared memory (after transformation if needed).
  // 从寄存器取数据到共享内存
  global_stream.commit();

  // Make sure the data is in shared memory.
  Traits::shared_store_fence(false);

  // The unrolling steps for the main loop.
  int const kUnrollingSteps =
      MultiplyAdd::AccumulatorsPerWarp::kD / MultiplyAdd::InstructionShape::kD;

  // Make sure we have at least 2 unrolling steps or our pipeling is not going to work.
  static_assert(kUnrollingSteps >= 2, "The pipelining assumes at least two steps");

  // The stream of data from shared memory to fragments.
  typename Traits::SharedLoadStream shared_load_stream(params, shared_storage);

  // Trigger the copy from shared memory for the 1st stream.
  // 
  shared_load_stream.copy(0);

  // Allocate the accumulators.
  typename MultiplyAdd::Accumulators accumulators;
  // Clear the accumulators.
  clear.clear(accumulators);

  // Enter the main loop and iterate.
  typedef typename Traits::Index Index;
  for (Index outer_k = params.k - kUnroll; outer_k > -kUnroll; outer_k -= kUnroll) {
    // If that's the last "load iteration" update the predicates.
    int const is_residue = outer_k <= kUnroll;
    if (is_residue) {
      global_stream.residue(outer_k);
    }

    // Load data for the next iteration of the main loop.
    global_stream.copy();

    CUTLASS_PRAGMA_UNROLL
    for (int step = 0; step < kUnrollingSteps - 1; ++step) {
      // Trigger the copy from shared memory for the next A/B values.
      shared_load_stream.copy(step + 1);
      // Make sure the values are available for the current iteration to do the multiply-add.
      shared_load_stream.commit(step);

      // Do the math on the fragments of the current iteration.
      MultiplyAdd multiply_add;
      multiply_add.multiply_add(shared_load_stream.fragment_a(step),
                                shared_load_stream.fragment_b(step),
                                accumulators,
                                accumulators);
    }

    // Make sure the data from shared memory has been entirely consumed.
    Traits::shared_load_fence(true);

    // Commit the data in shared memory for A/B.
    global_stream.commit();

    // Make sure the data is in shared memory.
    Traits::shared_store_fence(true);

    // Move to the next stage for the load (if it makes sense).
    shared_load_stream.inc_stage();
    // Trigger the copy from shared memory for the next loop iteration.
    shared_load_stream.copy(0);
    // Make sure the values are available for the current iteration to do the multiply-add.
    shared_load_stream.commit(kUnrollingSteps - 1);

    // Do the math on the fragments of the current iteration.
    MultiplyAdd multiply_add;
    multiply_add.multiply_add(shared_load_stream.fragment_a(kUnrollingSteps - 1),
                              shared_load_stream.fragment_b(kUnrollingSteps - 1),
                              accumulators,
                              accumulators);
  }

  // Epilogue.
  typedef typename Traits::Epilogue Epilogue;
  Epilogue epilogue(params.epilogue, shared_storage.epilogue, params.m, params.n);
  epilogue.epilogue(cutlass::make_Coord(0, block.y, block.x), accumulators);
}


__shaore__ int i[];
i[] = global[];
```



CUDA Core计算逻辑:

```
  // igemm_multiply_add.h
  CUTLASS_DEVICE void multiply_add(FragmentA const& a,
                                   FragmentB const& b,
                                   Accumulators const& c,
                                   Accumulators& d) {
    // The inputs.
    int const* a_int = reinterpret_cast<int const*>(&a[0]);
    int const* b_int = reinterpret_cast<int const*>(&b[0]);

    for (int j = 0; j < AccumulatorsPerThread::kH; ++j) {
      for (int i = 0; i < AccumulatorsPerThread::kW; ++i) {
        asm volatile("dp4a.s32.s32 %0, %1, %2, %3;"
                     : "=r"(d[j * AccumulatorsPerThread::kW + i])
                     : "r"(a_int[i]), "r"(b_int[j]), "r"(c[j * AccumulatorsPerThread::kW + i]));
      }
    }
  }
```



```
  // thread_multiply_add.h
  // Multiply : d = a*b + c.
  CUTLASS_DEVICE void multiply_add(FragmentA const& a,
                                   FragmentB const& b,
                                   Accumulators const& c,
                                   Accumulators& d) {
    for (int j = 0; j < AccumulatorsPerThread::kH; ++j) {
      for (int i = 0; i < AccumulatorsPerThread::kW; ++i) {
        d[j * AccumulatorsPerThread::kW + i] = a[i] * b[j] + c[j * AccumulatorsPerThread::kW + i];
      }
    }
  }
```

Tensor Core计算逻辑:

```
  /// Multiply : d = a*b.
  CUTLASS_DEVICE void multiply_add(FragmentA const& a,
                                   FragmentB const& b,
                                   Accumulators const& c,
                                   Accumulators& d) {
    for (int j = 0; j < Iterations::kH; ++j) {
      for (int i = 0; i < Iterations::kW; ++i) {
        // The input elements.
        ElementA const& elt_a = a[i];
        ElementB const& elt_b = b[j];
        ElementC const& elt_c = c[j * Iterations::kW + i];

        // The output element.
        ElementC& elt_d = d[j * Iterations::kW + i];

        // The wmma instruction.
        nvcuda::wmma::mma_sync(elt_d, elt_a, elt_b, elt_c);
      }
    }
  }
```

