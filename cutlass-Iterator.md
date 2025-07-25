



```

```

```
ldmatrix: 从共享内存加载到寄存器的指令
include/cutlass/arch/memory_sm75.h
```

```
template <>
CUTLASS_DEVICE void ldsm<layout::RowMajor, 1>(
    Array<unsigned, 1> & D,
    void const* ptr) {

  #if defined(CUTE_ARCH_LDSM_SM75_ACTIVATED)

    unsigned addr = cutlass_get_smem_pointer(ptr);

    int x;
    asm volatile ("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];" : "=r"(x) : "r"(addr));
    reinterpret_cast<int &>(D) = x;

  #else

    CUTLASS_UNUSED(D);
    CUTLASS_UNUSED(ptr);
    CUTLASS_NOT_IMPLEMENTED();

  #endif
}
```



```
调用ldsm
include/cutlass/gemm/warp/mma_tensor_op_tile_iterator.h
```

```
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset in units of bytes
      Index byte_offset) const {

    Array<unsigned, Policy::LdsmShape::kCount> *fetch_ptr = 
      reinterpret_cast<Array<unsigned, Policy::LdsmShape::kCount> *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < Policy::LdsmIterations::kStrided; ++s) {

      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < Policy::LdsmIterations::kContiguous; ++c) {

        int access_idx = c + s * Policy::LdsmIterations::kContiguous;

        AccessType const *source_ptr =
            pointer_[c % kPointerCount] +
            Layout::TileShape::kContiguous * (c / kPointerCount) +
            Policy::kLdsmOpInner * Policy::LdsmShape::kStrided * s * stride_ / Layout::kFactor;

        char const *source_byte_ptr = reinterpret_cast<char const *>(source_ptr) + byte_offset + byte_offset_;

        cutlass::arch::ldsm<layout::ColumnMajor, Policy::LdsmShape::kCount>(
          fetch_ptr[access_idx],
          source_byte_ptr
        );
      }
    }
  }
```

```
load_with_pointer_offset 所在的模板类：
class MmaTensorOpMultiplicandTileIterator
class MmaTensorOpAccumulatorTileIterator
```

```
load_with_pointer_offset 还会被load方法调用
```

```
  void load(Fragment &frag) const {
    load_with_pointer_offset(frag, 0);
  }
```

```
iterator_就是MmaTensorOpMultiplicandTileIterator的特化模板
```

```
  /// Underlying tile iterator
  Base iterator_;
  
    /// Underlying tile iterator implementation
  using Base = MmaTensorOpMultiplicandTileIterator<
      layout::PitchLinearShape<Shape::kRow, Shape::kColumn>, kOperand, Element,
      layout::TensorOpMultiplicandCongruous<sizeof_bits<Element_>::value,
                                            kCrosswise>,
      layout::PitchLinearShape<InstructionShape::kRow,
                               InstructionShape::kColumn>,
      kOpDelta, kThreads, PartitionsK_>;
```

