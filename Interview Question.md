原则：不看代码，轻量级地了解各个部分的知识，保证在面试的时候能够回答出对应的问题。记录答案比较简单的总结，临时想可能想不全。

对于库中的特性，让大模型给出简单的例子，把例子看懂。

优先级以探索未知和底层的能力优先，上层C++/Python专注于概念的理解。

先熟悉kernel开发有什么优化的方式，再去具体例子中确定细节，要不然会存在设计-被推翻反复的情况。

TODO：熟悉各种kernel的实现及优化方式。

躺着：探索自己不懂的东西

坐着：整理答案；整理问题；处理比较难理解解决的问题；整理需要写代码的kernel；

如何找到可能问的问题：牛客的面经；问大模型可能的问题（体系结构的常见面试题有哪些）；字节跳动的岗位JD；知乎面经中的关键字问大模型相关的面试题；

简历中的内容也需要重点关注。

### CPP

### 体系结构

- 指令乱序
- 多发射
- 流水线
- 缓存行
- 为什么内存对齐会提升性能：
  - **减少内存访问次数**: 数据对齐CPU只需**一次读取**即可获取完整数据
  - **优化缓存利用率**: 对齐数据更可能完整地位于同一缓存行内，避免跨行访问
  - **硬件设计支持**: 对齐数据允许CPU通过单条指令完成读写
  - **避免额外计算**: 读取非对齐数据时，CPU需执行移位、掩码等操作合并数据块，增加计算开销


### CUDA

- 专注于Tensor Core、大模型写的不好的不对的地方；

- BlockSize大小设置（算密集型的核函数，推荐使用 128 - 512 个线程 / 块  内存密集型的核函数，可考虑使用 256 - 1024 个线程 / 块）：

  - 一定是32的倍数、共享内存、寄存器大小有限、设备最大限制；

  - 大了：同时驻留的线程块数量减少、并发度就不高了、资源竞争、降低延迟隐藏能力、大线程块容易导致不同线程块之间的执行时间差异变大，出现空闲等待的情况；

  - 小了：调度开销增加、并行效率降低、共享内存利用不充分


- GridSize: 覆盖数据量、SM整数倍；

- **延迟-计算-访存**：GPU 一次FMA-4cycle、寄存器访问1cycle、共享内存1-20cycle。

- **流水线冒险**（Pipeline Hazard）是指在处理器流水线（Pipeline）执行指令时，由于指令之间存在某种依赖关系或资源冲突，导致流水线无法连续流畅地执行，从而不得不插入气泡（Bubble，即空操作周期）或暂停流水线的现象。

- **指令执行**的五个基本阶段（取指→译码→执行→访存→写回）。

- **分支发散**： 谓词寄存器、执行空操作。

- **锁步执行**：**线程束内的所有线程严格按照相同的指令序列同步执行**。

- **非锁步执行**：**线程束内的线程在遇到分支时动态分裂为不同的子组**，Ampere 架构 引入的优化机制。

- **volatile**: 

  - **C++**: 不要对变量进行缓存或重排序，每次访问都必须从内存中读取 / 写入

  - **CUDA** : 主要用于**共享内存**（`__shared__`）变量，确保线程对变量的修改立即对同一线程块内的其他线程可见。


- V100、A100、H100有什么新的特性

- 量化数据格式

- 统一地址空间，那一代被提出解决了哪些问题？

- Tensor Core的调用方式：

- 性能： ldmatrix > cp.async > wmma.load.a.sync

- ldmatrix(PTX):  没有**Bank Conflict**

- wmma.load.a.sync

- float2 4 8：

- **TMA** 主要用于在全局内存和共享内存之间进行高效的异步张量数据传输；

- **Wave Quantization**解决方案：

  - **持久化内核（Persistent Kernel）**：固定网格大小为SM数，动态分配任务至空闲SM，避免波量化
  - **Stream-K调度**：将余数任务均匀分配到所有SM，而非集中到少数SM，提升负载均衡

- | 指令 / 内建函数                      | 源数据位置                                      | 目标数据位置                 | 涉及硬件单元                                 | 说明                                                         |
  | ------------------------------------ | ----------------------------------------------- | ---------------------------- | -------------------------------------------- | ------------------------------------------------------------ |
  | `ldmatrix.sync.aligned`              | 共享内存（Shared Memory）                       | 寄存器（Register）           | 共享内存加载单元（Shared Memory Load Units） | 以 warp 为单位加载矩阵数据，常用于 Tensor Core 计算前的数据准备。 |
  | `wmma.load.a.sync`                   | 共享内存或全局内存（Global Memory）             | 寄存器（Register）           | 共享内存加载单元或全局内存加载单元           | 用于将矩阵数据加载到寄存器，供 Tensor Core 使用。            |
  | 数据加载指令`ld`/`cp`/`prefetch`     |                                                 |                              |                                              |                                                              |
  | `cp.async`                           | 全局内存（Global Memory）                       | 共享内存（Shared Memory）    | Tensor Memory Accelerator（TMA）             | Volta中开始支持<br />在 Hopper 架构中，支持异步张量数据传输，减少线程干预。 |
  | ld..const, .global, .local, .shared, |                                                 | 寄存器                       |                                              | 只有同步加载                                                 |
  |                                      |                                                 |                              |                                              |                                                              |
  | prefetch.global.L2                   | - 全局内存                                      | L2                           |                                              |                                                              |
  | prefetch.global.L1                   | - 全局内存                                      | L1                           |                                              | 会经过L2缓存                                                 |
  | `内存合并访问`                       | - 全局内存（Global Memory）<br />- DRAM Burst段 | 寄存器<br />共享内存         |                                              | 对应ld.global.*和ld.async                                    |
  | `向量化加载指令`                     | - 全局内存<br />- 固定内存                      | - 向量寄存器<br />- 共享内存 |                                              | ld.global.v2.u32<br />ld.global.v4.f32                       |
  |                                      |                                                 |                              |                                              |                                                              |
  |                                      |                                                 |                              |                                              |                                                              |
  |                                      |                                                 |                              |                                              |                                                              |
  
- WMMA（Volta）、MMA（Ampere）、WGMMA（Hopper：直接从共享内存加载矩阵操作数、TMA）

  - mma 的缺点就是上述抽象全部没有，warp 内每个线程输入、输出、寄存器，都要自己配对，相对应的 SMEM load/store 可以自己写算法规避 bank conflict。
  - LDMatrix + MMA + STMatrix 可以达到 WMMA一样的计算效果（load_matrix_sync、mma_sync、store_matrix_sync）

  ![image-20250605162222809](/Users/m/Library/Application Support/typora-user-images/image-20250605162222809.png)

- 内存对齐：**`__align__`** **`#pragma pack`**  

- PCIE瓶颈优化：

  - 减少Launch Kernel的次数：CUDA Graphs、Persistent Kernel
  - 减少数据传输次数：大batch; 统一地址空间（UVA）/ 零拷贝

- **CUDA Graphs ** : 

  - 优势：消除多次 CPU-GPU 通信开销。优化命令间依赖关系，减少等待时间。
  - **命令解析延迟**：
    - GPU 命令处理器解析 Kernel Launch 命令，约**50-100 个时钟周期**。
  - **资源分配延迟**：
    - 寄存器和共享内存分配，约**100-300 个时钟周期**（若未预分配）。
  - **SM 调度延迟**：
    - 线程块调度到 SM 并初始化，约**200-500 个时钟周期**。
  - **指令缓存预热**：
    - 首次执行时加载指令到缓存，约**100-200 个时钟周期**。

- **Persistent Kernel**:  

- **Thrust**、 **CUB**、**libcudacxx**  : 

- 不同shape GEMM的优化策略：

- cutlass的参数一般要如何设置：

- 融合过的kernel及实现

- cuda-example中reduce的实现: cg_reduce\multi_warp_cg_reduce

- 全库召回调用cutlass GEMM的代码：

- 时钟周期

  |            操作            |                  延迟（Latency）                   |                      吞吐（Throughput）                      |
  | :------------------------: | :------------------------------------------------: | :----------------------------------------------------------: |
  |         `FMA` FP32         |                 4 cycles（Ampere）                 |     1 instr / cycle / warp（每 SM 通常 16~32 op/cycle）      |
  |       `INT` MUL/ADD        |                      4 cycles                      |                    1 instr / cycle / warp                    |
  |      寄存器Reg → Reg       |               1 cycle（几乎无延迟）                |                                                              |
  |        寄存器间通信        |                     ≤ 1 cycle                      |                                                              |
  | 共享内存-Shared → Reg (ld) |           1–32 cycles（没有bank冲突是1）           | float x = smem[threadIdx.x];<br />需要通过地址解析、bank arbitration(**存储体仲裁**); |
  |           L1缓存           |                        1~32                        |                                                              |
  |           L2缓存           |                       32~64                        |                                                              |
  |          全局内存          |                      400~600                       |                                                              |
  |                            |                                                    |                                                              |
  |        CUDA Graphs         |             约**500-1000 个时钟周期**              |                                                              |
  |      **传统 Launch**       | 通常超过**2000 个时钟周期**（含 CPU-GPU 通信开销） |                                                              |



### 项目

- Bert类模型80%成分：使用tensorrt + FP16 + 调度策略（MPI）
  - tensorrt: CPU 只launch 一次, 模型的执行不需要CPU参与，20-30%；kernel 融合，10%；
  - FP16 : TensorCore 10-20%；
  - Triton调度策略：10-20%
- libtorch做了哪些优化？

- Torch.compile

- 融合过哪些算子？（去TRT里面找）

- DeepGEMM迁移Ada架构




量化的数据格式？ 底数 指数 以及表示的范围

溢出问题如何排查？

Zero策略的阶段和图片

Megatron、DeepSpeed的特性

### CUTLASS

- 优化方法：

  - **hreadBlock-Warp-Thread三级分块**: BlockTile（线程块级）、WarpTile（warp级）和ThreadTile（线程级）

  - **指令级并行（MMA指令）**: `mma.sync`或`wgmma`指令直接调用Tensor Core

  - **共享内存（SMEM）数据复用**

  - **TMA（Tensor Memory Accelerator）**: Hopper

  - **合并访问与对齐**: 全局内存内存事务合并访问

  - **寄存器文件（RF）优化**：调整寄存器

  - **Ping-Pong Kernel / Double Buffer**

  - **持久化内核（Persistent Kernels）**： 在代码的什么部分体现的？

  - **多精度支持**

  - **结构化稀疏GEMM**

  - **BlockSwizzling**: 相邻SM处理的线程块在数据空间上也相邻，提升L2缓存命中率

    **A是row-major，B是col-major，C是row-major**:

    ![img](https://pic2.zhimg.com/v2-b0cce73bcdf7bccf7ded53f2efc88123_1440w.jpg)

    - **调整网格形状**：将原始的`(M, N)`网格划分为更窄的逻辑网格，例如将`(M, N)`变为`(M * tile, N / tile)`，其中`tile`是2的幂次方（如1、2、4等）
    - **偏移量计算**：通过位运算将逻辑线程块索引映射回原始数据坐标，确保数据访问连续性

  - **swizzle减少Bank冲突**: shared memory 的列偏移可以由**行偏移的若干位**与 **Global Memory 中的列偏移** 异或得到

    八个块是一个最小的线程组内的八个块，一个块内，存 8 个 16位个数，四个 32位的bank。8个一组的组内不会有bank冲突，组间会有冲突。

  ![img](https://pic2.zhimg.com/v2-52ad0b8c4bf9ed091da002f68eba3795_1440w.jpg)
  
  - **融合算子**: 融合后处理
  - **动态集群调度**： 自动选择最优计算单元
  - 



### TODO

- https://deep-learning.feishu.cn/wiki/ObQjwqMcCiASWJk3OkncS5q9nZc?fromScene=spaceOverview
- https://deep-learning.feishu.cn/wiki/PnYmw4KcdiPBphkJBFdcqdDxnke?fromScene=spaceOverview
- https://zhuanlan.zhihu.com/p/678602674
- https://developer.nvidia.com/blog/optimizing-compute-shaders-for-l2-locality-using-thread-group-id-swizzling/
- CUTLASS Tutorial: Persistent Kernels and Stream-K: https://research.colfax-intl.com/cutlass-tutorial-persistent-kernels-and-stream-k/
- CUDA-X: https://developer.nvidia.com/gpu-accelerated-libraries
- magnum-io: 
  - https://developer.nvidia.com/magnum-io 
  - https://www.nvidia.com/en-us/data-center/magnum-io/
- GEMM的实现？如何分块是最高效的？
- 调用cublas的时候，是按照什么shape下发执行的？启发式的算法  与 cutlass有什么区别？ cutlass的tile信息可以指定
- cub库都支持哪些操作？
- 矩阵乘的cuda 实现与优化
- 调用cutlass的时候ThreadBlockTile、WarpTile、ThreadTile 的含义和影响
- Triton 服务中的并行实例数以及并行粒度。
- 分布式
- ~Persistent Kernels
- cutlass: StreamK、Data Parallel (DP)、多GEMM合并、Persistent Kernel
- ~针对于N = 250K的场景，探索cutlass的使用方式
- ~CTA
- ~Cooperative Groups
  - 可以是跨block的嘛？ 可以，但是有要求
- multi_warp_cg_reduce 的 reduce
- ~unified virtual addressing (UVA) vs. unified memory: perceived difference 的例子
- 通信的kernel 能不能跟计算的kernel融合？
- CPU的kernel fusion对于性能提升体现在哪些方面？
  - **减少指令开销**:  指令数量压缩 / 代码密度提升
  - **优化内存访问**: 减少中间数据存储 / 降低内存带宽压力
  - **提高并行效率**: 允许更多操作在流水线中并行执行 / 融合条件分支操作（如比较+跳转）可降低分支预测错误率，避免流水线清空
  - **降低功耗**: 减少指令执行次数和内存访问可显著降低CPU动态功耗 / 融合后减少的电路激活区域（如解码单元）可降低漏电功耗
  - 
- **MPI**是什么？
- GPUDirect
- tensorrt 多实例的原理是什么？
- NVLink读数据，是否支持读一个文件的分片？
- 同时是否可以在不同的SM上Launch执行不同的kernel?
- Nsight 分析自己写的kernel
- DeepGemm 迁移Ada架构
- Interview 仓库中的kernel过一遍
- prefix_copy 有时间自己做一下
- cutlass自带的算子融合的例子 https://github.com/NVIDIA/cutlass/tree/main/examples/13_two_tensor_op_fusion
- cutlass原生支持的分布式 GEMM: https://blog.shi-labs.com/distributed-gemm-88be6a481e2b
- cutlass: https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/efficient_gemm.md
- 不同形状的Layout（行优先/列优先/交错存储）: 交错存储代表什么意思？
- 余弦相似度有没有现成的融合好的算子？
- 总结这几个GTC:
  - S72576 - Getting Started with Multi-GPU Scaling: Distributed Libraries This talk
  - S72579 - Going Deeper with Multi-GPU Scaling: Task-based Runtimes 9AM today by Manolis Papadakis
  - S72578 - Advanced Multi-GPU Scaling: Communication Libraries 10AM today by Jiri Kraus
- MPI 和 GPUDirect 没有在 Magnum IO 里面
  - https://developer.nvidia.com/gpudirect
  - https://developer.nvidia.com/mpi-solutions-gpus
- CUDA优化技巧总结：https://zhuanlan.zhihu.com/p/271740706
- XLA中HLO的优化层次：HLO moudle computation
- perChanel量化的细节？

不重要TODO:

- https://zhuanlan.zhihu.com/p/545365047

```

```

Cutlass: https://github.com/NVIDIA/cutlass/wiki/Resources

Bank冲突（当每个线程存储（或加载）超过 4 个字节时（相当于每个 Warp 超过 128 个字节），GPU 不会发起单个事务。最大事务大小为 128 字节）：

https://forums.developer.nvidia.com/t/how-to-understand-the-bank-conflict-of-shared-mem/260900?utm_source=chatgpt.com

Tensorflow/Tensorflow Serving:
https://arxiv.org/pdf/1712.06139
https://arxiv.org/pdf/1605.08695

橡树岭CUDA培训：https://www.olcf.ornl.gov/cuda-training-series/

设备上修改张量图：

https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#encoding-a-tensor-map-on-device

PCIE: https://www.cnblogs.com/yi-mu-xi/p/10939735.html
PCIE专题：
https://blog.chinaaet.com/justlxy/p/5100053251

https://blog.chinaaet.com/justlxy/p/5100053328
PTX优化建议：https://developer.nvidia.com/blog/advanced-nvidia-cuda-kernel-optimization-techniques-handwritten-ptx/
