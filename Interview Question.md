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

- GPU分支预测的原理：GPU会将不同分支路径分配给多个ALU同时执行（如将`if-else`的两个分支分发给不同线程），最终仅保留有效路径的结果，丢弃无效路径的计算
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

- **`barrier`** 使用的是一种称为 “barrier resources” 的硬件资源（例如专门的同步计数器或原子机制），这种资源数量有限 [NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/differences-and-compatibility-between-mbarrier-and-barrier-in-ptx/314443?utm_source=chatgpt.com)[Stack Overflow](https://stackoverflow.com/questions/53662484/cuda-how-to-use-barrier-sync?utm_source=chatgpt.com)。不能分离“到达 arrive”与“等待 wait”两步操作。

  **`mbarrier`** 则在 **共享内存（shared memory）** 中维护其状态，因此不依赖有限的硬件资源 [NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/differences-and-compatibility-between-mbarrier-and-barrier-in-ptx/314443?utm_source=chatgpt.com)[reviews.llvm.org](https://reviews.llvm.org/D154090?utm_source=chatgpt.com)。

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
- Zero

![企业微信截图_f715778e-e18a-407c-ba52-395f88ed2fc4](file:///Users/m/Library/Containers/com.tencent.WeWorkMac/Data/Documents/Profiles/3675B093401F95C2759D981FC093A0ED/Caches/Images/2025-07/43bdfe899adc0d3ba3229c2575bb6cd4_HD/%E4%BC%81%E4%B8%9A%E5%BE%AE%E4%BF%A1%E6%88%AA%E5%9B%BE_f715778e-e18a-407c-ba52-395f88ed2fc4.png?lastModify=1753413830)

- **DeepGemm**:  
  - **Promotion / 两级累加技术**: 
    - **第一级（Tensor Core 计算）**: FP8×FP8→FP16/BF16
    - **第二级（CUDA Core 累加）**: 通过 CUDA Core 的 `FFMA`（Fused Multiply-Add）指令将 FP16/BF16 部分和累加到 FP32 寄存器中。
- **余弦相似度流水线**：
  - 性能分析需要关注的指标：缓存命中率；各级带宽；bank冲突；算力；
  - tiling策略：
    - m16n64k128（n>>m）: A（16x128x2B） B（128x64x2B）
    - N拆分到不同的warp上
    - cp.async: 32个线程一次总共读4x128x2B，for循环可以读很多行
  - perfetch: 
    - 从全局内存加载首个分块数据到共享内存
    - 从共享内存加载到寄存器
    - 计算小矩阵A的欧几德里范数，结果存在寄存器里面
    - A矩阵也可以直接存在寄存器里面，全部都放的下；
  - for 循环（对大矩阵不同grid分块后的循环）：
    - 从全局内存取下一个分块到共享内存（异步）
    - for循环（K维度[128]*M[64]维度的分块）：
      - 从共享内存加载B的K下一分块到寄存器
      - 计算当前分块(m16n8k16)，K维度的reduce在寄存器上完成（B常驻在寄存器中）
      - 计算当前大分块的平方和
      - 计算B矩阵的欧几德里范数
    - 计算最终结果
    - 结果写回全局内存
    - wait 下一个分块完成

**自我介绍**：

毕业之后加入阿里巴巴，负责模型部署系统的开发，(之后组织架构调整)之后阿里巴巴组建之江实验室，与燧原合作，开发燧原芯片的软件栈，24年从杭州换工作地点到上海加入小红书，负责算子开发和相关的优化工作，包括华为、燧原计算卡的部署上线，GPU/CPU计算相关的优化，取得了比较显著的性能收益，并在小红书工作期间发表两篇论文。



### Attention

$$
Q &= X W^Q \quad &\text{(Query)} \\
K &= X W^K \quad &\text{(Key)} \\
V &= X W^V \quad &\text{(Value)} \\
$$

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$



- Shape:

  输入矩阵 `X`：形状为 `(n,d_model)`，其中 `n`是输入序列的token数量，`d_model`是模型的隐藏层维度

  `W^Q`、`W^K`、`W^V`的形状均为 `(d_model, d_k)`（多头，`d_k = d_v = d_model / num_heads`）

  每个头的输出形状为 `(n,d_k)`，拼接后为 `(n,h⋅dk) = (n,dmodel)`，再通过 `W^O`（形状` (d_model,d_model)`）线性变换回原维度 `(n,d_model)`

|       **步骤**       |          **Q/K/V Shape**          |         **说明**         |
| :------------------: | :-------------------------------: | :----------------------: |
|       输入`X`        |    `(batch_size, n, d_model)`     |       初始输入序列       |
|  线性变换后`Q K V`   |    `(batch_size, n, d_model)`     |     独立的Q/K/V投影      |
|  多头拆分后`Q K V`   | `(batch_size, num_heads, n, d_k)` |   拆分为多个头并行计算   |
|  注意力分数（QK^T）  |  `(batch_size, num_heads, n, n)`  | 点积与缩放后的相似度矩阵 |
| 加权输出（Attn × V） | `(batch_size, num_heads, n, d_k)` |  每个头的注意力加权结果  |
|    拼接与最终输出    |    `(batch_size, n, d_model)`     |  恢复原始维度并线性变换  |
|       FFN输出        |    `(batch_size, n, d_model)`     |                          |

- FA优化点：
  - **FlashAttention-1**
    - **分块计算（Tiling）**：将Q、K、V矩阵划分为小块（Tile），在GPU的SRAM中局部计算注意力分数，避免存储完整的N×N注意力矩阵，显存占用从O(N²)降至O(N)
    - **算子融合（Kernel Fusion）**：将Softmax、Masking、Dropout等操作融合到单个CUDA核函数中，减少显存读写次数
    - **反向传播重计算（Recomputation）**：前向传播时不保存中间矩阵（如QKᵀ），反向传播时重新计算，牺牲计算时间换取显存节省
  - **FlashAttention-2**
    - **减少非矩阵乘法操作**：延迟Softmax的分母计算，减少除法次数，提升Tensor Core利用率
    - **循环顺序调整**：将外层循环改为Q，内层循环为KV，减少共享内存访问冲突，提升并行度
    - **细粒度并行化**：在序列长度维度并行化，支持更长的序列（如8K-32K），H100 GPU的FP16计算利用率达73%
    - **因果掩码优化**：跳过无效块（如上三角区域），加速自回归生成任务
  - **FlashAttention-3**
    - **异步流水线**：利用Hopper GPU的Tensor Memory Accelerator（TMA）和Warp专业化，分离GEMM与Softmax到不同Warp，实现计算与内存访问重叠
    - **低精度支持（FP8）**：采用FP8（e4m3格式）计算，结合块量化技术（QuIP），FP8计算速度达1.2 PFLOPs/s，误差降低60%
  
- FlashAttention 中的 Softmax: 

  - **计算当前块的局部统计量**：

    - `m_local = max(x_i)`// 当前块内的最大值
    - `f_local = exp(x_i - m_local)`// 当前块的指数值（减去局部最大值以保持数值稳定）
    - `l_local = sum(f_local)`// 当前块的指数和

  - **迭代更新全局统计量**:

    - **新的全局最大值**：`m_global_new = max(old_m_global, m_local)`

    - **新的全局指数和**：`l_global_new = exp(old_m_global - m_global_new) * old_l_global + exp(m_local - m_global_new) * l_local`

      *这里的 `old_m_global`和 `old_l_global`是处理之前所有块后累积的全局最大值和指数和。通过乘以一个修正因子（`exp(old_m_global - m_global_new)`和 `exp(m_local - m_global_new)`），将之前基于旧最大值的计算结果“校正”到新的全局最大值尺度上。*

    - **更新输出**
      - 对于当前块：`softmax_current_block = f_local * exp(m_local - m_global_new) / l_global_new`
      - 对于之前已经计算过的块：它们的输出也会在全局最大值更新后被重新缩放。FlashAttention 通过持续更新一个输出块（O_i）来避免存储中间结果。

- KVCache优化:

  - 缓存压缩与量化技术
    - **KV张量压缩**：通过降低KV向量的维度或精度来减少存储需求。MLA模型提出仅保存压缩后的潜在向量和共享键向量，大幅削减每token的缓存参数数量。
    - **低精度量化**：将KV Cache从FP16降至INT8或INT4
    - **分层存储**：构建GPU显存与CPU内存的二级缓存体系
  - 动态内存管理策略
    - **PagedAttention机制**: 将KV Cache分割为固定大小的Page（通常1MB-4MB），通过页表管理逻辑连续而物理分散的内存块。这种方法支持按需分配、灵活处理变长序列，并实现零拷贝的KV缓存共享

- KV Cache 如何保证精度：
  - **动态与非均匀量化策略**：
    - **动态范围调整**：根据输入数据的实际分布动态调整量化参数（如缩放因子和零点），避免静态量化因固定范围导致的精度损失。
    - **非均匀量化（NUQ）**： 针对KV Cache中数值分布的非均匀特性（如离群值），采用非均匀量化网格（如k-means聚类生成的量化点）来更精确地匹配数据分布
  - **混合精度与关键部分保护**：
    - **分层/分头差异化量化**：对模型中不同层或注意力头分配不同的量化精度。
    - **注意力关键token保护**：模型对某些token（如首token或标点符号）的KV Cache误差更敏感。

https://zhuanlan.zhihu.com/p/11886909512

### WGMMA

- 异步
- 大矩阵
- 从共享内存读取
- 多个warp协作

### CUTLASS

- 优化方法：

  - **ThreadBlock-Warp-Thread三级分块**: BlockTile（线程块级）、WarpTile（warp级）和ThreadTile（线程级）

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

    ```
    T array[][NX]给定共享内存中的数组，我们定义NX × sizeof(T) == SWIZZLE_SIZE。允许的值为SWIZZLE_SIZE大于或等于 32 的 2 的幂，例如 32、64、128、256、…… 等。
    
    [y][x]给定中的索引T array[][NX]，我们可以按如下方式计算混合索引x_swz：
    
    TC计算 -byte 段内 -byte 块的索引SWIZZLE_SIZE：
    i_chunk = (y × NX + x) × sizeof(T) / sizeof(TC)
    y_chunk = i / (SWIZZLE_SIZE / sizeof(TC))
    x_chunk = i % (SWIZZLE_SIZE / sizeof(TC))
    
    TC使用XOR 运算计算字节块的混合索引：
    x_chunk_swz = y_chunk ^ x_chunk
    
    计算混合索引：
    x_swz = x_chunk_swz × sizeof(TC) / sizeof(T) % NX + x % (sizeof(TC) / sizeof(T))
    ```
    
    八个块是一个最小的线程组内的八个块，一个块内，存 8 个 16位个数，四个 32位的bank。8个一组的组内不会有bank冲突，组间会有冲突。
  
  ![img](https://pic2.zhimg.com/v2-52ad0b8c4bf9ed091da002f68eba3795_1440w.jpg)
  
  - **融合算子**: 融合后处理
  - **动态集群调度**： 自动选择最优计算单元

### 编译器优化

- 编译器后端优化策略：Reorder(交换)、Split(拆分)、Fuse(融合)、Tile(平铺)、Vector(向量化)、展开(Unrolling)、并行(Paralle lizing)
  TVM 中的Reorder(交换)、Split(拆分)、Fuse(融合)、Tile(平铺)、Vector(向量化)、展开(Unrolling)、绑定(bindin g)

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

CUDA博客：https://mp.weixin.qq.com/mp/appmsgalbum?__biz=Mzk2NDA4NjU0Nw==&action=getalbum&album_id=3922953336307122177&from_itemidx=1&from_msgid=2247487257&#wechat_redirect

AXI:

-  https://developer.arm.com/documentation/102202/0300/AXI-protocol-overview
- https://adaptivesupport.amd.com/s/article/1053914?language=en_US
- https://verificationprotocols.blogspot.com/2017/03/axi-protocol.html
- https://blog.csdn.net/weixin_33691817/article/details/93521692

Outstanding：

- https://www.intel.com/content/www/us/en/docs/programmable/788295/23-3/maximum-number-of-outstanding-transactions.html
- https://developer.arm.com/documentation/102202/0300/Transfer-behavior-and-transaction-ordering
- https://docs.amd.com/r/en-US/am011-versal-acap-trm/
- https://docs.amd.com/r/en-US/am011-versal-acap-trm/Outstanding-Transactions
- https://docs.amd.com/r/en-US/pg313-network-on-chip/Latency

问题：

- *如果K维度是24, 如何解决bank 冲突？ 取余？

- GEMM/FA的优化方法

- L2 缓存优化（除了更窄的tilling）还有什么优化策略？

- PTX barrier 以及其他同步方式的异同？

- *.ca .cg缓存策略的使用？

- 指令并行

- swizzle 不是 2次幂如何处理？

- *伪相关 / bypass(旁路) / *IPC / scheduler

  perfill decode阶段需要关注的点有什么？


  量化的数据格式？ 底数 指数 以及表示的范围

  溢出问题如何排查？

  Zero策略的阶段和图片

  Megatron、DeepSpeed的特性

  *A100 x3 buffer性能最好，原因除了内存带宽大还有什么？减少阻塞 等待

  *cp.async 支持加载的数据数量（4B 8B 16B）？为什么会有这些？内存对齐/单次加载多个数据

  *cp.async 对应的等待指令

  = ldmatrix 为什么支持转置？ 与直接使用ld.shared有什么区别？主要原因是转置

  = 矩阵乘的内积或是外积，mma只支持row.col这种的计算

  *warp stall

  *理论占用率怎么算？

  *每个线程使用的寄存器限制：硬件-512 编译器限制-256

  *模板元编程

  ?模板修改成员变量

  *可变模板参数

  *enableif

  *多重引用、引用折叠: T&& &&`→ `T&&

  *lambda表达式的原理 如何修改成员变量？使用mutable修改

  *volatile 在C++上的意义？有没有防止指令重排的功能？没有\同一volatile变量操作不会移除、合并或重新排序，其他的不报证

  *mutable

  ?ldmatrix转置的目的

  *mma 是否支持转置: 只支持row.col

  *cp.async的填充

  *cp.async 的wait 等待的数量？

  *__threadfence() : 同一网格 (grid)、  非阻塞、仅强制完成内存操作的可见性；

  *__syncthreads() ：仅同步同一线程块的线程

  *__threadfence_block(): 对同一块内线程的可见性

  *__threadfence_system(): 扩展可见性到主机

  *bitfield
  *make_shared() 计数在一起

  *智能指针三种

  *完美转发/std::forward

  *wmma指令与mma指令的差别？bank 冲突

  *list vector map 异同

  *编译流程

  *stream用法，多stream是否需要等待默认stream: 要等待

  *cudaMalloc 同步、异步：同步

  *统一地址空间

  *循环展开的性能优缺点：两点

  *MMIO

  *IPC

  *内存布局

  *指令乱序是在哪个阶段实现的？发射（Issue） 执行（Execute）

  *XLA 在GPU上是如何做的JIT? 正常做

  GEMM对于不同形状的矩阵如何优化？

  block 的fp8量化

  阿里巴巴 爱橙面试题：

  Per block 量化

  FP8 量化、 FP8 Per block、int8量化的区别

  Per block 

分部分的softmax

**HLO到PTX的过程**

- **转换到 `LMHLO Dialect`**： **`LMHLO`** 是 **Late MHLO** 的缩写，可以理解为**具有内存缓冲区的 HLO**

- **转换到 `GPU Dialect`**: 用于表示 **GPU 通用操作** 的抽象，如 `gpu.launch`（启动 GPU 内核）、`gpu.memcpy`（内存拷贝）、`gpu.shuffle`（线程束内数据交换）

- **转换到 `LLVMIR Dialect`**：**`LLVMIR Dialect`** 允许在 MLIR 中**直接表示 LLVM 的指令和类型**

- **生成 PTX 代码**：MLIR 就可以利用 **LLVM进一步优化并编译生成 PTX 代码**

  - **从 LLVMIR Dialect 到标准 LLVM IR**
  - **LLVM 优化与代码生成** -> **NVPTX**
  - **NVPTX 后端生成 PTX**

  





