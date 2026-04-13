分核方式是在 dim=0 均匀分片每个 AI Core 处理一段连续的 K 元素。 
分块大小 block_size = 256 block_num = ceil(K / 256) 
第 i 个 block 处理区间 
start = i * 256 
end = start + 256 
实际长度 = end – start

每个 block 固定使用：
ub_cumsum：1024 × int64 = 8192 字节
ub_table：256 × int64 = 2048 字节
ub_row：256 × int64 = 2048 字节
ub_out：256 × int64 = 2048 字节
总使用 UB
8192 + 2048 + 2048 + 2048 = 14336 字节 ≈ 14KB


5.2	分核设计
分核方式是在 dim=0 均匀分片每个 AI Core 处理一段连续的 K 元素。 
分块大小 block_size = 256 block_num = ceil(K / 256) 
第 i 个 block 处理区间 
start = i * 256 
end = start + 256 
实际长度 = end – start
 4. 核执行流程（dim=0 块）
1.	CopyIn (GM → UB) 
读取当前块：table_indices[start:end] 读取当前块：row_indices[start:end] 全量读取 cumsum
2.	Compute 运行 
for (int i = 0; i < block_len; i++) {
 table_id = ub_table[i];
 row_id = ub_row[i]; 
ub_out[i] = ub_cumsum[table_id] + row_id; 
}
3.	CopyOut (UB → GM) 写回 linear_cache_indices[start:end]
4.	先写一个scale实现
struct LinearizeCacheIndicesTilingData {
    int64_t total_length;
    int64_t block_size;
    int64_t num_tables_1;
};
```python
extern "C"
__global__ void LinearizeCacheIndicesScalarKernel(
    const __gm__ int64_t* cache_hash_size_cumsum,  // [N+1]
    const __gm__ int64_t* update_table_indices,    // [K]
    const __gm__ int64_t* update_row_indices,       // [K]
    __gm__ int64_t* linear_cache_indices,           // [K]
    const LinearizeCacheIndicesTilingData* tiling
) {
    // 解析分块信息
    int64_t total_length = tiling->total_length;
    int64_t block_size = tiling->block_size;
    int64_t num_tables_1 = tiling->num_tables_1;

    // 计算全局线程索引
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 边界判断
    if (idx >= total_length) return;

    int64_t table_id = update_table_indices[idx];
    int64_t row_id = update_row_indices[idx];
   
    int64_t offset = cache_hash_size_cumsum[table_id]; 
    int64_t result = offset + row_id;

    linear_cache_indices[idx] = result;
}
```
ub
```python
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    OPS_LOG_E_IF_NULL("context", context, return GRAPH_FAILED);

    // 获取输入shape
    auto tableShape = context->GetInputShape(1);
    auto cumsumShape = context->GetInputShape(0);
    OPS_LOG_E_IF_NULL("tableShape", tableShape, return GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("cumsumShape", cumsumShape, return GRAPH_FAILED);

    int64_t totalLength = tableShape->GetOriginShape().GetShapeSize();
    int64_t numTablesPlus1 = cumsumShape->GetOriginShape().GetShapeSize();

    const auto ascendPlatform = platform_ascend::PlatformAscendC(context->GetPlatformInfo());
    ascendPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint32_t ubSize = 0;
    auto ret = ascendPlatform.GetCoreMemSize(CoreMemType::UB, ubSize);
    OPS_LOG_E_IF(ret != 0, context, return GRAPH_FAILED, "get ub size failed");

    // 预留安全空间，目标使用≈240KB
    uint32_t usableUB = ubSize - 16 * 1024;
    uint32_t perBufferUB = usableUB / BUFFER_NUM;

    // 计算最大block size（UB占满）
    int64_t elementsPerBlock = perBufferUB / DATA_TYPE_BYTES;
    elementsPerBlock = (elementsPerBlock / MAX_THREADS_PER_BLOCK) * MAX_THREADS_PER_BLOCK;

    // 最小保护
    if (elementsPerBlock < MAX_THREADS_PER_BLOCK) {
        elementsPerBlock = MAX_THREADS_PER_BLOCK;
    }

    // 计算分块
    int64_t totalBlocks = (totalLength + elementsPerBlock - 1) / elementsPerBlock;
    size_t maxCores = ascendPlatform.GetCoreNumAiv();
    size_t coreNum = std::min(totalBlocks, maxCores);

    // workspace
    size_t* workspaceSize = context->GetWorkspaceSizes(1);
    workspaceSize[0] = ascendPlatform.GetLibApiWorkSpaceSize();

    // 填充tiling
    LinearizeTilingData tiling{};
    tiling.totalLength = totalLength;
    tiling.elementsPerBlock = elementsPerBlock;
    tiling.numTablesPlus1 = numTablesPlus1;

    // 下发配置
    context->SetBlockDim(coreNum);
    context->SetLocalMemorySize(ubSize);

    // 保存tiling
    tiling.SaveToBuffer(
        context->GetRawTilingData()->GetData(),
        context->GetRawTilingData()->GetCapacity()
    );
    context->GetRawTilingData()->SetDataSize(sizeof(tiling));

    return GRAPH_SUCCESS;
}

REGISTER_TILING_FUNC(LinearizeCacheIndicesFromRowIdx, TilingFunc);
```
# `linearize_cache_indices_from_row_idx` 算子分析文档

> 源码版本：FBGEMM 1.5.0
> 涉及文件：
> - `fbgemm_gpu/src/split_embeddings_cache/linearize_cache_indices.cu`
> - `fbgemm_gpu/src/split_embeddings_cache/linearize_cache_indices.cpp`
> - `fbgemm_gpu/src/split_embeddings_cache/split_embeddings_cache_ops.cpp`
> - `fbgemm_gpu/include/fbgemm_gpu/split_embeddings_cache_cuda.cuh`
> - `fbgemm_gpu/test/tbe/cache/linearize_cache_indices_test.py`

---

## 一、功能概述

该算子将**表局部行索引（table-local row indices）** 线性化为**全局缓存索引（global linear cache indices）**，用于 TBE（Table Batched Embedding）推理系统的 UVM/LRU/LFU 嵌入缓存管理。

与 `linearize_cache_indices` 的区别：
- `linearize_cache_indices`：输入为 `(indices, offsets)` 稀疏格式，需二分查找确定表归属，适用于**前向推理查找**。
- `linearize_cache_indices_from_row_idx`：输入为 `(update_table_indices, update_row_indices)` 行索引对格式，每条记录已明确标注所属表，直接 O(1) 查表，适用于 **in-place 嵌入更新**。

---

## 二、接口说明

### 2.1 接口签名

**Python 调用接口（TorchScript Op）**

```python
torch.ops.fbgemm.linearize_cache_indices_from_row_idx(
    cache_hash_size_cumsum: Tensor,   # [T+1], int64
    update_table_indices: Tensor,     # [N],   int32/int64
    update_row_indices: Tensor,       # [N],   int32/int64
) -> Tensor                           # [N],   与 update_row_indices 同 dtype
```

**Torch Op 注册 Schema**（`split_embeddings_cache_ops.cpp` 第 18 行）

```
linearize_cache_indices_from_row_idx(
    Tensor cache_hash_size_cumsum,
    Tensor update_table_indices,
    Tensor update_row_indices
) -> Tensor
```

**C++ CUDA 函数签名**（`split_embeddings_cache_cuda.cuh` 第 85 行）

```cpp
at::Tensor linearize_cache_indices_from_row_idx_cuda(
    at::Tensor cache_hash_size_cumsum,
    at::Tensor update_table_indices,
    at::Tensor update_row_indices);
```

---

### 2.2 入参说明

| 参数名 | 维度 | dtype | 设备 | 说明 |
|--------|------|-------|------|------|
| `cache_hash_size_cumsum` | 1D `[T+1]` | `int64` | CUDA | 所有嵌入表哈希容量的前缀累加和。`[t]` 为第 t 张表在全局缓存中的起始偏移；`-1` 表示该表未缓存；最后一个元素为总缓存大小（哨兵值） |
| `update_table_indices` | 1D `[N]` | `int32`/`int64` | CUDA | 每条更新记录所属的表编号，取值范围 `[0, T-1]` |
| `update_row_indices` | 1D `[N]` | `int32`/`int64` | CUDA | 每条更新记录在其所属表内的局部行号；负数表示该行已被剪枝（pruned） |

**入参约束**

| 约束项 | 说明 |
|--------|------|
| 设备一致性 | 三个张量必须在同一块 CUDA GPU 上 |
| 表数量 | `T = cache_hash_size_cumsum.size(0) - 1`，必须 `T > 0` |
| 长度一致 | `update_table_indices` 与 `update_row_indices` 长度相同，均为 `N` |
| dtype 一致 | `update_table_indices` 与 `update_row_indices` 的 dtype 必须相同 |
| 表号范围 | `update_table_indices[i]` 应在 `[0, T]` 内，防止越界访问 `cache_hash_size_cumsum` |

---

### 2.3 出参说明

| 参数名 | 维度 | dtype | 设备 | 说明 |
|--------|------|-------|------|------|
| `linear_cache_indices` | 1D `[N]` | 与 `update_row_indices` 相同 | CUDA | 每条更新记录在全局缓存地址空间中的一维绝对地址 |

**输出语义**

```
设 t   = update_table_indices[i]
设 off = cache_hash_size_cumsum[t]
设 max = cache_hash_size_cumsum[T]   ← 哨兵值

if off >= 0  AND  update_row_indices[i] >= 0:
    linear_cache_indices[i] = update_row_indices[i] + off   ← 正常映射
else:
    linear_cache_indices[i] = max                           ← 无效（未缓存/已剪枝）
```

**有效性判断矩阵**

| `cache_hash_size_cumsum[t]` | `update_row_indices[i]` | 输出 |
|-----------------------------|-------------------------|------|
| `>= 0`（表已缓存） | `>= 0`（行有效） | `row_idx + curr_offset`（正常） |
| `>= 0`（表已缓存） | `< 0`（行已剪枝） | `max_offset`（哨兵） |
| `< 0`（表未缓存） | `>= 0`（行有效） | `max_offset`（哨兵） |
| `< 0`（表未缓存） | `< 0`（行已剪枝） | `max_offset`（哨兵） |

---

## 三、算子原理

### 3.1 背景：全局缓存地址空间

TBE 系统中存在 T 张独立嵌入表，每张表行号从 0 开始独立编号（局部坐标）。GPU 缓存将所有表统一映射到一段连续的全局地址空间（全局坐标）。`cache_hash_size_cumsum` 记录了每张表在该全局空间中的起始偏移量。

**全局缓存地址空间分段布局（以 4 张表、每张容量 12 为例）**

```
cache_hash_size_cumsum = [ 0,  12,  24,  36,  48 ]
                           ↑    ↑    ↑    ↑    ↑
                          t=0  t=1  t=2  t=3  哨兵

全局缓存地址空间（连续）：
┌────────────────┬────────────────┬────────────────┬────────────────┐
│   table_0 段   │   table_1 段   │   table_2 段   │   table_3 段   │
│  行 0 ~ 11     │  行 0 ~ 11     │  行 0 ~ 11     │  行 0 ~ 11     │
│  地址 0 ~ 11   │  地址 12 ~ 23  │  地址 24 ~ 35  │  地址 36 ~ 47  │
└────────────────┴────────────────┴────────────────┴────────────────┘
  ↑ cumsum[0]=0    ↑ cumsum[1]=12   ↑ cumsum[2]=24   ↑ cumsum[3]=36
                                                       ↑ cumsum[4]=48（哨兵）
```

**含未缓存表的情况（`cumsum[2] = -1`，table_2 不在缓存中）**

```
cache_hash_size_cumsum = [ 0,  12,  -1,  24,  36 ]

全局缓存地址空间（table_2 被跳过）：
┌────────────────┬────────────────┬────────────────┐
│   table_0 段   │   table_1 段   │   table_3 段   │
│  地址 0 ~ 11   │  地址 12 ~ 23  │  地址 24 ~ 35  │
└────────────────┴────────────────┴────────────────┘
  cumsum[0]=0      cumsum[1]=12     cumsum[3]=24

  table_2: cumsum[2]=-1 → 未缓存，输出哨兵值 36（= cumsum[4]）
```

---

### 3.2 核心原理：坐标系变换（一次加法）

算子本质是一次**坐标系变换**：将 `(表号 t, 局部行号 r)` 的二元组映射为全局一维地址。

```
                 局部坐标系（每张表独立）
                 ┌────────────────────────────┐
  table_0 局部:  │ row 0 | row 1 | ... | row r│
                 └────────────────────────────┘
                   ↓ + cumsum[0] = 0
                 ┌────────────────────────────┐
  全局缓存空间:  │  0   |  1   | ... |   r   │ ← table_0 段
                 ├────────────────────────────┤
  table_1 局部:  │ row 0 | row 1 | ... | row r│
                   ↓ + cumsum[1] = 12
                 ├────────────────────────────┤
                 │  12  |  13  | ... |  12+r  │ ← table_1 段
                 ├────────────────────────────┤
                 │           ...              │
                 └────────────────────────────┘

  公式：global_addr = local_row + cumsum[table_id]
```

---

## 四、计算流程

### 4.1 总体流程图

```
              ┌──────────────────────────────────────┐
              │  输入三个 CUDA Tensor                  │
              │  cache_hash_size_cumsum [T+1]          │
              │  update_table_indices   [N]            │
              │  update_row_indices     [N]            │
              └──────────────────┬───────────────────┘
                                 │
              ┌──────────────────▼───────────────────┐
              │  Host 侧预处理                         │
              │  ① 校验同一 CUDA 设备                  │
              │  ② T = cumsum.size(0)-1，断言 T > 0   │
              │  ③ N = update_row_indices.numel()      │
              │  ④ N==0 ? 返回空 Tensor : 继续         │
              │  ⑤ 分配输出 Tensor（empty_like）        │
              │  ⑥ 计算 grid = ⌈N / kMaxThreads⌉      │
              └──────────────────┬───────────────────┘
                                 │
              ┌──────────────────▼───────────────────┐
              │  启动 CUDA Kernel                      │
              │  grid  = ⌈N / kMaxThreads⌉            │
              │  block = kMaxThreads                   │
              │  N 个线程完全并行，每线程处理 1 条记录   │
              └──────────────────┬───────────────────┘
                                 │
              ┌──────────────────▼───────────────────┐
              │  CUDA Kernel（每线程独立执行）           │
              │                                        │
              │  index = blockIdx.x*blockDim.x         │
              │          + threadIdx.x                 │
              │                                        │
              │  if index >= N: return（越界退出）      │
              │                                        │
              │  t   = update_table_indices[index]     │
              │  max = __ldg(&cumsum[T])               │
              │  off = __ldg(&cumsum[t])               │
              │                                        │
              │  if off>=0 AND row_idx>=0:             │
              │      output[index] = row_idx + off     │
              │  else:                                 │
              │      output[index] = max               │
              └──────────────────┬───────────────────┘
                                 │
              ┌──────────────────▼───────────────────┐
              │  返回 linear_cache_indices [N]         │
              │  传入 lxu_cache_lookup 进行缓存查询     │
              └──────────────────────────────────────┘
```

---

### 4.2 CUDA Kernel 单线程执行逻辑（伪代码）

```
// linearize_cache_indices.cu 第 128~153 行
kernel linearize_cache_indices_from_row_idx_kernel(
    cache_hash_size_cumsum,   // int64[T+1]，只读
    update_table_indices,     // index_t[N]，只读
    update_row_indices,       // index_t[N]，只读
    linear_cache_indices      // index_t[N]，写出
):
    index = blockIdx.x * blockDim.x + threadIdx.x

    // 边界保护
    if index >= update_row_indices.size(0):
        return

    // 读取该条记录所属的表号
    t = update_table_indices[index]

    // 用 __ldg 经 L1 只读缓存读取偏移（对所有线程广播）
    max_offset  = __ldg(&cache_hash_size_cumsum[T])   // 哨兵值
    curr_offset = __ldg(&cache_hash_size_cumsum[t])   // 表起始偏移

    // 有效性判断 + 输出
    if curr_offset >= 0 AND update_row_indices[index] >= 0:
        linear_cache_indices[index] = update_row_indices[index] + curr_offset
    else:
        linear_cache_indices[index] = max_offset
```

---

### 4.3 端到端数值示例

**场景：4 张嵌入表，table_2 未缓存**

```
================== 输入 ==================

cache_hash_size_cumsum = [  0,  12,  -1,  24,  36 ]
                            t=0  t=1  t=2  t=3  哨兵(max=36)
                         (缓存)(缓存)(不缓存)(缓存)

update_table_indices   = [ 0,  0,  0,  0,  1,  1,  1,  1,  2,  2,  2,  2,  3,  3,  3,  3 ]
update_row_indices     = [10,  2,  3,  7,  1,  4,  5,  9,  2,  7,  6,  8,  5,  1,  0,  4 ]
                          idx: 0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15

================== CUDA Kernel 并行计算（16 个线程同时运行）==================

 idx │ table │ curr_off │ row_idx │ row>=0? │ off>=0? │    输出
─────┼───────┼──────────┼─────────┼─────────┼─────────┼──────────────
  0  │   0   │    0     │   10   │    ✅    │    ✅    │  10 + 0  = 10
  1  │   0   │    0     │    2   │    ✅    │    ✅    │   2 + 0  =  2
  2  │   0   │    0     │    3   │    ✅    │    ✅    │   3 + 0  =  3
  3  │   0   │    0     │    7   │    ✅    │    ✅    │   7 + 0  =  7
  4  │   1   │   12     │    1   │    ✅    │    ✅    │   1 + 12 = 13
  5  │   1   │   12     │    4   │    ✅    │    ✅    │   4 + 12 = 16
  6  │   1   │   12     │    5   │    ✅    │    ✅    │   5 + 12 = 17
  7  │   1   │   12     │    9   │    ✅    │    ✅    │   9 + 12 = 21
  8  │   2   │   -1     │    2   │    ✅    │    ❌    │  max_offset = 36
  9  │   2   │   -1     │    7   │    ✅    │    ❌    │  max_offset = 36
 10  │   2   │   -1     │    6   │    ✅    │    ❌    │  max_offset = 36
 11  │   2   │   -1     │    8   │    ✅    │    ❌    │  max_offset = 36
 12  │   3   │   24     │    5   │    ✅    │    ✅    │   5 + 24 = 29
 13  │   3   │   24     │    1   │    ✅    │    ✅    │   1 + 24 = 25
 14  │   3   │   24     │    0   │    ✅    │    ✅    │   0 + 24 = 24
 15  │   3   │   24     │    4   │    ✅    │    ✅    │   4 + 24 = 28

================== 输出 ==================

linear_cache_indices = [10, 2, 3, 7, 13, 16, 17, 21, 36, 36, 36, 36, 29, 25, 24, 28]
                                                        ↑   ↑   ↑   ↑
                                              table_2 的 4 条记录全部输出哨兵值 36
```

---

### 4.4 负索引（剪枝）处理示例

```
示例：
  update_table_indices[i] = 1    → curr_offset = cumsum[1] = 12  （表已缓存）
  update_row_indices[i]   = -1   → 行索引为负                    （已被剪枝）

  判断：curr_offset(12) >= 0  AND  row_idx(-1) >= 0  →  FALSE
  输出：linear_cache_indices[i] = max_offset（哨兵值）

下游算子（lxu_cache_lookup）收到哨兵值后，将该条目视为缓存未命中，不进行缓存操作。
```

---

## 五、在 TBE 推理系统中的位置

```
                  ┌─────────────────────────────────┐
                  │   触发入口：emb_inplace_update    │
                  │   （split_table_batched_          │
                  │    embeddings_ops_inference.py    │
                  │    第 2010 行）                   │
                  └────────────────┬────────────────┘
                                   │
                                   │ 输入：
                                   │  update_table_indices [N]
                                   │  update_row_indices   [N]
                                   │
                  ┌────────────────▼────────────────┐
                  │  linearize_cache_indices_        │  ← 本算子
                  │    from_row_idx                  │
                  │                                  │
                  │  局部 (t, r) → 全局地址           │
                  └────────────────┬────────────────┘
                                   │ linear_cache_indices [N]
                  ┌────────────────▼────────────────┐
                  │  prefetch_32way / prefetch_1way  │
                  │  lxu_cache_lookup                │
                  │  查询每条记录是否命中缓存          │
                  └────────────────┬────────────────┘
                                   │ lxu_cache_locations [N]
                  ┌────────────────▼────────────────┐
                  │  emb_inplace_update（核心）       │
                  │  同时更新 UVM 权重 + Cache 权重   │
                  └─────────────────────────────────┘
```

---

## 六、关键设计要点

| 设计点 | 说明 |
|--------|------|
| **O(1) 直接查表** | 输入已携带表号，直接以 `update_table_indices[i]` 为下标访问 `cumsum`，无需二分查找，每线程单次内存访问 |
| **`__ldg` 只读缓存广播** | `cache_hash_size_cumsum` 是小数组（T+1 个元素），所有线程高频访问。`__ldg` 将其加载进 L1 只读缓存，同一 Warp 内的线程访问同一地址时直接广播，避免全局内存带宽瓶颈 |
| **完全并行，无数据依赖** | N 条记录之间无任何依赖，N 个 CUDA 线程完全独立执行，理论加速比线性于 N |
| **哨兵值统一无效语义** | `max_offset = cumsum[T]`（总缓存大小）作为唯一的无效标记，统一表达"未缓存"和"已剪枝"两种情况，简化下游算子逻辑 |
| **dtype 泛型支持** | 通过 `AT_DISPATCH_INDEX_TYPES` 同时支持 `int32` 和 `int64` 索引，`cache_hash_size_cumsum` 固定 `int64` 以支持超大规模嵌入表 |
| **CPU 实现为存根** | CPU 版本仅返回 `empty_like(update_row_indices)`，不做实际计算，该算子**仅在 CUDA 设备上有实际语义** |

---

## 七、计算流程伪代码

### 7.1 顶层调度入口（Python / TorchScript）

```python
# ============================================================
# 顶层调度入口
# 文件：split_table_batched_embeddings_ops_inference.py 第 2010 行
# ============================================================

function emb_inplace_update_entry(update_table_indices, update_row_indices):

    # 仅当缓存权重非空（即启用了 LRU/LFU 缓存）时才执行线性化
    if lxu_cache_weights.numel() > 0:

        # ① 调用本算子：将局部 (表号, 行号) 映射为全局缓存地址
        linear_cache_indices =
            torch.ops.fbgemm.linearize_cache_indices_from_row_idx(
                cache_hash_size_cumsum,    # [T+1], int64
                update_table_indices,      # [N],   int32/int64
                update_row_indices         # [N],   int32/int64
            )
        # 返回值 linear_cache_indices: [N], dtype 与 update_row_indices 相同

        # ② 将全局缓存地址送入预取/查找流程
        if cache_assoc in [32, 64]:
            prefetch_32way(linear_cache_indices)
        elif cache_assoc == 1:
            prefetch_1way(linear_cache_indices)

        lxu_cache_locations = lxu_cache_locations_list.pop()

    # ③ 使用缓存位置执行原地权重更新
    torch.ops.fbgemm.emb_inplace_update(
        ...,
        lxu_cache_locations = lxu_cache_locations
    )
```

---

### 7.2 Host 侧主函数伪代码（C++）

```
# ============================================================
# Host 侧主函数
# 文件：linearize_cache_indices.cu  第 157~191 行
# 对应实现：linearize_cache_indices_from_row_idx_cuda()
# ============================================================

function linearize_cache_indices_from_row_idx_cuda(
    cache_hash_size_cumsum,   # Tensor [T+1], int64,        CUDA
    update_table_indices,     # Tensor [N],   int32/int64,  CUDA
    update_row_indices        # Tensor [N],   int32/int64,  CUDA
) -> Tensor:

    # ── 步骤 1：设备校验 ──────────────────────────────────────
    # 三个张量必须位于同一块 CUDA GPU（宏展开为 TORCH_CHECK）
    ASSERT: cache_hash_size_cumsum, update_table_indices,
            update_row_indices 在同一 CUDA 设备上
    # 对应源码：TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(...)

    # ── 步骤 2：锁定 CUDA 设备上下文 ────────────────────────────
    CUDA_DEVICE_GUARD(cache_hash_size_cumsum)
    # 防止多 GPU 环境下的上下文切换错误

    # ── 步骤 3：计算表数量并做合法性校验 ─────────────────────────
    T = cache_hash_size_cumsum.size(0) - 1
    ASSERT T > 0                     # 至少有一张嵌入表
    # 对应源码：TORCH_CHECK(T > 0)

    # ── 步骤 4：分配输出 Tensor ───────────────────────────────
    # 形状、dtype、设备均与 update_row_indices 完全相同
    linear_cache_indices = at::empty_like(update_row_indices)
    # 此时内存已在 GPU 上分配，但内容未初始化

    # ── 步骤 5：空输入快速返回路径 ────────────────────────────
    N = update_row_indices.numel()
    if N == 0:
        return linear_cache_indices   # 直接返回空 Tensor，不启动 Kernel

    # ── 步骤 6：dtype 分发 ────────────────────────────────────
    # 根据 update_row_indices 的实际 dtype（int32 或 int64）
    # 在编译期实例化对应模板版本的 Kernel
    AT_DISPATCH_INDEX_TYPES(update_row_indices.scalar_type()):
        # index_t = int32  或  index_t = int64

        # ── 步骤 7：计算 CUDA Grid/Block 尺寸 ─────────────────
        block_size = kMaxThreads               # 通常为 512 或 1024
        grid_size  = ceil(N / block_size)      # ⌈N / kMaxThreads⌉

        # ── 步骤 8：启动 CUDA Kernel（异步） ──────────────────
        FBGEMM_LAUNCH_KERNEL(
            kernel  = linearize_cache_indices_from_row_idx_kernel<index_t>,
            grid    = grid_size,
            block   = block_size,
            stream  = cuda_current_stream(),
            args    = (
                cache_hash_size_cumsum,   # 只读，经 L1 只读缓存访问
                update_table_indices,     # 只读
                update_row_indices,       # 只读
                linear_cache_indices      # 写出
            )
        )
        # Kernel 在 GPU 上异步执行，CPU 立即返回

    # ── 步骤 9：返回结果 ──────────────────────────────────────
    return linear_cache_indices
    # 调用方使用时会隐式同步，或由 CUDA stream 保序
```

---

### 7.3 CUDA Kernel 伪代码（单线程视角）

```
# ============================================================
# CUDA Kernel（每个 GPU 线程独立执行此函数一次）
# 文件：linearize_cache_indices.cu  第 126~153 行
# 函数：linearize_cache_indices_from_row_idx_kernel<index_t>
#
# 执行配置：
#   gridDim.x  = ⌈N / kMaxThreads⌉
#   blockDim.x = kMaxThreads
#   总线程数   = gridDim.x × blockDim.x  ≥ N
# ============================================================

kernel linearize_cache_indices_from_row_idx_kernel(
    cache_hash_size_cumsum,   # int64[T+1]，全局内存只读区
    update_table_indices,     # index_t[N]，全局内存只读区
    update_row_indices,       # index_t[N]，全局内存只读区
    linear_cache_indices      # index_t[N]，全局内存写出区
):

    # ── 阶段 1：计算全局线程 ID（唯一对应一条输入记录）─────────
    index = blockIdx.x * blockDim.x + threadIdx.x
    #        块偏移                  块内线程偏移
    # 每个线程的 index 在 [0, gridDim.x × blockDim.x) 范围内唯一

    # ── 阶段 2：越界保护（尾部 block 的多余线程直接退出）────────
    if index >= update_row_indices.size(0):   # 即 index >= N
        return
    # 保证不越界访问 update_table_indices / update_row_indices

    # ── 阶段 3：读取本线程负责的记录所属表号 ─────────────────
    t = update_table_indices[index]
    # 全局内存随机读（各线程的 t 值通常不同，存在非合并访问）

    # ── 阶段 4：读取哨兵值和当前表偏移（走 L1 只读缓存）────────
    # __ldg：Load via Read-Only Data Cache（纹理缓存路径）
    # cache_hash_size_cumsum 是小数组（T+1 个 int64），
    # 高概率全部驻留在 L1 只读缓存中，多线程访问时直接广播
    max_offset  = __ldg( &cache_hash_size_cumsum[T] )
    #                      ↑ 最后一个元素 = 总缓存大小 = 哨兵值
    curr_offset = __ldg( &cache_hash_size_cumsum[t] )
    #                      ↑ 第 t 张表的全局起始偏移

    # ── 阶段 5：有效性判断（双重条件）────────────────────────
    #   条件 A：curr_offset >= 0  →  该表已缓存（非 -1）
    #   条件 B：update_row_indices[index] >= 0  →  该行未被剪枝
    if curr_offset >= 0 AND update_row_indices[index] >= 0:

        # ── 阶段 6a：正常路径 —— 局部行号加表偏移 ─────────────
        linear_cache_indices[index] =
            update_row_indices[index] + curr_offset
        #   ↑ 全局缓存地址 = 表内局部行号 + 该表在全局空间的起始偏移

    else:

        # ── 阶段 6b：无效路径 —— 写入哨兵值 ───────────────────
        # 触发条件（满足其一即可）：
        #   - curr_offset < 0：该表未被缓存（整表不在 GPU 缓存中）
        #   - update_row_indices[index] < 0：该行已被剪枝（pruning）
        linear_cache_indices[index] = max_offset
        # 哨兵值 = cache_hash_size_cumsum[T] = 总缓存大小
        # 下游算子（lxu_cache_lookup）见到此值时将视为"缓存未命中"
```

---

### 7.4 CPU 存根伪代码（仅占位，无实际计算）

```
# ============================================================
# CPU 实现（存根，不做实际计算）
# 文件：linearize_cache_indices.cpp  第 25~30 行
# ============================================================

function linearize_cache_indices_from_row_idx_cpu(
    cache_hash_size_cumsum,   # 忽略（参数名注释掉）
    update_table_indices,     # 忽略（参数名注释掉）
    update_row_indices        # 仅用于确定输出形状和 dtype
) -> Tensor:

    # 仅分配同形状的空 Tensor，不填充任何值
    return at::empty_like(update_row_indices)

# ⚠️ 注意：该算子在 CPU 上无实际语义，
#    实际使用必须将三个输入 Tensor 置于 CUDA 设备上。
```

---

### 7.5 伪代码执行流程总览

```
调用方（Python）
    │
    │  torch.ops.fbgemm.linearize_cache_indices_from_row_idx(
    │      cache_hash_size_cumsum,        # GPU 内存
    │      update_table_indices,          # GPU 内存
    │      update_row_indices             # GPU 内存
    │  )
    │
    ▼
【Host C++ 函数：linearize_cache_indices_from_row_idx_cuda】
    │
    ├─ [1] 设备校验：三张量在同一 CUDA GPU
    ├─ [2] 锁定设备上下文（CUDA_DEVICE_GUARD）
    ├─ [3] 提取 T，断言 T > 0
    ├─ [4] 分配输出 Tensor（GPU，empty_like）
    ├─ [5] N == 0 ? ──YES──► 返回空 Tensor
    │               NO
    ├─ [6] dtype 分发（int32 / int64）
    ├─ [7] 计算 grid = ⌈N/kMaxThreads⌉
    └─ [8] 异步启动 CUDA Kernel ──────────────────────────────┐
                                                              │
    ◄─────────────────────────────────────────────────────────┘
    │  （CPU 立即返回，GPU 异步执行）
    └─ [9] 返回 linear_cache_indices Tensor

                    ┌─────────────────────────────────────┐
                    │  GPU 异步执行：N 个线程并行          │
                    │                                      │
                    │  Thread 0    Thread 1  ...  Thread N-1
                    │     │           │                │
                    │  [A] 计算 index（全局线程ID）      │
                    │  [B] 越界检查（index >= N → return）│
                    │  [C] 读 update_table_indices[index] │
                    │      → 得到表号 t                   │
                    │  [D] __ldg 读 cumsum[T] → max_offset│
                    │  [E] __ldg 读 cumsum[t] → curr_off  │
                    │  [F] 双重判断：                      │
                    │      curr_off>=0 AND row_idx>=0?    │
                    │        YES → output = row_idx+curr_off
                    │        NO  → output = max_offset    │
                    └─────────────────────────────────────┘
```

---

## 八、NPU 算子精度测试用例设计

### 8.1 测试策略

**验证目标**：NPU kernel 输出与 CPU 参考实现逐元素完全相等。

**精度标准**：`torch.equal`（整数运算，无浮点误差，要求完全相等，不设容差）。

**数据类型**：仅 `int32` / `int64`（算子语义为索引映射，不涉及浮点计算，f16/f32/bf16 不适用）。

**测试框架**：

```
同一份输入数据
    │
    ├──→ CPU 参考实现（纯 Python）──→ golden（正确答案）
    │
    └──→ torch.ops.fbgemm.linearize_cache_indices_from_row_idx（NPU）──→ npu_result
                                                                              │
                                                              torch.equal(npu_result.cpu(), golden)
                                                                              │
                                                              True  → NPU kernel 正确
                                                              False → NPU kernel 有 bug
```

---

### 8.2 CPU 参考实现（golden 生成）

```python
def ref_impl(cache_hash_size_cumsum, update_table_indices, update_row_indices):
    dtype  = update_row_indices.dtype
    cumsum = cache_hash_size_cumsum.cpu().to(torch.int64)
    table  = update_table_indices.cpu().to(torch.int64)
    row    = update_row_indices.cpu().to(torch.int64)
    max_off  = cumsum[-1].item()
    curr_off = cumsum[table]
    valid    = (curr_off >= 0) & (row >= 0)
    output   = torch.where(valid, row + curr_off,
                           torch.full_like(row, max_off))
    return output.to(dtype=dtype)
```

---

### 8.3 测试用例

#### TC-01：全部表已缓存（等大小表）

**目的**：验证正常路径，所有表均在缓存中，所有行均有效。

```
输入：
  cache_hash_size_cumsum = [0, 12, 24, 36, 48]   # 4 张表，每张容量 12，哨兵=48
  update_table_indices   = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
  update_row_indices     = [10, 2, 3, 7, 1, 4, 5, 9, 2, 7, 6, 8, 5, 1, 0, 4]
  dtype                  = int32 / int64

期望输出：
  [10, 2, 3, 7, 13, 16, 17, 21, 26, 31, 30, 32, 41, 37, 36, 40]

验证逻辑：
  table=0: off=0,  row+off = [10,2,3,7]
  table=1: off=12, row+off = [13,16,17,21]
  table=2: off=24, row+off = [26,31,30,32]
  table=3: off=36, row+off = [41,37,36,40]
```

---

#### TC-02：部分表未缓存（cumsum 含 -1）

**目的**：验证未缓存表的哨兵值输出路径（`curr_offset < 0` 分支）。

```
输入：
  cache_hash_size_cumsum = [0, 12, -1, 24, 36]   # table_2 未缓存，哨兵=36
  update_table_indices   = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
  update_row_indices     = [10, 2, 3, 7, 1, 4, 5, 9, 2, 7, 6, 8, 5, 1, 0, 4]
  dtype                  = int32 / int64

期望输出：
  [10, 2, 3, 7, 13, 16, 17, 21, 36, 36, 36, 36, 29, 25, 24, 28]

验证逻辑：
  table=0: off=0,  正常 → [10,2,3,7]
  table=1: off=12, 正常 → [13,16,17,21]
  table=2: off=-1, 未缓存 → 全部输出哨兵 36
  table=3: off=24, 正常 → [29,25,24,28]
```

---

#### TC-03：不均匀 update_table_indices 分布

**目的**：验证每条记录的表号独立查表，不依赖连续分布假设。

```
输入：
  cache_hash_size_cumsum = [0, 12, -1, 24, 36]   # table_2 未缓存，哨兵=36
  update_table_indices   = [0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3]
  update_row_indices     = [10, 2, 3, 7, 1, 4, 5, 9, 2, 7, 6, 8, 5, 1, 0, 4]
  dtype                  = int32 / int64

期望输出：
  [10, 2, 3, 19, 13, 16, 17, 21, 36, 36, 36, 36, 36, 36, 24, 28]

验证逻辑：
  table=0: off=0,  3条 → [10,2,3]
  table=1: off=12, 5条 → [19,13,16,17,21]
  table=2: off=-1, 6条 → 全部哨兵 36
  table=3: off=24, 2条 → [24,28]
```

---

#### TC-04：含剪枝行（row_idx 为负数）

**目的**：验证 `update_row_indices[i] < 0` 时输出哨兵值（`row_idx < 0` 分支）。

```
输入：
  cache_hash_size_cumsum = [0, 12, 24, 36, 48]   # 全部缓存，哨兵=48
  update_table_indices   = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
  update_row_indices     = [10, -1, 3, 7, 1, 4, -1, 9, 2, -1, 6, 8, 5, 1, -1, 4]
  dtype                  = int32 / int64

期望输出：
  [10, 48, 3, 7, 13, 16, 48, 21, 26, 48, 30, 32, 41, 37, 48, 40]

验证逻辑：
  row=-1 的位置（idx=1,6,9,14）输出哨兵 48，其余正常映射
```

---

#### TC-05：部分未缓存 + 含剪枝行（双重无效）

**目的**：同时覆盖两种无效路径，验证 OR 逻辑正确。

```
输入：
  cache_hash_size_cumsum = [0, 12, -1, 24, 36]   # table_2 未缓存，哨兵=36
  update_table_indices   = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
  update_row_indices     = [10, -1, 3, 7, 1, 4, -1, 9, 2, -1, 6, 8, 5, 1, -1, 4]
  dtype                  = int32 / int64

期望输出：
  [10, 36, 3, 7, 13, 16, 36, 21, 36, 36, 36, 36, 29, 25, 36, 28]

验证逻辑：
  table=2 全部哨兵（未缓存）
  table=0/1/3 中 row=-1 的位置也输出哨兵
```

---

#### TC-06：空输入（N=0）

**目的**：验证空输入快速返回路径，输出为空张量且 dtype 一致。

```
输入：
  cache_hash_size_cumsum = [0, 12, 24]
  update_table_indices   = []   # N=0
  update_row_indices     = []   # N=0
  dtype                  = int32 / int64

期望输出：
  []   # 空张量，numel=0，dtype 与输入一致

验证逻辑：
  result.numel() == 0
  result.dtype == dtype
```

---

#### TC-07：单条记录（N=1）

**目的**：验证最小非空输入，覆盖正常/剪枝/未缓存三种情形。

```
子用例 A（正常）：
  cumsum=[0,10,20], table=[1], row=[5], dtype=int32/int64
  期望：[15]   # 5 + 10 = 15

子用例 B（剪枝行）：
  cumsum=[0,10,20], table=[0], row=[-1], dtype=int32/int64
  期望：[20]   # 哨兵 20

子用例 C（未缓存表）：
  cumsum=[0,-1,20], table=[1], row=[5], dtype=int32/int64
  期望：[20]   # 哨兵 20

子用例 D（边界：row=0, off=0）：
  cumsum=[0,10,20], table=[0], row=[0], dtype=int32/int64
  期望：[0]    # 0 + 0 = 0
```

---

#### TC-08：全部表未缓存

**目的**：验证所有 cumsum 均为 -1 时，全部输出哨兵值。

```
输入：
  cache_hash_size_cumsum = [0, -1, -1, -1, 36]   # 哨兵=36
  update_table_indices   = [0, 1, 2, 3, 0, 1, 2, 3]
  update_row_indices     = [1, 2, 3, 4, 5, 6, 7, 8]
  dtype                  = int32

期望输出：
  [36, 36, 36, 36, 36, 36, 36, 36]   # 全部哨兵
```

---

#### TC-09：大批量随机数据（N=4096，T=8）

**目的**：验证大规模并行场景下 NPU kernel 与参考实现完全一致。

```
配置：
  T = 8，N = 4096
  table_2 和 table_5 标记为未缓存（cumsum=-1）
  约 5% 的 row_idx 为 -1（模拟剪枝）
  dtype = int32 / int64

验证方式：
  torch.equal(npu_result.cpu(), ref_impl(cumsum, table_idx, row_idx))

精度标准：
  完全相等（torch.equal），不设容差
```

---

### 8.4 用例覆盖矩阵

| 用例 | 全部缓存 | 部分未缓存 | 含剪枝行 | 空输入 | 单条记录 | int32 | int64 |
|------|----------|------------|----------|--------|----------|-------|-------|
| TC-01 | ✅ | | | | | ✅ | ✅ |
| TC-02 | | ✅ | | | | ✅ | ✅ |
| TC-03 | | ✅ | | | | ✅ | ✅ |
| TC-04 | ✅ | | ✅ | | | ✅ | ✅ |
| TC-05 | | ✅ | ✅ | | | ✅ | ✅ |
| TC-06 | | | | ✅ | | ✅ | ✅ |
| TC-07 | ✅ | ✅ | ✅ | | ✅ | ✅ | ✅ |
| TC-08 | | ✅（全未缓存）| | | | ✅ | |
| TC-09 | | ✅ | ✅ | | | ✅ | ✅ |

---

### 8.5 不适用的数据类型说明

| 数据类型 | 是否适用 | 原因 |
|----------|----------|------|
| `int32` | ✅ | 算子原生支持的索引类型 |
| `int64` | ✅ | 算子原生支持的索引类型 |
| `float32` | ❌ | 算子语义为整数索引映射，无浮点输入 |
| `float16` | ❌ | 同上 |
| `bfloat16` | ❌ | 同上 |

精度标准统一为 `torch.equal`（整数完全相等），无需 `atol`/`rtol` 容差。

---

## 九、与 `linearize_cache_indices` 对比

| 对比项 | `linearize_cache_indices` | `linearize_cache_indices_from_row_idx` |
|--------|--------------------------|----------------------------------------|
| 输入格式 | `(indices, offsets)` 稀疏格式 | `(update_table_indices, update_row_indices)` 行索引对 |
| 查表方式 | 二分查找 `offsets` O(log T) | 直接下标访问 `cumsum` O(1) |
| 适用场景 | 前向推理查找（forward lookup） | in-place 嵌入更新（embedding update） |
| 支持 VBE | ✅（通过 `B_offsets` 参数） | ❌ 不支持 |
| 输出 dtype | 固定 `int64` | 与输入 `update_row_indices` 相同 |
| CUDA Kernel 复杂度 | O(log T) per thread | O(1) per thread |

| 用例描述                             | 用例模型                                                                                                                                                                            | 预期结果                                                                   |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| **基础功能测试（不同表数量、缓存大小、更新数量、索引类型）** | `num_tables ∈ {1,4,8,16}`，`table_cache_size ∈ {12,64,256}`，`num_updates ∈ {0,1,16,128,1024}`，`index_dtype ∈ {torch.int32, torch.int64}`<br>随机生成 `table_indices` 和 `row_indices` | 输出形状与输入 `row_indices` 相同，类型与 `index_dtype` 相同，计算结果与 reference 函数一致     |
| **空输入序列**                        | `num_updates = 0`，`table_indices = []`，`row_indices = []`                                                                                                                       | 输出为空 tensor（`numel() == 0`）                                            |
| **未缓存表测试（特殊序列）**                 | `uncached_tables={1,3}`，其他参数如 `num_tables=4, table_cache_size=12, num_updates=32`，随机生成 `table_indices` 和 `row_indices`                                                          | 对未缓存表的行索引结果使用 sentinel（`max_offset`），其他表计算正常，类型正确                      |
| **被剪枝的行索引（异常/特殊序列）**             | `pruned_ratio = 0.3`，随机生成 `row_indices`，部分值设为 `-1`                                                                                                                              | 被剪枝（`row_indices=-1`）的输出结果为 sentinel，其他正常                              |
| **边界用例**                         | `EDGE_CASE_PARAMS = [(1,1,1),(1,1,0),(2,10,5),(32,512,2048)]`，随机生成索引，`index_dtype ∈ {torch.int32, torch.int64}`                                                                 | 输出与 reference 结果一致，类型与输入索引类型一致                                         |
| **文档示例用例**                       | `cumsum = [0,12,-1,24,36]`，`table_indices = [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3]`，`row_indices = [10,2,3,7,1,4,5,9,2,7,6,8,5,1,0,4]`                                               | 输出与示例 `expected = [10,2,3,7,13,16,17,21,36,36,36,36,29,25,24,28]` 完全一致 |
| 不一致类型                                              | 表现                                | 是否会异常                           |
| -------------------------------------------------- | --------------------------------- | ------------------------------- |
| `update_table_indices` 与 `update_row_indices` 长度不同 | 遍历循环时 i 超出较短数组                    | Python 会报 IndexError → **真正异常** |
| `cache_hash_size_cumsum` 长度 < `num_tables + 1`     | 访问 `cumsum[t]` 时 t >= len(cumsum) | IndexError → **真正异常**           |
| 空数组（长度 0）                                          | 函数返回空 tensor                      | 不是异常，仅特殊情况                      |

分核方式
仅在 dim=0 均匀分片每个 AI Core 处理一段连续的 K 元素。
分块大小
block_size = 256
block_num = ceil(K / 256)
第 i 个 block 处理区间
plaintext
start = i * 256
end = start + 256
实际长度 = end - start
4. 核执行流程（dim=0 块）
1) CopyIn (GM → UB)
读取当前块：table_indices[start:end]
读取当前块：row_indices[start:end]
全量读取 cumsum
2) Compute
cpp
运行
for (int i = 0; i < block_len; i++) {
    table_id = ub_table[i];
    row_id = ub_row[i];
    ub_out[i] = ub_cumsum[table_id] + row_id;
}
3) CopyOut (UB → GM)
写回 linear_cache_indices[start:end]
