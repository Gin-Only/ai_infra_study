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

## 七、与 `linearize_cache_indices` 对比

| 对比项 | `linearize_cache_indices` | `linearize_cache_indices_from_row_idx` |
|--------|--------------------------|----------------------------------------|
| 输入格式 | `(indices, offsets)` 稀疏格式 | `(update_table_indices, update_row_indices)` 行索引对 |
| 查表方式 | 二分查找 `offsets` O(log T) | 直接下标访问 `cumsum` O(1) |
| 适用场景 | 前向推理查找（forward lookup） | in-place 嵌入更新（embedding update） |
| 支持 VBE | ✅（通过 `B_offsets` 参数） | ❌ 不支持 |
| 输出 dtype | 固定 `int64` | 与输入 `update_row_indices` 相同 |
| CUDA Kernel 复杂度 | O(log T) per thread | O(1) per thread |
