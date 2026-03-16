## GlobalTensor / LocalTensor / TQue 编程要点（AscendC）

本参考用于补充说明 AscendC 中 `GlobalTensor`、`LocalTensor` 与 `TQue` 的关系与典型使用方式，便于在 Kernel 内正确管理 GM ↔ UB/Local 的搬运与流水。

---

### 1. GlobalTensor：GM 视图

- 作用：描述 GM 上的一段连续内存，提供按元素偏移（`tensor[offset]`）的访问入口。
- 典型初始化（Init 阶段）：

```cpp
GlobalTensor<float> srcGm;
uint64_t elemCount = ...;  // 以元素个数计
srcGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gmAddr), elemCount);
```

- 注意：
  - `bufferSize` 参数是 **元素个数**，不是字节数。
  - 每个核可绑定到各自的片段（例如 `gmAddr + blockIdx * perBlockSize * sizeof(T)`）。

---

### 2. LocalTensor：片上数据载体

- 作用：表示 UB/L1 等片上存储中的一段张量，算子计算 API 基本都以 `LocalTensor` 为操作对象。
- 由 `TQue` 分配和回收：

```cpp
TQue<TPosition::VECIN, 1> inQueue;
pipe.InitBuffer(inQueue, 1, tileElems * sizeof(float));

LocalTensor<float> xLocal = inQueue.AllocTensor<float>();
// ... 使用 xLocal，计算或做 DataCopy ...
inQueue.FreeTensor(xLocal);
```

---

### 3. TQue：流水与 UB 管理

- 作用：管理 LocalTensor 的生命周期，实现 CopyIn / Compute / CopyOut 三阶段流水。
- 常用位置枚举：
  - `TPosition::VECIN`：向量输入队列（GM→UB）
  - `TPosition::VECOUT`：向量输出队列（UB→GM）
  - 其他位置如 `VECCALC` 等依具体 API 而定。

#### 初始化与使用模板

```cpp
// 1) Init 阶段：为队列分配 UB 空间
TPipe pipe;
TQue<TPosition::VECIN, 1> inQueue;
pipe.InitBuffer(inQueue, 1, tileElems * sizeof(float));

// 2) CopyIn 阶段：GM -> Local
LocalTensor<float> xLocal = inQueue.AllocTensor<float>();
DataCopy(xLocal, srcGm[gmOffset], tileElems);
inQueue.EnQue(xLocal);

// 3) Compute 阶段：Local -> Local
xLocal = inQueue.DeQue<float>();
// ... 矢量/矩阵运算 ...
inQueue.FreeTensor(xLocal);

// 4) CopyOut 阶段：Local -> GM（若使用 VECOUT 队列）
TQue<TPosition::VECOUT, 1> outQueue;
pipe.InitBuffer(outQueue, 1, tileElems * sizeof(float));
LocalTensor<float> yLocal = outQueue.AllocTensor<float>();
// ... 写入 yLocal ...
outQueue.EnQue(yLocal);
yLocal = outQueue.DeQue<float>();
DataCopy(dstGm[gmDstOffset], yLocal, tileElems);
outQueue.FreeTensor(yLocal);
```

---

### 4. 确定性核间同步用 TQue（InitDetermineComputeWorkspace 场景）

配合 `InitDetermineComputeWorkspace / WaitPreBlock / NotifyNextBlock` 使用时，需要：

- Host/Tiling 侧：
  - 为 `gmWorkspace`（GlobalTensor）预留 GM 空间，大小 ≥ `blockNum * 32` Bytes。
  - 将该段首地址与长度通过 tiling 字段（如 `deterWorkspaceSyncInfoOffsetPtr` / `...BytesSize`）传给 Kernel。

- Kernel 侧：
  1. 将 GM 段绑定为 `GlobalTensor<int32_t> gmWorkspace`。
  2. 为 ubWorkspace 准备一个 `TQue<TPosition::VECIN, 1>`，其 UB 大小需满足：

```cpp
// blockNum = GetBlockNum()
const uint32_t syncUbBytes = static_cast<uint32_t>(GetBlockNum()) * 32U + 32U;
pipe.InitBuffer(syncDeterUbQueue, 1, syncUbBytes);
```

  3. 在 `Process()` 开头，每核一次：

```cpp
LocalTensor<int32_t> ubWorkspace = syncDeterUbQueue.AllocTensor<int32_t>();
InitDetermineComputeWorkspace(gmWorkspace, ubWorkspace);
```

  4. 每轮写 GM 前/后：

```cpp
WaitPreBlock(gmWorkspace, ubWorkspace);
// ... DataCopy / AtomicAdd 到目标 GM ...
NotifyNextBlock(gmWorkspace, ubWorkspace);
```

  5. 结束前释放：

```cpp
syncDeterUbQueue.FreeTensor(ubWorkspace);
```

要点：

- `gmWorkspace` 与 `ubWorkspace` 的大小约束必须同时满足，否则官方文档规定行为未定义。
- `InitDetermineComputeWorkspace` 仅需调用一次，之后在同一个 `ubWorkspace` 上重复使用 `WaitPreBlock/NotifyNextBlock`。
- 所有参与同步的核调用 `WaitPreBlock` 与 `NotifyNextBlock` 次数必须一致。

---

### 5. 与本工程的约定

- `syncDeterUbQueue` 始终作为「确定性核间同步 ubWorkspace」的队列使用，初始化时统一用 `GetBlockNum()*32+32` Bytes。
- 所有 GM ↔ UB 搬运遵循「GlobalTensor + TQue + LocalTensor」三段式模式，避免直接在 GM 上做复杂运算。

