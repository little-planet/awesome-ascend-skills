## AscendC Matmul / Cube 编写速查

本文件给 Agent 一个「如何在 AscendC 内核里正确调用 Cube 做矩阵乘」的标准模板，优先参考 `grouped_matmul` 与 `grouped_matmul_finalize_routing` 的实现。

### 1. 核心类型与 MatmulImpl 定义

1. **选择 A/B/C/Bias 的位置、格式和基础 dtype**：

- 一般约定：
  - A（输入 x）：`GM + ND`
  - B（权重 w）：`GM + NZ`（Cube 友好的权重布局）
  - C（中间输出）：`GM + ND`，int32 累加（量化场景）
  - Bias：`GM + ND`

2. **典型定义方式**（int8×int8→int32）：

```cpp
using aT    = MatmulType<TPosition::GM, CubeFormat::ND, int8_t>;   // X
using bT    = MatmulType<TPosition::GM, CubeFormat::NZ, int8_t>;   // W
using BiasT = MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;  // bias（可选）
using cT    = MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;  // C

// CFG_MDL 来源于 matmul_intf.h 中的默认配置，或根据场景定制
using MT = matmul::MatmulImpl<aT, bT, cT, BiasT, CFG_MDL>;
```

- 上述 `MT` 就是 **Cube GEMM 内核封装**，后续通过它调用硬件矩阵乘。

### 2. 基本调用流程（单个 Matmul Block）

> 适用于「A: m×k, B: k×n → C: m×n」的单块计算；多块/多组在第 3 节说明。

典型流程：

```cpp
// 1. 绑定 GM 地址（通常在 Init 阶段完成一次）
GlobalTensor<int8_t>    xGm;
GlobalTensor<int8_t>    weightGm;
GlobalTensor<int32_t>   cGm;        // 或 workspace

xGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(initParams.x));
weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(initParams.weight));
cGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(initParams.workspace));

// 2. 创建 MatmulImpl 对象（通常作为成员在构造时注入）
MT mm;

// 3. 在真正计算前配置「整体形状」与「当前 block 形状」
int64_t M = mAll;  // 当前 group 总 token 数
int64_t N = nAll;  // 输出维度
int64_t K = kAll;  // 中间维度

int64_t curM = curSingleM;  // 本 block 的 M
int64_t curN = curSingleN;  // 本 block 的 N

mm.SetOrgShape(M, N, K);             // 整体 (M, N, K)，供内部做 Tail 处理与调优
mm.SetSingleShape(curM, curN, K);    // 本 block (curM, curN, K)

// 4. 绑定 A/B：xOffset/weightOffset 由上层根据 block 索引和 tiling 计算
mm.SetTensorA(xGm[xOffset]);           // A: curM×K
mm.SetTensorB(weightGm[weightOffset]); // B: K×curN

// 5. K 轴多步累加：Iterate + GetTensorC
uint64_t cOffset = workspaceOffset;  // 为当前 block 分配的 C 区域起点
while (mm.Iterate()) {
    // 最后一个参数 true 表示累加写 C（多次迭代叠加到同一 C 上）
    mm.GetTensorC(cGm[cOffset], 0, true);
    cOffset += (baseM * baseN);      // 若采用分片 workspace，这里按 tiling 的块大小推进
}
```

关键点：

- **SetOrgShape vs SetSingleShape**：
  - `SetOrgShape(M, N, K)` 描述逻辑上的整体矩阵，用于 tail 处理、性能模型等。
  - `SetSingleShape(curM, curN, K)` 描述当前一次计算的子块。
- **Iterate / GetTensorC**：
  - `Iterate()` 依据 tiling 中的 `stepKa/stepKb` 沿 K 轴多次分段，配合硬件双缓冲。
  - 每次 `Iterate()` 后调用 `GetTensorC(...)` 把当前累加结果写入 C。

### 3. 分块与多组（Grouped Matmul 模式）

当存在 **多组专家或需要按 groupList 切分 M/K** 时，一般模式是：

1. Host 侧 tiling（如 `grouped_matmul_tiling.cpp`）计算：
   - `baseM/baseN/baseK`：单块大小。
   - 每组的 `mList/kList/nList`，以及 `groupNum`、`totalM` 等。
2. Device 侧通过类似 `MNConfig` + `GMMProcess` 或自定义结构，负责：
   - 根据 `blockIdx` 计算 (mIdx, nIdx)。
   - 根据 groupList / mList/kList/nList 计算当前 group 的 (m,k,n) 以及本 block 的 `(curM,curN)`。
   - 计算 A/B/C 在 GM/Workspace 中的偏移，再调用上节的 Matmul 调用模板。

参考逻辑（伪代码）：

```cpp
for groupIdx in [0, groupNum):
    m = groupM[groupIdx]
    blockDimM = Ceil(m / baseM)
    blockDimN = Ceil(n / baseN)
    for each block (mIdx, nIdx):
        curSingleM = (mIdx == blockDimM-1) ? m - mIdx*baseM : baseM
        curSingleN = (nIdx == blockDimN-1) ? n - nIdx*baseN : baseN

        xOffset      = (offsetM + mIdx*baseM) * K
        weightOffset = groupIdx * N * K + (nIdx*baseN) * K  // NZ 时注意 align
        workspaceOffset = ComputeWorkspaceOffset(coreIdx, blockIdx, baseM, baseN, parallNum)

        // 按「2. 基本调用流程」填入 SetOrgShape/SetSingleShape/SetTensorA/B/Iterate/GetTensorC
```

`grouped_matmul.h` 中的 `MNConfig` / `GMMProcess` 就是这种模式的标准实现。

### 4. 与 grouped_matmul_finalize_routing 的特例

`grouped_matmul_finalize_routing` 在标准 Matmul 基础上增加了：

1. **workspace 分片**：Cube 只把 int32 GEMM 结果写到 workspace（`mmOutGm`），不直接写最终 y。
   - `workSpaceOffset = singleM * singleN * (coreIdx + (cubeCount % parallNum) * coreNum);`
   - 不同 core / 不同并行度用 disjoint 的 offset 区间，避免 C 冲突。

2. **AIC / AIV 协作**：
   - AIC（Cube 核）：
     - 负责 `MMCompute`：MatmulImpl 调用，`GetTensorC` 写 int32 workspace，每次迭代后 `CrossCoreSetFlag(SYNC_AIC_TO_AIV)`。
   - AIV（Vector 核）：
     - 在 `VectorCompute` 中 `CrossCoreWaitFlag(SYNC_AIC_TO_AIV)` 等待 AIC 写完该块。
     - 通过 `DataCopyMMOut` 从 workspace 把 int32 C 搬到 UB。
     - 调 `AscendDequant`，结合 scale / per-token scale / bias / logits 生成最终 `float/bfloat16`。
     - 通过 `VectorAtomicProcess`（必要时配合 `SetAtomicAdd`）写回 y 并按 tokenRanks 做路由聚合。

3. **确定性模式**：
   - `tiling->deterministicFlag == 1` 时：
     - 先把每个核的贡献写到 `mmQuantOutGm`（一个额外的 GM buffer）。
     - `FRDeterministic` 里按 `(tokenIdx % (coreNum*taskRation)) == blockIdx` 的规则分配行到各核，再用原子加把同一行的贡献汇总到 y。

当你实现新的「GMM + finalize routing」类算子时：

- 若仅需简单 GEMM，可以直接按第 2 节模板调用 `MatmulImpl`，把结果写到 y。
- 若需要「量化 + per-token scale + logits + deterministic 聚合」，可优先复用 `grouped_matmul_finalize_routing` 的：
  - AIC 侧 `MMCompute` 调用模式。
  - AIV 侧 `VectorCompute` + `AscendDequant` + `VectorAtomicProcess` 模式。
  - workspace / CrossCoreFlag / FRDeterministic 的整体设计。

