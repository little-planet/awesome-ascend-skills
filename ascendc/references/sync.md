# 确定性核间同步（InitDetermineComputeWorkspace / WaitPreBlock / NotifyNextBlock）

供 Skill 与 Agent 阅读理解：在 AscendC 中如何用「确定性 workspace」实现多核保序执行（按核 0→1→…→N-1 顺序，保证每次写 GM 时上一核已写完）。

---

## 何时查阅本文档

- 需要在 **多核间保证执行顺序**（例如按 GID 保序写 grad_x、按 block 顺序原子累加）时；
- 使用 **InitDetermineComputeWorkspace / WaitPreBlock / NotifyNextBlock** 任一接口时；
- 实现「**按核顺序**」的 CopyOut / AtomicAdd 等写回逻辑时。

---

## 功能概述

三个接口配合使用，在 **同一链上** 形成「核 0 → 核 1 → … → 核 N-1」的严格顺序：

1. **InitDetermineComputeWorkspace**：初始化 GM 共享区（gmWorkspace），之后才能调用 WaitPreBlock / NotifyNextBlock。
2. **WaitPreBlock**：当前核读 gmWorkspace，若满足「上一核已完成」则继续执行；否则等待。
3. **NotifyNextBlock**：当前核写 gmWorkspace，通知下一核「本核已完成，可继续」。

典型用法：每个核在「写 GM（如累加到 grad_x）」前调用 **WaitPreBlock**，写完后调用 **NotifyNextBlock**，从而实现按核序的保序写回。

---

## API 说明（Skill 阅读理解用）

### InitDetermineComputeWorkspace

- **原型**：`__aicore__ inline void InitDetermineComputeWorkspace(GlobalTensor<int32_t>& gmWorkspace, LocalTensor<int32_t>& ubWorkspace)`
- **含义**：用 ubWorkspace 操作 gmWorkspace，完成核间同步用 GM 的初始化；**必须先调用本接口**，再调用 WaitPreBlock / NotifyNextBlock。
- **调用时机**：在 Process 或主循环开始前，**每个参与同步的核调用一次**；与示例中在 `Process()` 开头、`for` 循环前调用一致。

### WaitPreBlock

- **原型**：`__aicore__ inline void WaitPreBlock(GlobalTensor<int32_t>& gmWorkspace, LocalTensor<int32_t>& ubWorkspace)`
- **含义**：读 gmWorkspace，判断当前核是否可以继续执行；当「上一核」已通过 NotifyNextBlock 更新 GM 后，本核才返回并继续。
- **调用时机**：在「本核要写 GM（如 DataCopy 到 grad_x、AtomicAdd）」**之前** 调用；与示例中在 `DataCopy(dst,...)` 前调用一致。

### NotifyNextBlock

- **原型**：`__aicore__ inline void NotifyNextBlock(GlobalTensor<int32_t>& gmWorkspace, LocalTensor<int32_t>& ubWorkspace)`
- **含义**：写 gmWorkspace，通知「下一核」本核已完成，下一核的 WaitPreBlock 可被满足。
- **调用时机**：在「本核写完 GM」**之后** 调用；与示例中在 `DataCopy(dst,...)`、`SetAtomicNone()` 后调用一致。

**成对与次数**：每个核的 **WaitPreBlock 与 NotifyNextBlock 调用次数必须相同**，且所有核调用次数一致（例如每核每轮各 1 次）。

---

## 空间约束（Host 与 Kernel 必须满足）

| 对象 | 最小大小 | 说明 |
|------|----------|------|
| **gmWorkspace**（GM） | `blockNum * 32` Bytes | blockNum = 调用的核数，Kernel 侧用 `GetBlockNum()`；Host 侧 Tiling 分配 workspace 时需 ≥ 该值。 |
| **ubWorkspace**（Local/UB） | `blockNum * 32 + 32` Bytes | 用于操作 gmWorkspace 的临时 Local 张量，由 TQue AllocTensor 分配。 |

Kernel 侧分配 ubWorkspace 示例：`pipe->InitBuffer(syncDeterUbQueue, 1, GetBlockNum() * 32U + 32U);`，再从该队列 `AllocTensor<int32_t>()` 得到 ubWorkspace。

---

## 产品与模式约束（Skill 必读）

- **产品支持**：Atlas A2、AI Core 支持；**Atlas 推理系列产品 Vector Core 不支持**。若算子运行在 Vector Core（AIV）上，可能报错或未定义行为，需改用其他同步方式（如 CrossCoreSetFlag 等）。
- **分离模式（AIC + AIV）**：在分离模式下，**仅对 AIV 核生效**；且 **WaitPreBlock 与 NotifyNextBlock 之间仅支持插入矢量计算相关指令**，矩阵计算相关指令不生效。
- **blockDim**：算子调用时指定的逻辑 blockDim 必须 **不大于** 实际运行该算子的 AI 处理器核数，否则多轮调度可能插入异常同步导致 Kernel 卡死。

---

## 使用步骤（按顺序实现）

1. **Host/Tiling**：为「确定性同步」预留 GM 段，大小 ≥ `blockNum * 32` Bytes，并将该段首地址通过 tiling/workspace 下发给 Kernel（如 `deterWorkspaceSyncInfoOffsetPtr`）。
2. **Kernel Init**：将 workspace 中该段绑定为 `GlobalTensor<int32_t> gmWorkspace`（如 `syncDeterGm`）；为 ubWorkspace 准备 TQue，大小 `GetBlockNum()*32+32`，在 InitVector 等处 `InitBuffer`。
3. **Process 入口（每核一次）**：在进入主循环前，`ubWorkspace = syncDeterUbQueue.AllocTensor<int32_t>()`，调用 **InitDetermineComputeWorkspace(syncDeterGm, ubWorkspace)**。
4. **每轮写 GM 前**：调用 **WaitPreBlock(syncDeterGm, ubWorkspace)**。
5. **本核写 GM**：执行 DataCopy / SetAtomicAdd / DataCopy / SetAtomicNone 等。
6. **写 GM 后**：调用 **NotifyNextBlock(syncDeterGm, ubWorkspace)**。
7. **Process 结束**：`syncDeterUbQueue.FreeTensor(ubWorkspace)`。

这样同一链上执行顺序为：核0 写 → 核1 写 → … → 核N-1 写；若外层按 GID 递增，则实现「按 GID 保序、同一 nBlockIdxG 上上一 GID 先写完」的语义。

---

## 标准示例（与官方文档一致）

```cpp
// 每核 Process() 内：
AscendC::LocalTensor<int32_t> ubWorkspace = m_queTmp.AllocTensor<int32_t>();
AscendC::InitDetermineComputeWorkspace(m_gmWorkspace, ubWorkspace);
for (int64_t i = 0; i < m_tileNum; i++) {
    // copy in
    AscendC::LocalTensor<T> srcLocal = m_que.AllocTensor<T>();
    AscendC::DataCopy(srcLocal, m_srcGlobal[i * m_tileCount], m_tileCount);
    // copy out（保序）
    AscendC::WaitPreBlock(m_gmWorkspace, ubWorkspace);
    AscendC::SetAtomicAdd<T>();
    AscendC::DataCopy(m_dstGlobal[i * m_tileCount], srcLocal, m_tileCount);
    AscendC::SetAtomicNone();
    AscendC::NotifyNextBlock(m_gmWorkspace, ubWorkspace);
    m_que.FreeTensor(srcLocal);
}
m_queTmp.FreeTensor(ubWorkspace);
```

要点：Init 一次；每轮「写 GM」前 WaitPreBlock、写完后 NotifyNextBlock；每核调用次数一致。

---

## 官方文档参考

- [InitDetermineComputeWorkspace](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0206.html)
- [WaitPreBlock](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0207.html)
- [NotifyNextBlock](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0208.html)

实现与排错时以官方文档约束与示例为准；本文档供 Skill 统一理解用法与顺序。

---

# SyncAll（全核同步）

当多核操作同一块全局内存且存在读后写、写后读、写后写等数据依赖时，通过 SyncAll 插入同步，避免数据竞争。分为**软同步**（需 gmWorkspace/ubWorkspace）和**硬同步**（无参，硬件全核同步指令）。

---

## 何时查阅 SyncAll

- 需要**所有核到达同一同步点**再继续（例如：各核先写中间结果到 GM，再一起读 GM 做跨核累加）时；
- 融合算子中需要 **AIC 与 AIV 之间**或 **多核 Vector 之间**做一次全核栅栏时；
- 选择**软同步**（需 workspace）还是**硬同步**（无参、分离模式推荐）时。

---

## SyncAll API 概要

| 类型 | 原型 | 说明 |
|------|------|------|
| **软同步** | `template <bool isAIVOnly = true> void SyncAll(const GlobalTensor<int32_t>& gmWorkspace, const LocalTensor<int32_t>& ubWorkspace, const int32_t usedCores = 0)` | 需要 gmWorkspace（已初始化为 0）和 ubWorkspace；usedCores 不传则全核同步。**软同步不支持 isAIVOnly=false**（即融合算子全核同步只能用硬同步）。 |
| **硬同步** | `template <bool isAIVOnly = true> void SyncAll()` | 无参，由硬件保证全核同步。**分离模式下建议用硬同步**；软同步仅适用于纯 Vector 且性能较低。 |

- **isAIVOnly**：`true`（默认）表示仅对 Vector 核做全核同步；`false` 表示融合算子下先分别完成 Vector 与 Cube 的全核同步，再做两者之间的同步（仅硬同步支持）。
- **空间**：软同步时 gmWorkspace ≥ 核数×32 Bytes 且**须初始化为 0**（Host 侧或 Kernel 侧每核初始化全部）；ubWorkspace ≥ 核数×32 Bytes。
- **blockDim**：逻辑 blockDim 不得大于实际运行的 AI 处理器核数，否则多轮调度可能卡死。

---

## SyncAll 官方示例总结（带注释）

以下为 [SyncAll 文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0204.html) 示例的整理与注释：**8 核、每核处理 32 个 float**，先各自乘 2 写入公共 workGm，**SyncAll 等待所有核写完**，再每核从 workGm 读取其他核数据做累加，最后写回 dstGm。输入 srcGm 全 1，输出 dstGm 全 16（2×8=16）。

```cpp
#include "kernel_operator.h"

// 软同步所需 GM 缓存单位：每核 32 Bytes（与约束一致）
const int32_t DEFAULT_SYNCALL_NEED_SIZE = 8;  // 8 * sizeof(int32_t) = 32 Bytes

class KernelSyncAll {
public:
    __aicore__ inline KernelSyncAll() {}

    // Init：绑定 GM（src/dst/work/sync），为软同步分配 workQueue（ubWorkspace）
    __aicore__ inline void Init(__gm__ uint8_t* srcGm, __gm__ uint8_t* dstGm, __gm__ uint8_t* workGm,
                                __gm__ uint8_t* syncGm)
    {
        blockNum = AscendC::GetBlockNum();           // 参与同步的核总数
        perBlockSize = srcDataSize / blockNum;       // 每核处理的数据个数（float）
        blockIdx = AscendC::GetBlockIdx();           // 当前核 ID

        // 每核只看自己那一段 src/dst
        srcGlobal.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(srcGm + blockIdx * perBlockSize * sizeof(float)),
            perBlockSize);
        dstGlobal.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(dstGm + blockIdx * perBlockSize * sizeof(float)),
            perBlockSize);
        // 公共工作区：所有核都会写/读，用于跨核数据交换
        workGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workGm), srcDataSize);
        // 软同步用 GM 缓存：大小 blockNum * 32 Bytes，入口前需在 Host 侧初始化为 0
        syncGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(syncGm),
                                   blockNum * DEFAULT_SYNCALL_NEED_SIZE);

        pipe.InitBuffer(inQueueSrc1, 1, perBlockSize * sizeof(float));
        pipe.InitBuffer(inQueueSrc2, 1, perBlockSize * sizeof(float));
        // ubWorkspace：软同步要求 ≥ blockNum*32 Bytes
        pipe.InitBuffer(workQueue, 1, blockNum * DEFAULT_SYNCALL_NEED_SIZE * sizeof(int32_t));
        pipe.InitBuffer(outQueueDst, 1, perBlockSize * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        CopyIn();                                    // 每核：src -> Local
        FirstCompute();                              // 每核：Local * 2 -> outQueueDst
        CopyToWorkGlobal();                          // 每核：本核结果写入 workGm[blockIdx*perBlockSize]

        // ---------- 全核同步：等所有核都写完 workGm 再继续 ----------
        AscendC::LocalTensor<int32_t> workLocal = workQueue.AllocTensor<int32_t>();
        AscendC::SyncAll(syncGlobal, workLocal);     // 软同步：读/写 syncGlobal，保证全核到达此处
        workQueue.FreeTensor(workLocal);

        // ---------- 同步点之后：每核从 workGm 读其他核数据并累加 ----------
        AscendC::LocalTensor<float> srcLocal2 = inQueueSrc2.DeQue<float>();
        AscendC::LocalTensor<float> dstLocal = outQueueDst.AllocTensor<float>();
        DataCopy(dstLocal, srcLocal2, perBlockSize); // 当前核的结果先拷到 dstLocal 作为累加基
        inQueueSrc2.FreeTensor(srcLocal2);

        for (int i = 0; i < blockNum; i++) {
            if (i != blockIdx) {
                CopyFromOtherCore(i);   // 从 workGm 读第 i 核的数据到 Local
                Accumulate(dstLocal);   // 累加到 dstLocal
            }
        }
        outQueueDst.EnQue(dstLocal);
        CopyOut();   // dstLocal -> dstGm
    }

private:
    // 本核结果写入公共 workGm 的对应段
    __aicore__ inline void CopyToWorkGlobal()
    {
        AscendC::LocalTensor<float> dstLocal = outQueueDst.DeQue<float>();
        AscendC::DataCopy(workGlobal[blockIdx * perBlockSize], dstLocal, perBlockSize);
        outQueueDst.FreeTensor(dstLocal);
    }

    // 从 workGm 读第 index 核的数据到 inQueueSrc1
    __aicore__ inline void CopyFromOtherCore(int index)
    {
        AscendC::LocalTensor<float> srcLocal = inQueueSrc1.AllocTensor<float>();
        AscendC::DataCopy(srcLocal, workGlobal[index * perBlockSize], perBlockSize);
        inQueueSrc1.EnQue(srcLocal);
    }

    // 将 inQueueSrc1 中刚读入的数据累加到 dstLocal
    __aicore__ inline void Accumulate(const AscendC::LocalTensor<float> &dstLocal)
    {
        AscendC::LocalTensor<float> srcLocal1 = inQueueSrc1.DeQue<float>();
        AscendC::Add(dstLocal, dstLocal, srcLocal1, perBlockSize);
        inQueueSrc1.FreeTensor(srcLocal1);
    }

    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<float> srcLocal = inQueueSrc1.AllocTensor<float>();
        AscendC::DataCopy(srcLocal, srcGlobal, perBlockSize);
        inQueueSrc1.EnQue(srcLocal);
    }

    // 每核：Local 数据乘 2，结果同时留一份到 inQueueSrc2 和 outQueueDst
    __aicore__ inline void FirstCompute()
    {
        AscendC::LocalTensor<float> srcLocal1 = inQueueSrc1.DeQue<float>();
        AscendC::LocalTensor<float> srcLocal2 = inQueueSrc2.AllocTensor<float>();
        AscendC::LocalTensor<float> dstLocal = outQueueDst.AllocTensor<float>();
        float scalarValue(2.0);
        AscendC::Muls(dstLocal, srcLocal1, scalarValue, perBlockSize);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::DataCopy(srcLocal2, dstLocal, perBlockSize);
        inQueueSrc1.FreeTensor(srcLocal1);
        inQueueSrc2.EnQue(srcLocal2);
        outQueueDst.EnQue(dstLocal);
    }

    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<float> dstLocal = outQueueDst.DeQue<float>();
        AscendC::DataCopy(dstGlobal, dstLocal, perBlockSize);
        outQueueDst.FreeTensor(dstLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueSrc1;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueSrc2;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> workQueue;   // 用作 SyncAll 的 ubWorkspace
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueDst;
    AscendC::GlobalTensor<float> srcGlobal;
    AscendC::GlobalTensor<float> dstGlobal;
    AscendC::GlobalTensor<float> workGlobal;   // 各核共享的中间 GM
    AscendC::GlobalTensor<int32_t> syncGlobal;  // 软同步用 gmWorkspace，Host 侧需初始化为 0
    int srcDataSize = 256;
    int32_t blockNum = 0;
    int32_t blockIdx = 0;
    uint32_t perBlockSize = 0;
};

extern "C" __global__ __aicore__ void kernel_syncAll_float(__gm__ uint8_t* srcGm, __gm__ uint8_t* dstGm,
                                                           __gm__ uint8_t* workGm, __gm__ uint8_t* syncGm)
{
    KernelSyncAll op;
    op.Init(srcGm, dstGm, workGm, syncGm);
    op.Process();
}
// 输入 srcGm: [1,1,...,1]（256 个）；输出 dstGm: [16,16,...,16]（每核 32 个 16，共 8*2=16）
```

要点：**先各核写 workGm → SyncAll(syncGlobal, workLocal) → 再各核读 workGm 并累加**；syncGm 在 Host 侧传入前须初始化为 0。若改用硬同步，则无需 syncGm 与 workQueue，直接调用 `SyncAll<>()` 或 `SyncAll<false>()`（融合场景）。

---

## SyncAll 官方文档参考

- [SyncAll](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0204.html)
