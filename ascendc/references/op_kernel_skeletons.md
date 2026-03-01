# AscendC op_kernel 骨架示例（FFN / GMM / MoE）

以下为三类算子的 op_kernel 命名空间与主类骨架，供复制后按需实现 Init/Process 与内部逻辑。详见工程内 `op_kernel/*.h`。

## FFN

```cpp
namespace FFN {

enum ActiveType {
    ACTIVE_GELU = 0,
    ACTIVE_RELU = 1,
    ACTIVE_FASTGELU = 2,
    ACTIVE_SILU = 3,
    ACTIVE_SIGMOID = 4,
    ACTIVE_TANH = 5
};

template <typename T, ActiveType ACTIVE, bool WITH_BIAS>
struct Param {
    using InputType = T;
    using OutputType = T;
    static constexpr ActiveType kActive = ACTIVE;
    static constexpr bool kWithBias = WITH_BIAS;
};

template <class P> class FfnCompute {
public:
    using InputType = typename P::InputType;
    using OutputType = typename P::OutputType;

    void Init(const InitParams &initParams, const FFNTiling *tiling) {
        // Initialize global tensors, UB buffer, queues, etc.
    }

    void Process() {
        // First linear: x * weight1 + bias1; activation; second linear; write back
    }

private:
    void ApplyActivation(InputType *src, OutputType *dst, uint32_t size) {
        switch (P::kActive) {
            case ACTIVE_GELU:   /* ... */ break;
            case ACTIVE_FASTGELU: /* ... */ break;
            // ...
        }
    }
};

} // namespace FFN
```

## GMM

```cpp
namespace GroupedMatmul {

template <typename T, typename WeightT, typename BiasT, typename OutputT>
struct Param {
    using InputType = T;
    using WeightType = WeightT;
    using BiasType = BiasT;
    using OutputType = OutputT;
};

template <class P> class GroupedMatmulCompute {
public:
    using InputType = typename P::InputType;
    using WeightType = typename P::WeightType;
    using BiasType = typename P::BiasType;
    using OutputType = typename P::OutputType;

    void Init(const InitParams &initParams, const GroupedMatmulTiling *tiling) {
        // Initialize global tensors, grouping info, UB buffer, queues
    }

    void Process() {
        for (uint32_t groupIdx = 0; groupIdx < tiling_->groupNum; ++groupIdx) {
            ComputeGroup(groupIdx);
        }
    }

private:
    void ComputeGroup(uint32_t groupIdx) {
        // Set input/weight/output offsets; matmul; add bias; write back
    }
};

} // namespace GroupedMatmul
```

## MoE

```cpp
namespace MoeInitRouting {

template <typename T, typename IndexT>
struct Param {
    using InputType = T;
    using IndexType = IndexT;
};

template <class P> class MoeInitRoutingCompute {
public:
    using InputType = typename P::InputType;
    using IndexType = typename P::IndexType;

    void Init(const InitParams &initParams, const MoeInitRoutingTiling *tiling) {
        // Initialize global tensors, UB buffer, queues
    }

    void Process() {
        // Expand x by rowIdx/expertIdx; write expandedXOut, expandedRowIdx, expandedExpertIdx
    }

private:
    void ExpandInput(const InputType *x, IndexType *rowIdx, IndexType *expertIdx,
                    InputType *expandedX, IndexType *expandedRowIdx, IndexType *expandedExpertIdx) {
        // Expansion logic
    }
};

} // namespace MoeInitRouting
```

## 通用注意

- 工具函数如 `DataCopyPad2D` 等保持 GM↔UB 双重重载与 `DataCopy2DDimParams` 风格。
- Init：初始化 GM、UB、队列；Process：整体流程与写回；私有方法实现具体计算。
- 与参考算子结构尽量一致，只做必要字段/张量增删与业务逻辑修改。
