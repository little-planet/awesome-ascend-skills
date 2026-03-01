# 使用 JSON + graph/types.h 驱动 op_host/op_kernel 对齐

算子工程中常有描述接口的 JSON（如 `moe_init_routing_grouped_matmul_grad.json`），列出每个输入/输出的 name、param_type、type、format。为保持 op_host / infershape / tiling / op_kernel 一致，建议按以下步骤处理。

1. **从 JSON 提取接口**
   - 读取 `input_desc[i].name/type/format`、`output_desc[j].name/type/format`，区分数据张量与索引/标量。

2. **在 op_host/*_def.cpp 中映射 DataType/Format**
   - DataType 用 `ge::DataType`（见 `graph/types.h` 的 enum DataType，约 L80–L123）。
   - Format 用 `ge::Format`（同一文件 enum Format，约 L189–L247）。
   - 映射示例：`"fp16"`→`DT_FLOAT16`，`"bf16"`→`DT_BF16`，`"float"`→`DT_FLOAT`，`"int32"`→`DT_INT32`，`"ND"`→`FORMAT_ND`。
   - 每个 Input/Output 的 `.DataType({...})`、`.Format({...})`、`.UnknownShapeFormat({...})` 元素个数必须相同（见 [type_format_reference.md](type_format_reference.md)）。

3. **在 infershape 中对齐 shape**
   - 用 JSON 语义确定输出 shape 与哪一输入对齐；在 `*_infershape.cpp` 中用 `GetInputShape`/`GetOutputShape`、SetDimNum/SetDim，索引与 JSON/op_host 顺序一致。

4. **在 tiling 中做一致性校验**
   - 从 TilingContext 取关键输入 shape/dtype；用 JSON 的 type 列表构造 `supportedDtype`；维度约束与 kernel 假设一致，不符则返回 GRAPH_FAILED 并打日志。

5. **op_kernel 命名与接口一致**
   - tiling 结构体字段、GM 张量命名与 JSON/语义对应，避免含义不清的 x1/x2/y。

6. **注意**
   - 不手写随意枚举值，从 `graph/types.h` 查找；JSON 新增类型时先在 types.h 确认并核对内核是否支持；若 JSON 与现有实现冲突，以内核能力为上限并注释说明。
