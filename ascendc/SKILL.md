---
name: ascendc
description: Guides development of AscendC transformer ops (FFN/GMM/MoE) and CANN aclnn examples by following existing patterns under ops-transformer. Use when adding or modifying such ops, kernels, tiling/infershape, or aclnn examples.
keywords:
  - ascend
  - ascendc
  - kernel
  - npu
  - 开发环境
  - 算子
  - 昇腾
---

# AscendC Transformer 算子开发

指导 Agent 按现有模式开发/修改 AscendC 的 FFN、GMM、MoE 类算子及对应 CANN `aclnn_*` 示例。具体代码与模板见 **references** 目录。

## When to Use

- 新增或修改 FFN / GMM / MoE 类 AscendC 算子
- 补充或修改 op_host 定义、tiling/infershape、op_kernel 实现
- 编写或调整 CANN `aclnn_*` 示例与单测
- 对齐、重构或修 bug 时保持与现有算子风格一致

---

## Overall Workflow

1. **定位参考算子与示例**：按类型在 `ops-transformer/ffn/`、`gmm/`、`moe/` 下找 `*_def.cpp`、`*_tiling*.h/.cpp`、`op_kernel/*.h` 及 `examples/test_aclnn_*.cpp`。
2. **在 op_host 定义图算子接口**（Input/Output/Attr、AICore 配置、OP_ADD）。
3. **在 op_kernel 实现 AscendC 内核**（Init、Process、队列与 UB 管理）。
4. **完成 tiling、infershape 与注册**（若有对应文件则复用并改字段）。
5. **编写或更新 CANN 示例与单测**。
6. **若有 Python 前端**：按现有 test 模板补用例并做数值/形状校验。

---

## References 索引

| 文档 | 说明 |
|------|------|
| [references/type_format_reference.md](references/type_format_reference.md) | op_host 类型/格式枚举与 DataType·Format·UnknownShapeFormat 个数约定 |
| [references/ascendc_kernel_implement.md](references/ascendc_kernel_implement.md) | Kernel 开发：CopyIn/Compute/CopyOut、TQue、GlobalTensor 等 |
| [references/op_host_examples.md](references/op_host_examples.md) | FFN/GMM/MoE 的 Input/Output/Attr 定义示例代码 |
| [references/op_kernel_skeletons.md](references/op_kernel_skeletons.md) | FFN/GMM/MoE 的 op_kernel 命名空间与主类骨架 |
| [references/op_host_json_types_flow.md](references/op_host_json_types_flow.md) | 用 JSON + graph/types.h 驱动 op_host/infershape/tiling 对齐的流程 |
| [references/aclnn_example_template.md](references/aclnn_example_template.md) | aclnn 示例通用模板与生成步骤 |
| [references/genop_usage.md](references/genop_usage.md) | genop 命令、生成结构及生成后定制 |
| [references/genop_functionality_index.md](references/genop_functionality_index.md) | genop 功能索引（可选） |

---

## Step 1: 复用现有模式

- **必读参考**：FFN → `ops-transformer/ffn/ffn/op_host/ffn_def.cpp`、`ffn_tiling.cpp`、`examples/test_aclnn_ffn.cpp`；GMM → `grouped_matmul_def.cpp`、`op_kernel/grouped_matmul.h`；MoE → `moe_init_routing_def.cpp`、`examples/test_aclnn_moe_init_routing.cpp`。类型/格式枚举见 [type_format_reference.md](references/type_format_reference.md)，Kernel 写法见 [ascendc_kernel_implement.md](references/ascendc_kernel_implement.md)。
- **行为准则**：先完整复制同类型算子骨架，再做最小必要修改；保持命名、宏（如 ASCEND_IS_AIC）、队列与 UB 管理、AICore 与芯片配置一致。

---

## Step 2: op_host 定义

- **模式**：继承 `OpDef`，在 `namespace ops` 内定义；Input 用 `Input("name")` + `.ParamType(REQUIRED/OPTIONAL)` + `.DataType({...})`、`.Format({...})`、`.UnknownShapeFormat({...})`；Output 同理；属性用 `.Attr("name").AttrType(...).Int/Float/ListInt(...)`；AICore 用 `OpAICoreConfig`（DynamicCompileStaticFlag、DynamicFormatFlag 等）并 `AddConfig("ascend910b", config)`；最后 `OP_ADD(YourOpClassName)`。
- **示例代码**：见 [references/op_host_examples.md](references/op_host_examples.md)（FFN/GMM/MoE 的 Input/Output/Attr 片段）。
- **要点**：新算子时从参考算子完整复制类与构造函数，只改类名、输入输出名与个数、DataType/Format、属性与默认值；无特殊原因不随意改 AICore 与 ExtendCfgInfo；需 aclnn 时沿用 `"aclnnSupport.value", "support_aclnn"`。

---

## Step 3: op_kernel 实现

- **共性**：命名空间与算子一致；包含 `kernel_operator.h`、矩阵类用 `lib/matmul_intf.h`；类型别名与 `MatmulImpl` 等按参考算子；用模板区分 dtype/量化/激活等。
- **骨架代码**：见 [references/op_kernel_skeletons.md](references/op_kernel_skeletons.md)（FFN/GMM/MoE 的 Param 与 Compute 类骨架）。
- **要点**：确认是否仍基于 `MatmulImpl` 及 tiling 字段；只增删 GM 输入、调整 ComputeDequantAndActivate 等业务逻辑；保持队列/UB 分配、PipeBarrier、DataCopyPad、SetAtomicAdd 等模式不变。

---

## Step 4: Tiling / Infershape / 与 JSON 对齐

- 在 `op_host/` 下找 `*_tiling*.h/.cpp`、`*_infershape.cpp`，按参考算子分析 tiling 参数与 shape→tiling 的映射。
- **JSON + types.h 流程**：见 [references/op_host_json_types_flow.md](references/op_host_json_types_flow.md)（从 JSON 提取接口 → op_host DataType/Format 映射 → infershape 对齐 → tiling 校验 → op_kernel 命名一致）。

---

## Step 5: CANN aclnn 示例

- **流程**：Init(aclInit/SetDevice/CreateStream) → 为每个输入/输出 CreateAclTensor → aclnnXxxGetWorkspaceSize → 按需 aclrtMalloc workspace → aclnnXxx(...) → aclrtSynchronizeStream → 拷回并打印 → 销毁张量/释放内存/ResetDevice/aclFinalize。
- **模板与占位**：见 [references/aclnn_example_template.md](references/aclnn_example_template.md)。新示例时复制该模板，替换 include、dtype、张量构造与 aclnnXxx 调用，保持 CHECK_RET 与成对释放。

---

## Step 6: 测试与验证

- 若有 Python 单测：以现有 `test_npu_<op_name>_*.py` 为模板，构造典型与边界 shape，用参考实现或简单算法算期望值，断言 shape/dtype 与数值误差在可接受范围。

---

## 约束与示例

- **约束**：不凭空发明模式，先搜索并对齐相邻算子；改文件前先通读；涉及芯片/动态 shape/确定性时优先与现有算子一致；示例与测试宜小且可手算验证。
- **示例**：用户要求“新增类似 grouped_matmul_finalize_routing 的 GMM 路由算子”时，Agent 应：在 gmm 目录找到并阅读 `grouped_matmul_finalize_routing_*`；复制 `*_def.cpp` 与 op_kernel 主类并改名与调整接口；参照 tiling/infershape 保证 Graph→kernel 映射正确；按 aclnn 模板写示例并补单测。

---

## genop 与通用示例生成

- **genop**：在 ops-transformer 下执行 `bash build.sh --genop=op_class/op_name` 生成新算子目录与占位文件。详见 [references/genop_usage.md](references/genop_usage.md) 与 [references/genop_functionality_index.md](references/genop_functionality_index.md)。
- **通用 aclnn 示例生成**：从 op_host/op_kernel 提取输入输出与属性，按 [references/aclnn_example_template.md](references/aclnn_example_template.md) 填充分支；缺失信息处用 FILL IN 注释并提示用户补全；遵循模板中的最佳实践（先模板、再提取、再定制、再测试）。
