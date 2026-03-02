---
name: torch_npu
description: 华为昇腾 Ascend Extension for PyTorch (torch_npu) 的环境检查、部署与能力指引。在用户使用 @torch_npu、昇腾 NPU、CANN、或需要将 PyTorch 迁移到 NPU 时自动应用。支持：多版本安装指引、Flash Attention 和量化算子 API、精度验证模板。
---

# torch_npu 能力与使用指引

## 何时使用本 Skill

- 用户 **@torch_npu**、提到昇腾 NPU、CANN、Ascend、或 PyTorch 在 NPU 上运行。
- 用户需要 **环境检查/部署**：检查或部署 PyTorch、检查环境是否支持 NPU。
- 用户使用 **@torch_npu_doc** 时：基于本 skill 的 [reference.md](reference.md) 提供项目内中文文档能力说明与操作步骤。
- 用户需要 **NPU 格式转换的代码提示**：书写或补全 `torch_npu.npu_format_cast`、`npu_format_cast_`、`torch_npu.Format`、`get_npu_format` 时，按 §2.1 子能力提供参数与枚举提示。
- 用户需要 **API 支持查询**：查询特定算子在不同 torch-npu 版本中的支持情况。
- 用户需要 **精度验证**：构造单算子验证代码，对比 NPU 与 CPU 的输出精度。

---

## 1. 环境检查与部署

### 1.1 自动检查 PyTorch 与 NPU 环境

在回答或生成脚本时，按需执行或建议用户执行以下检查：

**检查 PyTorch 与 Python 版本是否在配套范围内**（参见 README.zh.md 中的「PyTorch与Python版本配套表」）：

- 支持 PyTorch 1.11.0～2.9.0 等多版本，对应 Python 3.7～3.11（视具体 PyTorch 版本而定）。

**检查环境是否支持 NPU：**

```python
import torch
import torch_npu  # 2.5.1 及以后可不显式 import，仍建议写以便兼容

# 是否可用 NPU、设备数量
if torch.npu.is_available():
    count = torch.npu.device_count()
    # 使用 device='npu' 或 .npu()
else:
    # 未安装 CANN / 未 source set_env.sh / 无 NPU 设备
    pass
```

**检查 CANN 环境变量（安装后验证）：**

- 使用前需执行：`source /usr/local/Ascend/ascend-toolkit/set_env.sh`（路径以实际 CANN 安装为准）。
- 若 `ASCEND_HOME_PATH`、`ASCEND_OPP_PATH` 未设置或路径不存在，torch_npu 会报错并提示执行 `source set_env.sh`。

### 1.2 部署步骤摘要

**详细安装指引**：参见 [installation_guide.md](references/installation/installation_guide.md)
**版本配套表**：参见 [version_compatibility.md](references/installation/version_compatibility.md)

快速安装步骤：
1. **安装 CANN**：按 [CANN 安装指南](https://www.hiascend.com/cann) 安装。
2. **安装 PyTorch**：x86 用 `pip3 install torch==2.7.1+cpu`，aarch64 用 `pip3 install torch==2.7.1`。
3. **安装依赖**：`pip3 install pyyaml setuptools`。
4. **安装 torch_npu**：`pip3 install torch-npu==2.7.1`（版本需配套）。
5. **验证**：`source set_env.sh` 后运行验证代码。
---

## 2. torch_npu 能力目录（简略）

| 类别 | 能力说明 |
|------|----------|
| **设备与内存** | `torch.npu`：设备管理、`device_count`、`current_device`、`set_device`、`synchronize`、`Stream`/`Event`、内存统计与分配（`memory_allocated`、`empty_cache`、`MemPool` 等）。 |
| **张量/存储** | `tensor.npu()`、`tensor.is_npu`、NPU Storage、序列化 `torch.save`/`load` 支持 NPU，DDP/多进程 reductions。 |
| **训练/优化** | `torch.npu.amp` 混合精度、`torch_npu.optim`、FSDP 补丁（`ShardedGradScaler`）、梯度检查点默认 NPU。 |
| **分布式** | `torch_npu.distributed`：HCCL/LCCL 后端、`is_hccl_available`、`reinit_process_group`、RPC、symmetric memory、DTensor 规则。 |
| **扩展 API** | `torch_npu.contrib`：NMS、IoU 系列、ROIAlign、DCN、FusedAttention、自定义模块（如 `DropoutWithByteMask`）等。 |
| **图与编译** | NPU Graph（`npugraphify`）、Dynamo、Inductor、torch.compile 支持 NPU。 |
| **推理/ONNX** | ONNX 导出与 NPU 定制算子封装（如 OneHot、RoiAlign、NMS、FastGelu、MultiHeadAttention 等）。 |
| ** profiling** | `torch_npu.profiler`、MSTX 补丁、性能 dump。 |
| **其他** | HiFloat8Tensor、erase_stream、matmul_checksum、transfer_to_npu（可选）、op_plugin。 |

详细 API 以 [昇腾 Ascend Extension for PyTorch 自定义 API 参考](https://www.hiascend.com/document/detail/zh/Pytorch/720/apiref/torchnpuCustomsapi/context/%E6%A6%82%E8%BF%B0.md) 及项目 `README.zh.md` 为准。

---

## 2.1 子能力：torch_npu.npu_format_cast 代码提示

当用户在代码中书写或询问 `torch_npu.npu_format_cast`、`npu_format_cast_`、NPU 张量格式转换时，应提供以下**代码提示与补全指引**，便于在 IDE 中完成各项提示。

### API 签名与参数

- **`torch_npu.npu_format_cast(tensor, acl_format, customize_dtype=None)`**  
  - `tensor`：NPU 上的 `torch.Tensor`（需先 `.npu()`）。  
  - `acl_format`：目标存储格式，可为 **`int`** 或 **`torch_npu.Format`** 枚举成员。  
  - `customize_dtype`：可选，用于 ONNX 等场景的自定义 dtype。  
  - 返回：新张量（不修改原张量）。

- **`torch_npu.npu_format_cast_(tensor, acl_format)`**  
  - 同上，但为 **in-place** 版本，直接修改 `tensor` 的格式。

- **`torch_npu.get_npu_format(tensor)`**  
  - 返回张量当前 NPU 存储格式（`torch_npu.Format` 或整型）。

### 常用 Format 枚举（torch_npu.Format）

在代码提示中可优先提示以下常用值（来自 `torch_npu.npu._format.Format`）：

| 枚举名 | 值 | 常见用途 |
|--------|----|----------|
| `Format.NCHW` | 0 | 默认 4D 卷积布局 |
| `Format.NHWC` | 1 | 通道在后的 4D 布局 |
| `Format.ND` | 2 | 通用 ND 布局 |
| `Format.NC1HWC0` | 3 | Conv/BatchNorm 等算子常用 |
| `Format.FRACTAL_Z` | 4 | 3D 卷积等 |
| `Format.FRACTAL_NZ` | 29 | 线性/矩阵乘、Attention 权重等 |
| `Format.NDC1HWC0` | 32 | 5D |
| `Format.FRACTAL_Z_3D` | 33 | 3D 卷积 |
| `Format.UNDEFINED` | -1 | 未定义 |

其他可选：`NC1HWC0_C04`(12)、`HWCN`(16)、`NDHWC`(27)、`NCDHW`(30)、`NC`(35)、`NCL`(47)、`FRACTAL_NZ_C0_*`(50–54) 等。

### 代码提示与补全规则

1. **补全第二参数**：当用户输入 `torch_npu.npu_format_cast(x, ` 时，提示 `acl_format` 可选为 `int` 或 `torch_npu.Format.xxx`，并列出常用枚举（如 `Format.NCHW`、`Format.NHWC`、`Format.FRACTAL_NZ`、`Format.NC1HWC0`）。  
2. **补全 Format 枚举**：当用户输入 `torch_npu.Format.` 时，提示上述枚举成员列表。  
3. **配对使用**：若代码中已有 `get_npu_format(t)`，在需要转成相同格式时，可提示 `torch_npu.npu_format_cast(other, torch_npu.get_npu_format(t))`。  
4. **常见场景**：  
   - 线性层权重量子化/迁移到 NPU：`torch_npu.npu_format_cast(weight.npu(), 29)`（FRACTAL_NZ）；  
   - 与参数格式一致的梯度：`torch_npu.npu_format_cast(p.grad, torch_npu.get_npu_format(p))`；  
   - 模块迁移时 BN/Conv 的 NC1HWC0：`torch_npu.npu_format_cast(tensor, 3)` 或 `Format.NC1HWC0`。

### 文档来源说明

详细 API 参考和示例请查阅以下文档：
- **Flash Attention 算子**：[flash_attention_guide.md](references/api_support/flash_attention_guide.md)
- **量化算子**：[quantization_guide.md](references/api_support/quantization_guide.md)
- **完整文档索引**：[reference.md](references/reference.md)
---

## 3. API 支持查询

当用户查询特定 API 在 NPU 上的支持情况时：
- **Flash Attention**：`npu_prompt_flash_attention`（预填充）、`npu_incre_flash_attention`（增量推理）— 仅支持 Ascend910B
- **量化算子**：`npu_quantize`、`npu_anti_quant`、`LinearQuant`（推荐）
- **格式转换**：`npu_format_cast`、`npu_format_cast_`、`get_npu_format`

详细用法和示例参见 [references/api_support/](references/api_support/) 目录下的文档。

---

## 4. 精度验证模板

当用户需要验证算子精度时，使用以下模板：

```python
import torch
import torch_npu

def verify_operator(op_func, inputs_spec, rtol=1e-4, atol=1e-5):
    # 1. 构造输入数据
    inputs_cpu = [torch.randn(**spec) for spec in inputs_spec]
    inputs_npu = [inp.clone().npu() for inp in inputs_cpu]
    
    # 2. CPU 执行
    output_cpu = op_func(*inputs_cpu)
    
    # 3. NPU 执行
    output_npu = op_func(*inputs_npu).cpu()
    
    # 4. 精度对比
    diff = torch.abs(output_cpu - output_npu)
    rel_diff = diff / (torch.abs(output_cpu) + 1e-8)
    passed = (diff < atol).all() and (rel_diff < rtol).all()
    
    print(f"Operator: {op_func.__name__}")
    print(f"  Max abs diff: {diff.max().item():.6e}")
    print(f"  Max rel diff: {rel_diff.max().item():.6e}")
    print(f"  Result: {'✓ PASS' if passed else '✗ FAIL'}")
    return passed
```

**精度标准**：
- FP32: abs diff < 1e-5, relative diff < 1e-4
- FP16: abs diff < 1e-3, relative diff < 1e-2
- BF16: abs diff < 1e-2, relative diff < 5e-2
---

## 5. 参考资源

- **安装指引**：[references/installation/installation_guide.md](references/installation/installation_guide.md)
- **版本配套**：[references/installation/version_compatibility.md](references/installation/version_compatibility.md)
- **Flash Attention**：[references/api_support/flash_attention_guide.md](references/api_support/flash_attention_guide.md)
- **量化算子**：[references/api_support/quantization_guide.md](references/api_support/quantization_guide.md)
- **完整文档索引**：[references/reference.md](references/reference.md)
- **官方文档**：[昇腾社区](https://www.hiascend.com/software/ai-frameworks?framework=pytorch)
- **GitCode 仓库**：https://gitcode.com/Ascend/pytorch
