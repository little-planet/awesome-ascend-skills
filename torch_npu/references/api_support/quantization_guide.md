# 量化算子适配指引

本文档详细说明如何在 torch_npu 中使用量化相关算子。

---

## 概述

torch_npu 提供了多种量化算子，支持训练后量化（PTQ）和量化感知训练（QAT）场景：

- **量化/反量化**：`npu_quantize`, `npu_anti_quant`
- **量化线性层**：`LinearQuant`（推荐）
- **量化矩阵乘**：`npu_quant_matmul`

---

## 设备兼容性

⚠️ **重要限制**：
- **Ascend910A** 和 **Ascend310P** **不支持** `QuantBatchMatmulV3` 算子
- 推荐在 **Ascend910B** 及以上设备使用量化算子

**检查设备兼容性**：
```python
import torch
import torch_npu

if torch.npu.is_available():
    device_name = torch.npu.get_device_name(0)
    print(f"Device: {device_name}")
    # 910A 和 310P 有部分量化算子限制
    if "910A" in device_name or "310P" in device_name:
        print("Warning: Limited quantization support on this device")
```

---

## API 详细说明

### 1. npu_quantize

将浮点张量**量化**为低精度张量。

#### API 签名

```python
torch_npu.npu_quantize(
    inputs: torch.Tensor,      # [必填] 输入张量（FP32/FP16）
    scales: torch.Tensor,      # [必填] 量化缩放因子
    zero_points: torch.Tensor, # [必填] 量化零点
    dtype: torch.dtype,        # [必填] 目标数据类型
    axis: int,                 # [必填] 量化轴
    div_mode: bool = True      # [选填] 除法模式（True=除，False=乘）
)
```

#### 支持的数据类型

- `torch.quint8`：无符号 8 位整数
- `torch.qint8`：有符号 8 位整数
- `torch.qint32`：有符号 32 位整数

#### 完整示例

```python
import torch
import torch_npu

# 1. 构造输入
batch_size = 5
in_features = 16
out_features = 8

inputs = torch.randn(batch_size, in_features, out_features).npu()

# 2. 构造量化参数（per-channel 量化）
scales = torch.tensor([0.1] * out_features, dtype=torch.float32).npu()
zero_points = torch.tensor([0] * out_features, dtype=torch.int32).npu()

# 3. 执行量化（沿 axis=2 进行 per-channel 量化）
quantized = torch_npu.npu_quantize(
    inputs=inputs,
    scales=scales,
    zero_points=zero_points,
    dtype=torch.quint8,
    axis=2,
    div_mode=True
)

print(f"Input shape: {inputs.shape}")        # [5, 16, 8]
print(f"Output dtype: {quantized.dtype}")    # torch.quint8
```

---

### 2. npu_anti_quant

将低精度张量**反量化**为浮点张量。

#### API 签名

```python
torch_npu.npu_anti_quant(
    x: torch.Tensor,                        # [必填] 输入量化张量
    scale: torch.Tensor,                    # [必填] 反量化缩放因子
    offset: Optional[torch.Tensor] = None,  # [选填] 反量化偏移
    dst_dtype: Optional[torch.dtype] = None,# [选填] 目标数据类型
    src_dtype: Optional[torch.dtype] = None # [选填] 源数据类型
)
```

#### 支持 INT4 反量化

`npu_anti_quant` 支持从 **INT4** 反量化：

```python
from ml_dtypes import int4

# INT4 打包为 INT32（每个 int32 包含 8 个 int4）
input_x = torch.randint(-1, 1, (10, 25), dtype=torch.int32).npu()

# 反量化参数
scale = torch.randn(200, dtype=torch.float32).npu()
offset = torch.randn(200, dtype=torch.float32).npu()

# 反量化：INT4 -> FP16
output = torch_npu.npu_anti_quant(
    input_x,
    scale,
    offset=offset,
    dst_dtype=torch.float16,
    src_dtype=torch.int8  # 或 torch.int4（需要 ml_dtypes）
)

print(f"Output shape: {output.shape}")   # [10, 25]
print(f"Output dtype: {output.dtype}")   # torch.float16
```

---

### 3. LinearQuant（推荐）

**量化线性层模块**，推荐用于替代已弃用的 `LinearA8W8Quant`。

#### API 签名

```python
from torch_npu.contrib.module import LinearQuant

class LinearQuant(nn.Module):
    def __init__(
        self,
        in_features: int,          # [必填] 输入特征数
        out_features: int,         # [必填] 输出特征数
        *,
        bias: bool = True,         # [选填] 是否使用偏置
        offset: bool = False,      # [选填] 是否使用偏移
        pertoken_scale: bool = False, # [选填] 是否使用 per-token 缩放
        output_dtype: Optional[torch.dtype] = None  # [选填] 输出数据类型
    )
```

#### 支持的量化格式

- **A4W4**：INT4 激活 × INT4 权重
- **A8W8**：INT8 激活 × INT8 权重

#### 完整示例（A8W8）

```python
import torch
import torch_npu
from torch_npu.contrib.module import LinearQuant

# 1. 定义量化线性层
in_features = 128
out_features = 256

model = LinearQuant(
    in_features=in_features,
    out_features=out_features,
    bias=False,
    pertoken_scale=False,
    offset=False,
    output_dtype=torch.float16  # 输出为 FP16
)
model = model.npu()

# 2. 构造量化输入（INT8 打包为 INT32）
batch_size = 32
x = torch.randint(-1, 1, (batch_size, in_features), dtype=torch.int32).npu()

# 3. 构造量化权重（INT8 打包为 INT32）
weight = torch.randint(-1, 1, (out_features, in_features), dtype=torch.int32).npu()
model.weight.data = weight

# 4. 构造缩放因子
scale = torch.randn(1, dtype=torch.float32).npu()
model.scale.data = scale

# 5. 前向推理
output = model(x)

print(f"Input shape: {x.shape}")          # [32, 128]
print(f"Output shape: {output.shape}")    # [32, 256]
print(f"Output dtype: {output.dtype}")    # torch.float16
```

---

### 4. npu_quant_matmul

**量化矩阵乘法**算子，底层被 `LinearQuant` 调用。

#### API 签名

```python
torch_npu.npu_quant_matmul(
    linear_quant_input: torch.Tensor,      # [必填] 量化输入
    weight: torch.Tensor,                  # [必填] 量化权重
    scale: torch.Tensor,                   # [必填] 缩放因子
    offset: Optional[torch.Tensor] = None, # [选填] 偏移
    pertoken_scale: Optional[torch.Tensor] = None, # [选填] per-token 缩放
    bias: Optional[torch.Tensor] = None,   # [选填] 偏置
    output_dtype: Optional[torch.dtype] = None     # [选填] 输出数据类型
)
```

#### 完整示例

```python
import torch
import torch_npu

# 1. 构造量化输入和权重（INT32 打包）
batch_size = 16
in_features = 64
out_features = 128

x = torch.randint(-1, 1, (batch_size, in_features), dtype=torch.int32).npu()
weight = torch.randint(-1, 1, (out_features, in_features), dtype=torch.int32).npu()

# 2. 构造缩放因子和偏移
scale = torch.randn(1, dtype=torch.float32).npu()
offset = torch.randn(1, dtype=torch.float32).npu()

# 3. 执行量化矩阵乘
output = torch_npu.npu_quant_matmul(
    linear_quant_input=x,
    weight=weight,
    scale=scale,
    offset=offset,
    output_dtype=torch.float16
)

print(f"Output shape: {output.shape}")    # [16, 128]
print(f"Output dtype: {output.dtype}")    # torch.float16
```

---

## 量化流程示例

### 完整的 PTQ（训练后量化）流程

```python
import torch
import torch_npu
from torch_npu.contrib.module import LinearQuant

class QuantizedModel(torch.nn.Module):
    def __init__(self, original_model):
        super().__init__()
        
        # 1. 替换线性层为量化线性层
        self.quantized_layers = torch.nn.ModuleDict()
        for name, module in original_model.named_modules():
            if isinstance(module, torch.nn.Linear):
                quant_layer = LinearQuant(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    bias=module.bias is not None,
                    output_dtype=torch.float16
                )
                self.quantized_layers[name.replace('.', '_')] = quant_layer
    
    def forward(self, x):
        # 2. 量化输入
        scales = torch.tensor([0.1], dtype=torch.float32).npu()
        zero_points = torch.tensor([0], dtype=torch.int32).npu()
        
        x_quant = torch_npu.npu_quantize(
            x, scales, zero_points,
            dtype=torch.quint8,
            axis=1
        )
        
        # 3. 通过量化层
        for layer in self.quantized_layers.values():
            x_quant = layer(x_quant)
        
        # 4. 反量化输出
        output = torch_npu.npu_anti_quant(
            x_quant,
            scale=scales,
            dst_dtype=torch.float16
        )
        
        return output

# 使用示例
original_model = torch.nn.Sequential(
    torch.nn.Linear(128, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 64)
).npu()

quantized_model = QuantizedModel(original_model).npu()

# 推理
x = torch.randn(32, 128).npu()
output = quantized_model(x)
```

---

## 精度注意事项

### 1. 量化误差来源

量化会引入以下误差：
- **舍入误差**：浮点值 → 整数值
- **截断误差**：超出量化范围的值被截断
- **缩放误差**：scale 和 zero_point 计算不准确

### 2. 精度对比

```python
import torch
import torch_npu
from torch_npu.contrib.module import LinearQuant

def verify_quant_accuracy():
    """验证量化精度"""
    # FP32 参考实现
    fp32_linear = torch.nn.Linear(128, 256).npu()
    x_fp32 = torch.randn(32, 128).npu()
    output_fp32 = fp32_linear(x_fp32)
    
    # INT8 量化实现
    quant_linear = LinearQuant(
        128, 256,
        bias=True,
        output_dtype=torch.float16
    ).npu()
    
    # 量化输入
    scales = torch.tensor([0.1], dtype=torch.float32).npu()
    zero_points = torch.tensor([0], dtype=torch.int32).npu()
    x_quant = torch_npu.npu_quantize(
        x_fp32, scales, zero_points,
        dtype=torch.quint8,
        axis=1
    )
    
    # 量化权重
    w_quant = torch_npu.npu_quantize(
        fp32_linear.weight.data,
        scales, zero_points,
        dtype=torch.qint8,
        axis=0
    )
    quant_linear.weight.data = w_quant
    quant_linear.scale.data = scales
    
    # 量化推理
    output_quant = quant_linear(x_quant)
    
    # 反量化
    output_dequant = torch_npu.npu_anti_quant(
        output_quant,
        scale=scales,
        dst_dtype=torch.float32
    )
    
    # 计算误差
    diff = torch.abs(output_fp32 - output_dequant)
    rel_diff = diff / (torch.abs(output_fp32) + 1e-8)
    
    print(f"Max abs diff: {diff.max().item():.6e}")  # 预期 < 1e-1
    print(f"Max rel diff: {rel_diff.max().item():.6e}")  # 预期 < 5e-2
    
    # 量化允许更大误差（< 5e-2）
    assert (rel_diff < 5e-2).all(), "Quantization accuracy check failed"
```

### 3. 精度判定标准

| 数据类型 | 绝对误差阈值 | 相对误差阈值 |
|---------|------------|------------|
| FP32 → INT8 | < 1e-1 | < 5e-2 |
| FP16 → INT8 | < 1e-1 | < 5e-2 |
| FP32 → INT4 | < 2e-1 | < 1e-1 |

---

## 性能优化建议

### 1. 选择合适的量化格式

| 量化格式 | 精度 | 速度 | 内存 | 推荐场景 |
|---------|------|------|------|---------|
| **A8W8** | 中 | 快 | 低 | 推理（推荐） |
| **A4W4** | 低 | 最快 | 最低 | 边缘设备 |
| **FP16** | 高 | 中 | 中 | 训练/高精度推理 |

### 2. Per-Channel vs Per-Tensor

- **Per-Tensor**：整个张量使用同一个 scale（速度快，精度低）
- **Per-Channel**：每个通道使用不同的 scale（速度慢，精度高）

```python
# Per-Tensor 量化（axis=None）
quantized = torch_npu.npu_quantize(
    x, scale, zero_point,
    dtype=torch.quint8,
    axis=None  # 整个张量使用同一个 scale
)

# Per-Channel 量化（axis=0）
scales = torch.tensor([0.1, 0.2, 0.15], dtype=torch.float32).npu()
quantized = torch_npu.npu_quantize(
    x, scales, zero_points,
    dtype=torch.quint8,
    axis=0  # 每个通道使用不同的 scale
)
```

### 3. 校准数据准备

对于 PTQ，建议使用 **500-1000 个样本**进行校准：

```python
# 收集校准数据
calibration_data = []

def collect_calibration_data(model, dataloader, num_samples=1000):
    model.eval()
    with torch.no_grad():
        for i, (x, _) in enumerate(dataloader):
            if i >= num_samples:
                break
            calibration_data.append(x.npu())
    return calibration_data

# 计算量化参数（scale 和 zero_point）
def compute_quant_params(calibration_data):
    all_data = torch.cat(calibration_data, dim=0)
    
    # 使用 min-max 方法计算 scale
    min_val = all_data.min()
    max_val = all_data.max()
    
    # INT8: qmin=-128, qmax=127
    qmin, qmax = -128, 127
    scale = (max_val - min_val) / (qmax - qmin)
    zero_point = qmin - min_val / scale
    
    return scale, zero_point
```

---

## 常见问题

### 问题 1：设备不支持

**错误信息**：
```
RuntimeError: QuantBatchMatmulV3 not supported on this device
```

**解决**：
- 检查设备：`torch.npu.get_device_name(0)`
- Ascend910A 和 Ascend310P 不支持，使用 FP16 或升级设备

### 问题 2：精度下降 > 10%

**原因**：
- 校准数据不足
- 量化格式选择不当（如使用 INT4）
- scale 计算不准确

**解决**：
1. 增加校准数据量（500-1000 样本）
2. 使用 Per-Channel 量化替代 Per-Tensor
3. 尝试 A8W8 替代 A4W4

### 问题 3：INT4 打包格式

**问题**：INT4 如何打包？

**解决**：
```python
from ml_dtypes import int4

# INT4 打包为 INT32（每个 int32 包含 8 个 int4）
# 例如：[int4_0, int4_1, ..., int4_7] -> int32

# 使用 torch_npu.npu_anti_quant 反量化时指定 src_dtype=torch.int8
output = torch_npu.npu_anti_quant(
    packed_int32,
    scale,
    offset=offset,
    src_dtype=torch.int8
)
```

### 问题 4：LinearA8W8Quant 已弃用

**错误信息**：
```
DeprecationWarning: LinearA8W8Quant is deprecated, use LinearQuant instead
```

**解决**：
```python
# ❌ 已弃用
from torch_npu.contrib.module import LinearA8W8Quant
model = LinearA8W8Quant(128, 256)

# ✅ 推荐
from torch_npu.contrib.module import LinearQuant
model = LinearQuant(128, 256, output_dtype=torch.float16)
```

---

## 已弃用 API

❌ **不要使用以下 API**：

```python
# ❌ 已弃用，使用 LinearQuant 替代
from torch_npu.contrib.module import LinearA8W8Quant
model = LinearA8W8Quant(...)
```

✅ **使用推荐 API**：

```python
# ✅ 推荐
from torch_npu.contrib.module import LinearQuant
model = LinearQuant(..., output_dtype=torch.float16)
```

---

## 完整示例：量化 BERT

```python
import torch
import torch_npu
from torch_npu.contrib.module import LinearQuant

class QuantizedBERT(torch.nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        
        # 替换所有线性层
        self.layers = torch.nn.ModuleList()
        for name, module in bert_model.named_modules():
            if isinstance(module, torch.nn.Linear):
                quant_layer = LinearQuant(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    output_dtype=torch.float16
                )
                self.layers.append(quant_layer)
    
    def quantize_input(self, x):
        """量化输入"""
        scales = torch.tensor([0.1], dtype=torch.float32).npu()
        zero_points = torch.tensor([0], dtype=torch.int32).npu()
        return torch_npu.npu_quantize(
            x, scales, zero_points,
            dtype=torch.quint8,
            axis=1
        )
    
    def dequantize_output(self, x):
        """反量化输出"""
        scales = torch.tensor([0.1], dtype=torch.float32).npu()
        return torch_npu.npu_anti_quant(
            x, scale=scales,
            dst_dtype=torch.float16
        )
    
    def forward(self, x):
        # 量化
        x = self.quantize_input(x)
        
        # 通过量化层
        for layer in self.layers:
            x = layer(x)
        
        # 反量化
        x = self.dequantize_output(x)
        
        return x

# 使用示例
from transformers import BertModel

bert = BertModel.from_pretrained('bert-base-uncased').npu()
quantized_bert = QuantizedBERT(bert).npu()

# 推理
input_ids = torch.randint(0, 30522, (8, 128)).npu()
output = quantized_bert(input_ids.float())
```

---

## 参考文档

- [PyTorch 官方量化文档](https://docs.pytorch.org/ao/stable/api_ref_quantization.html)
- [torch_npu API 列表](https://github.com/Ascend/pytorch/blob/master/docs/api/torch_npu_apis.md)
- [Ascend 量化文档](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/operatordev/tbeaicpudevg/atlasopdev_10_0086.html)
- [源码参考 - test_linear_quant.py](https://github.com/Ascend/pytorch/blob/master/test/contrib/test_linear_quant.py)
- [源码参考 - test_npu_quantize.py](https://github.com/Ascend/pytorch/blob/master/test/onnx/test_wrapper_onnx_ops.py)
