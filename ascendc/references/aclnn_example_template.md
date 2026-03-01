# CANN aclnn 示例模板与用法

生成 `aclnn_*` 示例时：先从 op_host / op_kernel 提取输入输出与属性，再按本模板填充分支；保持 CHECK_RET、成对分配/释放、Init → CreateAclTensor → GetWorkspaceSize → 执行 → 同步 → 打印/清理 的顺序。

## 通用模板结构（main 流程）

```cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_[operator_name].h"
#include <iostream>
#include <vector>

#define Kernel_dtype [appropriate_data_type]
#define Acl_dtpe [corresponding_acl_data_type]
#define CHECK_RET(cond, return_expr) do { if (!(cond)) { return_expr; } } while (0)
#define LOG_PRINT(message, ...) do { printf(message, ##__VA_ARGS__); } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) shapeSize *= i;
  return shapeSize;
}

template <typename T>
void PrintOutResult(std::vector<int64_t> &shape, void **deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<T> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(T),
      *deviceAddr, size * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result failed. ERROR: %d\n", ret); return);
  for (int64_t i = 0; i < size; i++)
    LOG_PRINT("idx[%ld] (offset: %ld Bytes) : %f\n", i, i * sizeof(T), resultData[i]);
}

int Init(int32_t deviceId, aclrtStream *stream) {
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape,
    void **deviceAddr, aclDataType dataType, aclTensor **tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--)
    strides[i] = shape[i + 1] * strides[i + 1];
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
      aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct inputs (from op_host)
  // 3. Construct outputs
  // 4. aclnn[OperatorName]GetWorkspaceSize(..., &workspaceSize, &executor);
  // 5. if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ...);
  // 6. aclnn[OperatorName](workspaceAddr, workspaceSize, executor, stream);
  // 7. aclrtSynchronizeStream(stream);
  // 8. PrintOutResult / copy back
  // 9. aclDestroyTensor(...); aclrtFree(...); aclrtDestroyStream; aclrtResetDevice; aclFinalize();
  return 0;
}
```

## 占位用法示例

```cpp
// Input tensors - FILL IN based on operator definition
aclTensor *input1 = nullptr;
void *input1DeviceAddr = nullptr;
std::vector<int64_t> input1Shape = {1, 1, 1, 1}; // FILL IN actual shape
std::vector<Kernel_dtype> input1HostData(GetShapeSize(input1Shape), 0.0f);
ret = CreateAclTensor(input1HostData, input1Shape, &input1DeviceAddr, Acl_dtpe, &input1);
```

## 生成步骤摘要

1. 从 op_host 读取 `Input("name")` / `Output("name")`、`.DataType({...})`、`.Format({...})`、`.Attr("name")`。
2. 从 op_kernel 确认参数结构与张量个数。
3. 替换模板中的 `[operator_name]`、`Kernel_dtype`/`Acl_dtpe`、输入输出张量构造与 `aclnnXxxGetWorkspaceSize` / `aclnnXxx` 调用。
4. 保持 CHECK_RET、成对释放；缺失信息处用 FILL IN 注释标出。
