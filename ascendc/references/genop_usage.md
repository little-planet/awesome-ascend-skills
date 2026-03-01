# genop 使用说明

## 命令

在 `ops-transformer` 目录下执行：

```bash
bash build.sh --genop=op_class/op_name
```

- `op_class`：算子类别，如 `gmm`、`moe`、`ffn`
- `op_name`：新算子名，如 `my_custom_op`

示例：

```bash
bash build.sh --genop=gmm/my_custom_gmm_op
```

会在 `gmm/my_custom_gmm_op` 下生成完整目录与占位文件。

## 生成结构

- `op_host/`：算子定义、tiling、infershape
- `op_kernel/`：AscendC 内核
- `examples/`：CANN 示例
- `CMakeLists.txt`：构建配置

关键文件：`op_host/*_def.cpp`、`op_host/*_tiling.cpp`、`op_kernel/*.h`、`examples/test_aclnn_*.cpp`。

## 生成后需做的定制

1. 在 `op_host/*_def.cpp` 中修改输入/输出/属性与 AICore 配置。
2. 在 `op_kernel/*.h` 中实现计算逻辑与数据类型/量化分支。
3. 在 `op_host/*_tiling.cpp` 中调整 tiling 参数与 shape 处理。
4. 在 `examples/` 中编写或调整 test_aclnn_* 用例并验证。

## 优点

- 自动生成目录与占位，减少手写错误。
- 与现有算子风格一致，便于对齐与维护。
