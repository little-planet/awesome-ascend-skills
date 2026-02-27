# Genop Functionality Index

## Overview

The `genop` functionality is a powerful tool provided in the `ops-transformer` project that streamlines the creation of new AscendC operator projects. This document serves as an index to help users understand and utilize this functionality effectively.

## What is genop?

`genop` is a command-line tool that generates the initial directory structure and boilerplate code for new AscendC operators. It uses a template-based approach to ensure consistency with existing operators and reduce the time required to set up a new operator project.

## Key Components

### 1. Command Structure

```bash
bash build.sh --genop=op_class/op_name
```

- **op_class**: The category of the operator (e.g., `gmm`, `moe`, `ffn`)
- **op_name**: The name of the new operator (e.g., `my_custom_op`)

### 2. Generated Structure

When you run the `genop` command, it creates the following directory structure:

```
op_class/
└── op_name/
    ├── CMakeLists.txt        # Build configuration
    ├── op_host/              # Operator host code
    │   ├── op_name_def.cpp   # Operator definition
    │   └── op_name_tiling.cpp # Tiling logic
    ├── op_kernel/            # Kernel implementation
    │   └── op_name.h         # AscendC kernel code
    └── examples/             # Usage examples
        └── test_aclnn_op_name.cpp # CANN API example
```

### 3. Template Files

The `genop` tool uses template files located at:
- `/Users/wlf/Desktop/KernelCode/ops-transformer/scripts/opgen/template/add`

These templates provide the basic structure and boilerplate code for new operators.

## Usage Workflow

1. **Generate the operator project**:
   ```bash
   bash build.sh --genop=gmm/my_custom_gmm_op
   ```

2. **Customize the operator**:
   - Modify `op_host/*_def.cpp` to define inputs, outputs, and attributes
   - Implement kernel logic in `op_kernel/*.h`
   - Update tiling logic in `op_host/*_tiling.cpp`
   - Write examples in `examples/test_aclnn_*.cpp`

3. **Build and test the operator**:
   ```bash
   bash build.sh --pkg --soc=ascend910b --ops=my_custom_gmm_op
   ```

## Benefits

- **Consistency**: Follows the same patterns as existing operators
- **Efficiency**: Reduces setup time from hours to minutes
- **Accuracy**: Minimizes manual errors in project setup
- **Standardization**: Ensures all operators follow the same structure

## Related Files

### Core Files
- `/Users/wlf/Desktop/KernelCode/ops-transformer/build.sh` - Main build script with genop functionality
- `/Users/wlf/Desktop/KernelCode/ops-transformer/scripts/opgen/opgen_standalone.py` - Python script that generates the operator project

### Check Script
- `/Users/wlf/Desktop/KernelCode/awesome-ascend-skills/ascendc/scripts/check_genop.sh` - Script to verify genop functionality

## Examples

### Creating a new GMM operator
```bash
bash build.sh --genop=gmm/my_custom_gmm_op
```

### Creating a new MoE operator
```bash
bash build.sh --genop=moe/my_custom_moe_op
```

### Creating a new FFN operator
```bash
bash build.sh --genop=ffn/my_custom_ffn_op
```

## Troubleshooting

### Common Issues

1. **Template directory not found**:
   - Ensure the template directory exists at `/Users/wlf/Desktop/KernelCode/ops-transformer/scripts/opgen/template/add`

2. **Permission denied**:
   - Make sure you have write permissions in the ops-transformer directory

3. **Invalid operator name**:
   - Operator names can only contain letters, numbers, and underscores

### Error Messages

| Error Message | Possible Cause | Solution |
|--------------|---------------|----------|
| "目标目录已存在" | The operator directory already exists | Choose a different operator name or delete the existing directory |
| "找不到模板目录" | The template directory is missing | Check if the template directory exists |
| "算子类型包含无效字符" | Invalid characters in op_class | Use only letters, numbers, and underscores |

## Best Practices

1. **Use descriptive operator names** that clearly indicate the functionality
2. **Follow existing patterns** for similar operators
3. **Test early** to catch issues before proceeding with complex implementations
4. **Document your operator** with clear comments and documentation

## Conclusion

The `genop` functionality is an essential tool for efficiently creating new AscendC operator projects. By following the patterns and guidelines outlined in this document, you can quickly set up new operators and focus on implementing the core functionality rather than boilerplate code.
