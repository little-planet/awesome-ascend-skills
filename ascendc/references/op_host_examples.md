# op_host 算子定义示例（Input/Output/Attr）

以下为 FFN、GMM、MoE 三类算子在 op_host 中的 Input/Output/Attr 定义片段，供复制后按需修改。对应源码：`ffn_def.cpp`、`grouped_matmul_def.cpp`、`moe_init_routing_def.cpp`。

## FFN 算子

```cpp
// Input definitions
Input("x")
    .ParamType(REQUIRED)
    .DataType({DT_FLOAT16, DT_BF16, DT_INT8})
    .Format({FORMAT_ND});
Input("weight1")
    .ParamType(REQUIRED)
    .DataType({DT_FLOAT16, DT_BF16, DT_INT8})
    .Format({FORMAT_ND});
Input("weight2")
    .ParamType(REQUIRED)
    .DataType({DT_FLOAT16, DT_BF16, DT_INT8})
    .Format({FORMAT_ND});
Input("bias1")
    .ParamType(OPTIONAL)
    .DataType({DT_FLOAT16, DT_BF16, DT_INT32})
    .Format({FORMAT_ND});
// Output definitions
Output("y")
    .DataType({DT_FLOAT16, DT_BF16, DT_INT8})
    .Format({FORMAT_ND});

// Attribute definitions
Attr("activation").AttrType(OPTIONAL).Int({0}); // 0: GELU, 1: RELU, 2: FASTGELU, 3: SILU, 4: SIGMOID, 5: TANH
Attr("inner_precise").AttrType(OPTIONAL).Int({0}); // 0: BF16, 1: FLOAT32
```

## GMM 算子

```cpp
// Input definitions
Input("x")
    .ParamType(REQUIRED)
    .DataType({DT_FLOAT16, DT_BF16, DT_INT8})
    .Format({FORMAT_ND});
Input("weight")
    .ParamType(REQUIRED)
    .DataType({DT_FLOAT16, DT_BF16, DT_INT8})
    .Format({FORMAT_ND});
Input("bias")
    .ParamType(OPTIONAL)
    .DataType({DT_FLOAT16, DT_BF16, DT_INT32})
    .Format({FORMAT_ND});

// Output definitions
Output("y")
    .DataType({DT_FLOAT16, DT_BF16, DT_INT32, DT_INT8})
    .Format({FORMAT_ND});

// Attribute definitions
Attr("split_item").AttrType(OPTIONAL).ListInt({}); // Grouping information
Attr("dtype").AttrType(OPTIONAL).Int({0}); // 0: FLOAT16, 1: BF16, 2: INT8
Attr("transpose_weight").AttrType(OPTIONAL).Int({0}); // 0: No transpose, 1: Transpose
```

## MoE 算子

```cpp
// Input definitions
Input("x")
    .ParamType(REQUIRED)
    .DataType({DT_FLOAT16, DT_BF16})
    .Format({FORMAT_ND});
Input("rowIdx")
    .ParamType(REQUIRED)
    .DataType({DT_INT32})
    .Format({FORMAT_ND});
Input("expertIdx")
    .ParamType(REQUIRED)
    .DataType({DT_INT32})
    .Format({FORMAT_ND});

// Output definitions
Output("expandedXOut")
    .DataType({DT_FLOAT16, DT_BF16})
    .Format({FORMAT_ND});
Output("expandedRowIdx")
    .DataType({DT_INT32})
    .Format({FORMAT_ND});
Output("expandedExpertIdx")
    .DataType({DT_INT32})
    .Format({FORMAT_ND});

// Attribute definitions
Attr("activeNum").AttrType(OPTIONAL).Int({0}); // Number of active experts
```
