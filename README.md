# Awesome Ascend Skills

A streamlined knowledge base for Huawei Ascend NPU development, structured as AI Agent Skills.

## Skills

| Skill | Description |
|-------|-------------|
| [npu-smi](npu-smi/SKILL.md) | npu-smi device management: queries, configuration, firmware upgrades, virtualization, certificates |
| [hccl-test](hccl-test/SKILL.md) | HCCL collective communication performance testing and benchmarking |

## Installation

### Claude Code

```bash
# Copy skills to Claude Code directory
cp -r npu-smi ~/.claude/skills/
cp -r hccl-test ~/.claude/skills/
```

### OpenCode

```bash
# Project-level (recommended)
mkdir -p .opencode/skills
cp -r npu-smi .opencode/skills/
cp -r hccl-test .opencode/skills/

# User-level
cp -r npu-smi ~/.config/opencode/skills/
cp -r hccl-test ~/.config/opencode/skills/
```

### Codex

```bash
# Project-level
mkdir -p .agents/skills
cp -r npu-smi .agents/skills/
cp -r hccl-test .agents/skills/

# User-level
cp -r npu-smi ~/.agents/skills/
cp -r hccl-test ~/.agents/skills/
```

### Cursor

```bash
# Project-level
mkdir -p .cursor/skills
cp -r npu-smi .cursor/skills/
cp -r hccl-test .cursor/skills/

# User-level
cp -r npu-smi ~/.cursor/skills/
cp -r hccl-test ~/.cursor/skills/
```

## Structure

```
awesome-ascend-skills/
├── npu-smi/
│   ├── SKILL.md                      # Core quick reference
│   ├── references/                   # Detailed documentation
│   │   ├── device-queries.md
│   │   ├── configuration.md
│   │   ├── firmware-upgrade.md
│   │   ├── virtualization.md
│   │   └── certificate-management.md
│   └── scripts/
│       └── npu-health-check.sh
├── hccl-test/
│   ├── SKILL.md                      # HCCL testing guide
│   ├── references/
│   └── scripts/
└── README.md
```

## How Skills Work

Skills use **progressive disclosure** to manage context:

1. **Discovery**: Only `name` + `description` loaded (~100 tokens)
2. **Activation**: Full `SKILL.md` loaded when triggered
3. **On-Demand**: `references/` and `scripts/` loaded as needed

## Official Documentation

- https://www.hiascend.com/document (Huawei Ascend)
- https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/envdeployment/instg/instg_0045.html (npu-smi)

## Contributing

1. Fork the repository
2. Make your changes
3. Ensure SKILL.md has proper frontmatter (name, description)
4. Submit a PR

## License

MIT
