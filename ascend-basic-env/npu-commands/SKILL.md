---
name: npu-commands
description: Ascend NPU command-line utilities and hardware management. Use for npu-smi usage, device management, monitoring, and basic hardware operations. Covers all npu-smi subcommands including info query, configuration, upgrade, virtualization, and certificate management.
---

# npu-smi Command Reference

Complete guide for managing Huawei Ascend NPU devices using npu-smi.

## Quick Reference

```bash
# List all devices
npu-smi info -l

# Check device health
npu-smi info -t health -i <id>

# View chip details
npu-smi info -t npu -i <id> -c <chip_id>

# Monitor temperature/power/memory
npu-smi info -t temp -i <id> -c <chip_id>
npu-smi info -t power -i <id> -c <chip_id>
npu-smi info -t memory -i <id> -c <chip_id>

# View running processes (not supported on all platforms)
npu-smi info proc -i <id> -c <chip_id>
```

## Prerequisites

- npu-smi tool installed
- Root permissions (most configuration/upgrade commands)
- Runtime user group permissions (some query commands)

## Parameter Reference

| Parameter | Description | How to Get |
|-----------|-------------|------------|
| `id` | Device ID | `npu-smi info -l` |
| `chip_id` | Chip ID | `npu-smi info -m` |
| `vnpu_id` | vNPU ID | Auto-assigned or specified |
| `phy_id` | Physical chip ID | `ls /dev/davinci*` |

---

## 1. Query Commands (npu-smi info)

### List Devices

List all NPU devices in the system.

```bash
npu-smi info -l
```

**Output:**
| Field | Description |
|-------|-------------|
| NPU ID | Device identifier |
| Name | Device name |

---

### Query Device Health

Query the health status of a specific NPU device.

```bash
npu-smi info -t health -i <id>
```

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| id | integer | Yes | Device ID (from `npu-smi info -l`) |

**Output:**
| Field | Description |
|-------|-------------|
| Healthy | Health status (OK/Warning/Error) |

---

### Query Device Details

Query detailed board information including firmware version.

```bash
npu-smi info -t board -i <id>
npu-smi info -t npu -i <id> -c <chip_id>
```

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| id | integer | Yes | Device ID |
| chip_id | integer | Yes (for npu) | Chip ID (from `npu-smi info -m`) |

**Output:**
| Field | Description |
|-------|-------------|
| NPU ID/Chip ID | Device/Chip identifier |
| Name | Device/Chip name |
| Health | Health status |
| Power Usage | Power consumption |
| Temperature | Device temperature |
| Memory Usage | Memory usage |
| AI Core Usage | AI Core utilization |
| Firmware Version | Firmware version (in board output) |
| Software Version | Driver version (in board output) |

---

### List All Chips

Query summary information of all chips.

```bash
npu-smi info -m
```

**Output:**
| Field | Description |
|-------|-------------|
| NPU ID | Device identifier |
| Chip ID | Chip identifier |
| Name | Chip name |
| Health | Health status |

---

### Query Temperature

```bash
npu-smi info -t temp -i <id> -c <chip_id>
```

**Output:** NPU Temperature, AI Core Temperature

---

### Query Power

```bash
npu-smi info -t power -i <id> -c <chip_id>
```

**Output:** Power Usage, Power Limit

---

### Query Memory

```bash
npu-smi info -t memory -i <id> -c <chip_id>
```

**Output:** Memory Usage, Memory Total, Memory Usage Rate

---

### Query Processes

**Note:** This command is not supported on all platforms (e.g., Ascend 910B series).

```bash
npu-smi info proc -i <id> -c <chip_id>
```

**Output:** PID, Process Name, Memory Usage, AI Core Usage

---

### Query ECC Errors

```bash
npu-smi info -t ecc -i <id> -c <chip_id>
```

**Output:** ECC Error Count, ECC Mode

---

### Query Utilization

```bash
npu-smi info -t usages -i <id> -c <chip_id>
```

**Output:** AI Core Usage, Memory Usage, Bandwidth Usage

---

### Query Sensors

```bash
npu-smi info -t sensors -i <id> -c <chip_id>
```

**Output:** Temperature, Voltage, Current sensor data

---

### Query Frequency

```bash
npu-smi info -t freq -i <id> -c <chip_id>
```

**Output:** AI Core Frequency, Memory Frequency

---

### Query P2P Status

```bash
npu-smi info -t p2p -i <id> -c <chip_id>
```

**Output:** P2P Status, P2P Mode

---

### Query PCIe Info

```bash
npu-smi info -t pcie-info -i <id> -c <chip_id>
```

**Output:** PCIe Speed, PCIe Width

---

### Query Product Info

```bash
npu-smi info -t product -i <id> -c <chip_id>
```

**Output:** Product Name, Product Serial Number

---

## 2. Configuration Commands (npu-smi set)

### Set Temperature Threshold

```bash
npu-smi set -t temperature -i <id> -c <chip_id> -d <value>
```

| Parameter | Description |
|-----------|-------------|
| value | Temperature threshold in Celsius |

---

### Set Power Limit

```bash
npu-smi set -t power-limit -i <id> -c <chip_id> -d <value>
```

| Parameter | Description |
|-----------|-------------|
| value | Power limit in Watts (W) |

---

### Set ECC Mode

```bash
npu-smi set -t ecc-mode -i <id> -c <chip_id> -d <value>
```

| Value | Mode |
|-------|------|
| 0 | Disable ECC |
| 1 | Enable ECC |

---

### Set Persistence Mode

```bash
npu-smi set -t persistence-mode -i <id> -d <value>
```

| Value | Mode |
|-------|------|
| 0 | Disable |
| 1 | Enable |

---

### Set Compute Mode

```bash
npu-smi set -t compute-mode -i <id> -c <chip_id> -d <value>
```

| Value | Mode |
|-------|------|
| 0 | Default mode |
| 1 | Exclusive mode |
| 2 | Prohibited mode |

---

### Set Fan Mode

```bash
npu-smi set -t pwm-mode -d <value>
```

| Value | Mode |
|-------|------|
| 0 | Manual mode |
| 1 | Automatic mode (default) |

**Note:** In automatic mode, max fan speed ratio is 39.

---

### Set Fan Speed Ratio

```bash
npu-smi set -t pwm-duty-ratio -d <value>
```

| Parameter | Range | Description |
|-----------|-------|-------------|
| value | [0-100] | Fan speed ratio |

**Note:** Cannot set in automatic mode.

---

### Set MAC Address

```bash
npu-smi set -t mac-addr -i <id> -c <chip_id> -d <mac_id> -s <mac_string>
```

| Parameter | Description |
|-----------|-------------|
| mac_id | Network port: 0=eth0, 1=eth1, 2=eth2, 3=eth3 |
| mac_string | MAC address format: "XX:XX:XX:XX:XX:XX" |

**Note:** System restart required after setting.

---

### Set Power State (Sleep)

```bash
npu-smi set -t power-state -i <id> -c <chip_id> -d <value>
```

| Parameter | Range | Description |
|-----------|-------|-------------|
| value | [200, 604800000] | Sleep time in ms |

---

### Set Boot Medium

```bash
npu-smi set -t boot-select -i <id> -c <chip_id> -d <value>
```

| Value | Medium |
|-------|--------|
| 3 | M.2 SSD |
| 4 | eMMC |

**Note:** Power cycle required. Ensure OS is installed on selected medium.

---

### Set CPU Frequency

```bash
npu-smi set -t cpu-freq-up -i <id> -d <value>
```

| Value | Configuration |
|-------|---------------|
| 0 | CPU 1.9GHz / AICore 800MHz |
| 1 | CPU 1.0GHz / AICore 800MHz |

---

### Set System Log Persistence

```bash
npu-smi set -t sys-log-enable -d <mode>
```

| Value | Mode |
|-------|------|
| 0 | Disable |
| 1 | Enable |

---

### Collect System Logs

```bash
npu-smi set -t sys-log-dump -s <level> -f <path>
```

| Parameter | Description |
|-----------|-------------|
| level | Log level, range: 1~10 |
| path | Log storage path (absolute path must exist) |

---

### Clear Log Configuration

```bash
npu-smi set -t clear-syslog-cfg
```

---

### Set AI CPU Custom Op Security Verification

```bash
npu-smi set -t custom-op-secverify-enable -i <id> -d <value>
```

| Value | Mode |
|-------|------|
| 0 | Disable verification |
| 1 | Enable verification |

---

### Set AI CPU Custom Op Verification Mode

```bash
npu-smi set -t custom-op-secverify-mode -i <id> -d <value>
```

| Value | Mode |
|-------|------|
| 0 | Disable verification |
| 1 | Huawei certificate |
| 2 | Customer certificate |
| 3 | Huawei or customer certificate |
| 4 | Community certificate |
| 5 | Huawei or community certificate |
| 6 | Customer or community certificate |
| 7 | Huawei/customer/community certificate |

---

### Set AI CPU Op Timeout

```bash
npu-smi set -t op-timeout-cfg -i <id> -c <chip_id> -d <value>
```

| Parameter | Range | Description |
|-----------|-------|-------------|
| value | [20, 32] | Timeout value |

---

### Set AI CPU Custom Op Verification Certificate

```bash
npu-smi set -t custom-op-secverify-cert -i <id> -f "<cert_path>"
```

| Parameter | Description |
|-----------|-------------|
| cert_path | Certificate file path(s), multiple paths separated by space |

**Note:** Total certificate content length must not exceed 8192 bytes.

---

### Set P2P Memory Copy Configuration

```bash
# For all chips
npu-smi set -t p2p-mem-cfg -i <id> -d <value>

# For specific chip
npu-smi set -t p2p-mem-cfg -i <id> -c <chip_id> -d <value>
```

| Value | Mode |
|-------|------|
| 0 | Disable |
| 1 | Enable |

---

## 3. Upgrade Commands (npu-smi upgrade)

### Query Upgrade Status

```bash
npu-smi upgrade -q <item> -i <id>
```

| Parameter | Description |
|-----------|-------------|
| item | Firmware type: mcu, bootloader, vrd |
| id | Device ID |

**Output:** Conclusion (PASS/Running), Message

---

### Query Firmware Version

```bash
npu-smi upgrade -b <item> -i <id>
```

**Output:** Version

---

### Upgrade Firmware

```bash
npu-smi upgrade -t <item> -i <id> -f <file_path>
```

| Parameter | Description |
|-----------|-------------|
| item | Firmware type: mcu, bootloader, vrd |
| file_path | Upgrade file path (not needed for vrd) |

**Output:** Validity, transfile, Status, Message

---

### Activate Firmware

```bash
npu-smi upgrade -a <item> -i <id>
```

**Output:** Status, Message

---

### Upgrade Workflow

1. **Query current version**
   ```bash
   npu-smi upgrade -b mcu -i 0
   ```

2. **Upgrade firmware**
   ```bash
   npu-smi upgrade -t mcu -i 0 -f ./Ascend-hdk-xxx-mcu_x.x.x.hpm
   ```

3. **Query upgrade status**
   ```bash
   npu-smi upgrade -q mcu -i 0
   ```

4. **Activate firmware**
   ```bash
   npu-smi upgrade -a mcu -i 0
   ```

5. **Restart device** (MCU requires restart)

---

## 4. Clear Commands (npu-smi clear)

### Clear All Chips ECC Error Count

```bash
npu-smi clear -t ecc-info -i <id>
```

---

### Clear Specific Chip ECC Error Count

```bash
npu-smi clear -t ecc-info -i <id> -c <chip_id>
```

---

### Restore Default Certificate Expiration Threshold

```bash
npu-smi clear -t tls-cert-period -i <id> -c <chip_id>
```

---

## 5. AVI (Virtualization) Commands

### Query AVI Mode

```bash
npu-smi info -t vnpu-mode
```

---

### Query AVI Template Info

```bash
npu-smi info -t template-info
npu-smi info -t template-info -i <id>
```

---

### Query vNPU Info

```bash
npu-smi info -t info-vnpu -i <id> -c <chip_id>
```

---

### Query vNPU Config Recovery Status

```bash
npu-smi info -t vnpu-cfg-recover
```

---

### Set AVI Mode

```bash
npu-smi set -t vnpu-mode -d <mode>
```

| Value | Mode |
|-------|------|
| 0 | Container mode |
| 1 | VM mode |

---

### Create vNPU

```bash
npu-smi set -t create-vnpu -i <id> -c <chip_id> -f <vnpu_config> [-v <vnpu_id>] [-g <vgroup_id>]
```

| Parameter | Description |
|-----------|-------------|
| vnpu_config | AVI template name |
| vnpu_id | vNPU ID (optional, range: [phy_id*16 + 100, phy_id*16 + 115]) |
| vgroup_id | vNPU group ID: 0,1,2,3 (optional) |

---

### Destroy vNPU

```bash
npu-smi set -t destroy-vnpu -i <id> -c <chip_id> -v <vnpu_id>
```

---

### Set vNPU Config Recovery

```bash
npu-smi set -t vnpu-cfg-recover -d <mode>
```

| Value | Mode |
|-------|------|
| 0 | Disable |
| 1 | Enable (default) |

---

## 6. Certificate Management Commands

### Get CSR

```bash
npu-smi info -t tls-csr-get -i <id> -c <chip_id>
```

---

### Import/Update TLS Certificate

```bash
npu-smi set -t tls-cert -i <id> -c <chip_id> -f "<tls_cert> <ca_root_cert> <sub_ca_cert>"
```

---

### View Certificate Info

```bash
npu-smi info -t tls-cert -i <id> -c <chip_id>
```

---

### Set Certificate Expiration Threshold

```bash
npu-smi set -t tls-cert-period -i <id> -c <chip_id> -s <period>
```

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| period | [7, 180] | 90 | Expiration threshold in days |

---

### Read Certificate Expiration Threshold

```bash
npu-smi info -t tls-cert-period -i <id> -c <chip_id>
```

---

### Restore Default Certificate Expiration Threshold

```bash
npu-smi clear -t tls-cert-period -i <id> -c <chip_id>
```

---

### Query Rootkey Status

```bash
npu-smi info -t rootkey -i <id> -c <chip_id>
```

---

## Examples

### Monitor Device Status

```bash
# List devices
npu-smi info -l

# Check device 0 health
npu-smi info -t health -i 0

# View chip 0 temperature and power
npu-smi info -t temp -i 0 -c 0
npu-smi info -t power -i 0 -c 0

# View processes (platform dependent)
npu-smi info proc -i 0 -c 0
```

### Configure Device

```bash
# Set temperature threshold to 85°C
npu-smi set -t temperature -i 0 -c 0 -d 85

# Set power limit to 300W
npu-smi set -t power-limit -i 0 -c 0 -d 300

# Enable ECC mode
npu-smi set -t ecc-mode -i 0 -c 0 -d 1

# Set fan to manual mode and adjust speed
npu-smi set -t pwm-mode -d 0
npu-smi set -t pwm-duty-ratio -d 40
```

### Upgrade Firmware

```bash
# Query current version
npu-smi upgrade -b mcu -i 0

# Upgrade MCU firmware
npu-smi upgrade -t mcu -i 0 -f ./Ascend-hdk-310b-mcu_24.15.19.hpm

# Query upgrade status
npu-smi upgrade -q mcu -i 0

# Activate firmware
npu-smi upgrade -a mcu -i 0
```

### Manage vNPU

```bash
# Query AVI mode
npu-smi info -t vnpu-mode

# Query templates
npu-smi info -t template-info

# Create vNPU
npu-smi set -t create-vnpu -i 0 -c 0 -f vir02 -v 103

# Query vNPU info
npu-smi info -t info-vnpu -i 0 -c 0

# Destroy vNPU
npu-smi set -t destroy-vnpu -i 0 -c 0 -v 103
```

### Manage Certificates

```bash
# Get CSR
npu-smi info -t tls-csr-get -i 0 -c 0

# Import certificate
npu-smi set -t tls-cert -i 0 -c 0 -f "rsa.d2.pem rsa.rca.pem rsa.oca.pem"

# View certificate info
npu-smi info -t tls-cert -i 0 -c 0

# Set expiration threshold
npu-smi set -t tls-cert-period -i 0 -c 0 -s 90
```

---

## Supported Platforms

- Atlas 200I DK A2 Developer Kit
- Atlas 500 A2 Smart Station
- Atlas 200I A2 Acceleration Module (RC/EP scenarios)

## Notes

- Most configuration/upgrade commands require root permissions
- Device ID from `npu-smi info -l`
- Chip ID from `npu-smi info -m`
- Command availability varies by hardware platform
- MAC address and boot medium changes require restart
- MCU firmware requires restart after activation
- VRD requires power cycle (30+ seconds off) to activate
- Concurrent MCU firmware upgrades not supported

## Official Documentation

- **npu-smi Reference**: https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/envdeployment/instg/instg_0045.html
- **hccn_tool**: https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/envdeployment/instg/instg_0052.html
