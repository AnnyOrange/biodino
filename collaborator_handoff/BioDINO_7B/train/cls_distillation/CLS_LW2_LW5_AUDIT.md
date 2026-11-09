# CLS 与历史 lw2/lw5 的核对

2026-11-09。结论：**复用相同的 CLS ridge recoverability 内核；当前四组配置不是原 lw2/lw5 配方的原样 CLS-only 版本。** 本次没有修改实验权重或教师组合。

## 源码与历史配置

- 旧路径 `dinov3/train/external_distill.py::ExternalDistillation` 的 CLS 取第一张 global crop 的 `student_cls[0]` 与对应教师 CLS，调用 `ExternalRecoveryStream`。
- 新路径 `dinov3/train/global_teacher_distill.py` 直接导入同一个 `ExternalRecoveryStream`，使用相同的第一张 global crop CLS。
- 学生为 backbone 最终 `x_norm_clstoken`，即进入 DINO head 之前的 CLS。教师同样取 backbone CLS；没有改为分类概率、DINO head bottleneck 或 patch mean。
- 内核为带 bias 的 ridge decoder、运行 covariance/cross-covariance、教师维度方差加权 MSE；目标 detach，梯度只流向学生。ridge 0.1，校准 32 次 forward，每 8 次求解。
- 旧实现曾随工作树变动；此次对照是当前保留的旧 `ExternalDistillation` 路径与实际历史配置，不宣称已经恢复并逐字节核验当年每一次运行的全部源码快照。

核对配置：

1. `outputs/01_training_runs/cooldown_20261013/20tb_distill_cooldown_len4880/config.yaml`：`loss_weight=2`、`cls_weight=1`、`patch_weight=1`、`momentum=0.995`。该运行的 `launch_manifest.json` 指向从 no-GRAM74663 完整状态分叉的 lw2。
2. `outputs/01_training_runs/cooldown_20261013/20tb_distill_cooldown_len4880_from47823_lw5/config.yaml`：`loss_weight=5`、`cls_weight=1`、`patch_weight=1`、`momentum=0.98`。

## 确实存在的配置差异

| 项 | 上述历史配置/旧路径 | 当前四组 CLS 草案 |
|---|---|---|
| 主体 | CLS + 空间 patch 蒸馏 | 用户要求的 CLS-only |
| 多教师合并 | 各教师 weight=1，直接求和 | 权重归一化，双教师各 0.5 |
| 总辅助系数 | lw2=2，lw5=5 | 1 |
| 额外 loss 渐入 | 内核校准后直接使用配置系数 | 内核校准之外，新增 256 optimizer update ramp |
| 统计 momentum | 所查 lw2=0.995，lw5=0.98 | 0.98 |
| 教师/学生规模 | L 学生、DINOv3-L + H-optimus-0 | 7B 学生与用户指定的四组教师 |
| 教师加载 | 旧 loader | 新的严格加载、原生归一化、BF16 存储与分块推理 |

旧 CLS 部分为 `lambda * sum_t(w_t * L_cls_t)`；当前为 `1 * ramp * sum_t(normalized_w_t * L_cls_t)`。因此“同一内核”不表示同一整体梯度。特别是在相同两个教师输入、相同统计等条件下，ramp 达到 1 后，新 CLS 系数相对 lw2 是 1/4、相对 lw5 是 1/10；实际更换教师后不能将它视为实测梯度比值。

教师取平均、额外 ramp 和总权重 1 是新草案增加的设置；不能将当前草案称为已经严格复现了 lw2/lw5。若实验意图是只更换教师/学生并删掉 patch 项，应明确保留旧求和方式、选择对应的 lambda 与统计 momentum，并取消新增 ramp。

## 本次 CPU 数值检查

使用 `/home/lxy/miniconda3/envs/dinov3/bin/python`，torch `2.10.0+cu128`；显式隐藏 CUDA，CPU 单线程。默认 Python 无 torch，已改用现有环境，没有安装或升级环境。

调用真实旧/new forward 与 `ExternalRecoveryStream`，用合成教师输出替代大模型加载；手动构造 wrapper 状态，不覆盖真实教师 constructor/manifest 加载。旧 patch 系数设为 0，从而隔离 CLS。

| momentum | 教师数 | forward 次数 | loss 最大绝对差 | CLS 梯度最大绝对差 | CLS 统计 buffers 最大绝对差 |
|---|---:|---:|---:|---:|---:|
| 0.995 | 1 | 41 | 0 | 0 | 0 |
| 0.995 | 2 | 41 | 0 | 0 | 0 |
| 0.98 | 1 | 41 | 0 | 0 | 0 |
| 0.98 | 2 | 41 | 0 | 0 | 0 |

对照时显式补偿新旧教师求和/平均和 ramp 的数学系数；前 40 次禁用额外 ramp，第 41 次检查 41/256 的 ramp。覆盖校准前后和多次 ridge 求解。上述结果说明：**相同输入与状态下，去除明确的外层系数差异，新 CLS 内核和旧路径一致。**

没有测试实际 7B/H-optimus/GenBio 教师数值一致性、FSDP/all-reduce、多机 DCP 恢复、GPU 峰值或效果。CPU 检查不能替代真实训练验证。
