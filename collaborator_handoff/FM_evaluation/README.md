# FM ID＋OOD 端到端评测（GitHub源码交付）

本目录直接从GitHub获取，不再提供额外代码ZIP。训练继续使用原固定commit和合作方S3适配；本分支用于独立评测。源码、固定协议、环境及验收工具在同一仓库，运行时生成独立workspace。Spheroid入口位于仓库根目录scripts/run_unified_paper_spheroid.py。

**本地已检查：18 个入口可导入、CPU 依赖版本、验收工具拒绝坏输入的测试。目标 B200 的容器构建、7B GPU 推理和完整评测尚未运行，必须在接收方完成。**

## 收到哪些文件

1. 此评测分支的完整Git checkout；请从仓库根目录运行下面命令。
2. 新数据：https://huggingface.co/datasets/annyorange/4tb/tree/main/dataset_transfer_fm_id_ood_supplement_20261102 ，14个文件约85.91GiB，状态见UPLOAD_STATUS.json。已有数据：https://huggingface.co/datasets/annyorange/4tb/tree/main/dataset_transfer_formal_v3_remaining_20260920 。复用清单在tools/existing_hf；下载其中24个归档，PCam/NCT按tools/prepare_existing_parquet.py生成两个安装ZIP。当前安装器的40个逻辑输入不等于需要新上传40包。
3. Hugging Face 旧仓库里的三个 ZIP：`deepcad_current_run_datasets_20260608.zip`、`deepcad_missing_datasets_20260616.zip`、`ood.zip`。不依赖尚未完整索引的历史 31 GB tar。
4. 待测的完整 EMA `.pth` 和解析后的训练 YAML。分片 EMA 需先按原训练交付包导出。二者放在同一 weights 目录，本包不含模型权重。

所有数据包可放同一 uploads 目录，保留文件名。原始 RSNA Pneumonia DICOM 包保留供追溯；实际运行使用补充 TIFF 包。

## 机器准备

Linux、NVIDIA 驱动、Docker 和 NVIDIA Container Toolkit；先分配一张 B200。流程串行运行，不启动历史集群调度器。默认容器最多 192 GiB RAM、8 CPU；开始 GPU 验收时要求至少 128 GiB 可用 RAM 和 80 GiB 空闲显存。请为原始包、解压输入、CTC 输入和临时特征预留至少 2 TiB，并继续监测剩余空间；实际空间随缓存而变。

## 1. 启动独立环境

在宿主机执行，替换五个参数。workspace 必须是新的目录；benchmark 可以是新的空目录。

```bash
bash collaborator_handoff/FM_evaluation/start.sh \
  /data/uploads /data/new_7b_eval /data/benchmark /data/weights 0
```

脚本校验GitHub源码与协议文件、生成独立workspace、构建镜像，在容器内安装并校验固定CTC运行依赖后进入shell。CTC依赖从官方GitHub固定commit和PyPI固定hash wheel获取，不把编译库或嵌套Git仓库打包进本仓库。GPU 编号填实际分配的卡。容器把主机自定义路径映射到固定路径，不修改 manifest 的成员、标签或哈希。镜像构建需要网络下载依赖；本机 Docker socket 无权限，因此此 Dockerfile 尚未实机构建验收。

以下命令均在容器中执行。

## 2. 配置模型并安装数据

```bash
python /acceptance/tools/check_env.py
python /acceptance/tools/accept.py configure \
  --checkpoint /weights/teacher_checkpoint.pth \
  --train-config /weights/resolved_train_config.yaml
python /acceptance/tools/install_data.py --archives /uploads --prepare-ctc
python /acceptance/tools/accept.py data --full-hash
```

数据安装会按固定目录解包，补 HPA/RxRx1 协议 CSV、准备完整 CTC 输入，并保留已有文件；随后逐文件 SHA256 检查。发现缺文件或不同内容会失败并列出具体路径，不自动更换数据或重新划分。`data` 不带 `--full-hash` 仅做快速存在性检查，不能解锁评测。

CTC 数据准备会读取约 364 GiB 的原始包并逐文件验证，可能耗时较长。原始包映射 `/uploads` 必须在后续运行中保持可访问。

## 3. 跑真实 ID＋OOD 冒烟评测

```bash
python /acceptance/tools/accept.py run --mode smoke
```

使用指定 7B checkpoint 完整运行 BloodMNIST 分类及 X-ray 来源判别，经过加载、推理、评分、预测文件保存和结果验证。没有缩减它们的数据或改变协议。通过仅代表这两个任务的端到端链路可用，不代表全部 ID/OOD 已完成。

## 4. 跑完整评测矩阵

```bash
python /acceptance/tools/accept.py run --mode full
```

`jobs.json` 固定 55 个数据集键、221 步：212 个评测作业及 9 个分割特征准备作业。检索/聚类、FILM、Spheroid 的一个作业内部包含规定的五个 seed；不能把作业数理解成指标行数。

- ID：49 个数据集键；BCCD、NuCLS、Spheroid 归 ID。
- OOD：6 个数据集键；BBBC041 保留此前未认证新物种的限制。
- 分割完成 E20/E50、五个 seed 和三个 PanNuke fold。
- CTC 完成五个 fold × 五个 seed。
- NuCLS 完成 fold2–5 × 五个 seed，沿用 fold1 选参。
- 分类、回归及检索按原确定性协议；不虚构五 seed。
- 新 checkpoint 生成自己的特征与 head，不附带任何旧模型的成绩或特征缓存。

同一 workspace 绑定一个 checkpoint 及配置 SHA256。已验收作业可以重用；改变 checkpoint 要用新的 workspace。失败立即停止，不能忽略报错继续宣称完成。中断后先检查日志及对应任务目录；无验收收据的已有结果不会被静默接受。

## 5. 交回验收结果

结果保存在宿主机 workspace 下：

```text
outputs/02_eval_runs/unified_paper/acceptance/
  identity.json       # checkpoint 和训练配置 SHA256
  data_report.json    # 必须 data_sha256 / PASS
  smoke_report.json   # 两个真实任务通过
  full_report.json    # 全部 221 步通过后才生成
  jobs/              # 每步命令、退出码、结果 SHA256、预测文件
  logs/              # 每步完整日志
outputs/02_eval_runs/unified_paper/results/biodino_7b/
```

请交回 acceptance 目录和完整逐项结果、预测、样本清单；不要只交汇总截图。`full_report.json` 的 PASS 表示执行与保存产物验收，不表示配对显著性/置信区间统计也已完成。

## 已修复与仍待目标机验证

- 补齐分割的缓存准备与 inventory，使用当前正确 Multimodal 划分。
- CTC 与 BBBC038 的 7B 最后一层适配补为第 39 层（40-block 架构）。
- 加入当前 Spheroid 冻结 slice encoder＋三维中心检测入口和固定清单。
- 补 HPA/RxRx1 协议 CSV、RxRx1 成员索引、CTC 固定清单和评分运行时。
- 本地 CPU 检查不能证明 B200 峰值显存、CUDA 扩展或镜像构建通过；这些由步骤 1–4 的真实运行确认。失败保留日志，不改 batch、seed、fold 或预算来使验收变绿。

## 数据复用准备

下载旧HF目录时保留repo内部层级，然后在宿主机运行：

```bash
python collaborator_handoff/FM_evaluation/tools/prepare_existing_parquet.py \
  --source /data/existing_hf --output /data/uploads
```

这一步逐文件验证48个parquet的SHA256，生成当前安装器需要的PCam/NCT两个ZIP，不改变数据和划分。24个已有原始归档按tools/existing_hf/HF_REUSE_ARCHIVES.csv下载并校验；旧HF目录里的历史划分不要覆盖当前冻结清单。

## 选点与保留状态

按项目当前安排使用test选优，task内再task间等权汇总，结果明确记录为test选优。混合指标的方向、归一化及聚合配置尚未交付，不能直接对MAE、AUROC等原始值平均。本工具输出完整逐项结果，不宣称已经实现统一选点总分。完整评测和恢复核验前保留候选；选定后保留可继续训练的完整DCP、optimizer及EMA，不只保留推理权重。

## 本次GitHub迁移验证范围

代码与协议的来源hash、工作区生成、验收工具单元测试均在本地核查；Docker构建、外网CTC依赖安装及目标B200完整矩阵尚待接收端验证。本地静态核查不能代替GPU验收。
