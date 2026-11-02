# 关于 7B 20TB 继续预训练执行说明的回复

感谢详细核对。以下按您原文第1至第6节顺序回复。

## 1. 代码版本

继续使用README指定的固定commit `1b1c9b9b1f4ad9ba16a7fbfe897083086c01e476`，不需要合并历史分支，也不要求随main自动更新。

针对S3读取和节点本地盘保存的适配请继续保留。方便的话，请同步这三个文件相对固定commit的diff，以及实际启动命令、解析后的训练配置和依赖版本清单，便于我们核对并在后续更新时保留适配。无需提供整个服务器环境，也请勿在材料中包含访问令牌或其他凭据。

交付的 `7b_b200.yaml` 已与指定commit的源配置核对，解析内容一致；实际运行覆盖的batch、LR等参数单独记录。

Python要求是我们README写得不准确：该commit实际要求Python>=3.11，并非仅建议。不过，不要求因此中断已运行的训练，也不要直接升级活跃环境。后续启动/恢复前准备独立Python3.11环境，验证现有checkpoint可恢复后再切换。torch2.9.1+cu128不因Python升级而强制更换。完整评测使用单独交付的环境。

## 2. 训练数据

当前四路数据配比与我们20TB训练使用的配方一致。已核对route2在本机、hxw、原上传清单和HF中的文件名及大小：均为3671个r0、283个r9，没有HF漏传或您这边漏下载的问题。

关于summary的69个编号差异，按本方对生成历史的补充说明：当时处理过程中删除过部分tar并重新生成，旧summary JSON没有同步清理，不能再用其累计数量判断当前数据是否缺失。目前沿用我们20TB训练使用的这份发布数据即可，不需要按旧summary补文件。该历史原因由本方说明，文件清单一致性则已独立核查；不将此描述为重新完成了全量内容去重认证。

初始化继续使用所述官方ViT-7B/16权重，请保留完整SHA256及加载记录。您报告的backbone缺失key为0已收到，随机初始化的head参数保留具体名称即可。

## 3. 训练配置

### 3.1 RGB mean/std

已经验证，与我们20TB L实际训练配置完全一致，无需修改，使用YAML中的完整精度：

```yaml
rgb_mean: [0.43700109438633017, 0.4152422629198214, 0.4177610059272783]
rgb_std:  [0.34138578574532263, 0.34111549859535956, 0.3432925245219299]
```

这是配置一致性核验，不是本轮重新扫描全部像素计算统计；S3存储变化本身不要求更改归一化参数，保持解码与预处理一致即可。

### 3.2 Global batch与LR

GB4096可以采用。我们S+至H+原方案使用GB1024，但接受您目前GB4096、62,464次更新的方案，不要求因此重训。下游按相同FM ID/OOD协议比较，报告实际训练配置及近似等样本访问预算，不能称为同更新次数的严格对照。

请确认6e-5→2e-6是实际生效的LR。如果已手动设置，应保持 `optim.scaling_rule=fixed`，避免再次自动缩放。本仓库sqrt_wrt_1024实现包含额外系数，不能只按名称推算。以解析配置和日志为准，不要求未经核实修改正在运行的LR。

### 3.3 Checkpoint选取

按完整FM ID评测结果选择checkpoint（OOD完整评测单独报告，不参与选点），不固定最终点。按本次安排，使用test成绩选优：先在每个task类内平均，再对task类等权平均。7B、H+、L需要采用相同的汇总规则和可比的评测覆盖；规定的任务、seed、fold及预算未齐时，不以部分总分认定最佳点。

这里还需由我们提供明确的评分配置：每项主指标、方向、归一化、fold/seed/预算的汇总层级以及平局规则。MAE等误差指标与AUROC等比例指标不能直接取原始数值平均；这份配置尚未完成，先保留全部逐项成绩，不让您自行猜测。结果明确标记为test选优结果，不将同一批test分数再称为独立留出集的无选择估计。

## 4. 保存与恢复

测试开始前保留完整DCP、optimizer和对应EMA；保存几个checkpoint就可以启动评测，不必等全部训练结束。您已经完成的单节点、两节点保存与恢复验证已收到，实际64rank恢复记录请另行保留。

评测过程中保留当前最佳完整状态、最新可恢复状态，以及尚未测齐的候选。待候选完整评测、最佳点确定且恢复验证通过后，再清理不再需要的历史点。训练和选点完成后，可以仅保留选定最佳点的完整DCP、optimizer、EMA、配置及评测记录，供后续继续训练；不能只留下推理用EMA。

这更新了此前README“全部永久保留”的交付要求。现有max_to_keep:null可继续避免提前删除，不需要在评测未齐时开启自动清理。不要在保存或上传尚未完成时删除分片。

## 5. 当前状态与数据补充

已收到您截至2026-10-31的进度记录；该记录是历史快照，最终完成以实际更新数为准。

缺少的数据正在补传，下载及状态见 [HF补传目录](https://huggingface.co/datasets/annyorange/4tb/tree/main/dataset_transfer_fm_id_ood_supplement_20261102)；本地上传目录已准备，本轮清单为14个数据文件、约85.91GiB；代码、执行入口及说明通过GitHub提供，不再附两个代码ZIP。远端已完成多少以实际上传记录为准，不把本地准备完成说成已经全部送达。您已经持有且校验一致的数据可跳过。

已有HF目录包含全部CTC、CHAMMI、Multimodal原始包，以及PCam、NCT、RxRx1等数据，不重复上传这部分。PCam/NCT的重封装脚本与路径、hash映射一起提供。历史HF目录中的旧协议和划分不能覆盖本次冻结清单。

当前归类：BCCD、NuCLS、Spheroid属于ID外部下游任务；BBBC041保留未认证新物种的限制。

## 6. 待确认事项逐条回复

### 6.1 验证集与指标

完整评测覆盖FM ID+OOD，不用screening45替代；checkpoint仅按完整FM ID选点，OOD单独报告。选点按本次明确的test成绩、task内平均再task间等权平均执行，具体指标方向和归一化表由我们补齐，详见3.3。任务内部已有的验证集选head/LR规则保持不变。

### 6.2 规模偏离

接受GB4096方案，可以进行相同下游协议的成绩比较，按第3节记录实际配置并核对LR。

### 6.3 RGB mean/std

已核实与20TB L实际配置逐值一致，无需修改，使用第3.1节的完整精度。核查记录见 [RGB配置核对](https://github.com/AnnyOrange/biodino/blob/collaborator/fm-id-ood-20261102/collaborator_handoff/BioDINO_7B/attachments/provenance/rgb_normalization_verification_20261102.json)。

### 6.4 69个route2 shard编号差异

生成处理过程中调整过tar，旧summary未同步清理。沿用当前与我们20TB训练一致的发布数据即可，无需按旧summary补文件，详见第2节。

### 6.5 FILM

提供 `FILM_data.zip`，约1.698GiB，使用已经交付的固定biological-group划分，无需另找公开来源。

### 6.6 RSNA Pneumonia

直接提供FM实际使用的 `rsna_pneumonia_eval_tiff.zip`，20,403张固定输入图像、约39.866GiB；不需要您自行猜测DICOM→uint16的缩放方法。我们尚未重新认证历史转换链，因此不提供未经核实的转换公式。

固定manifest已包含框，运行评测不需要从CSV重新建划分。[`stage_2_train_labels.csv`](https://github.com/AnnyOrange/biodino/blob/collaborator/fm-id-ood-20261102/collaborator_handoff/BioDINO_7B/attachments/provenance/stage_2_train_labels.csv) 已放入GitHub，用于来源核对，SHA256为：

`bb40b7e956e9922a6b275ed4a158197568cf9ab618017d53db6159b1b624bb65`

### 6.7 Spheroid LSFM

原交付仅此项明确留待补入口，本次入口见GitHub的 [`scripts/run_unified_paper_spheroid.py`](https://github.com/AnnyOrange/biodino/blob/collaborator/fm-id-ood-20261102/scripts/run_unified_paper_spheroid.py)。使用既定官方划分和规定五seed。其余已交付任务沿用原定完整FM ID/OOD方案，不是更换整套评测。

### 6.8 评测分工

在您那边权重所在环境执行，我们提供数据、程序、规则及结果核对。执行步骤见 [完整评测README](https://github.com/AnnyOrange/biodino/blob/collaborator/fm-id-ood-20261102/collaborator_handoff/FM_evaluation/README.md)。先用一个已完整导出的EMA跑通smoke，再推进各候选点的完整评测；61个导出点全部登记追踪，不把仅有权重当作评测完成。

每个checkpoint使用独立结果目录及权重SHA256。完整结果标记 `FM_full_ID_OOD`；若另跑screening45，独立标记。当前没有自动连接您S3的本方队列；选点及保留按照第3.3、4节，不根据未齐的部分任务提前清理候选。

### 6.9 历史配方脚本

历史脚本见 [GitHub配方依据](https://github.com/AnnyOrange/biodino/blob/collaborator/fm-id-ood-20261102/collaborator_handoff/BioDINO_7B/attachments/provenance/launch_7b_20tb_hxw_20261015.sh.txt)，仅回应您对配方依据的核对需求，不需要单独再次上传或执行。真正使用的训练配置仍是 `7b_b200.yaml` 及您记录的运行覆盖参数。
