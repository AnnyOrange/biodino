# 20TB 训练模态覆盖与 OOD 候选审计

日期：2026-10-12  
范围：用户列出的 36 个 OOD 候选；判断相对 HS6-L 20TB route2 预训练图像的成像模态覆盖，不把下游任务名称本身当成 OOD。

## 判定规则

- **模态 OOD**：训练来源中未发现该成像物理/采集方式，或只存在非常弱的相邻采样。显微镜图像任务不同不自动构成模态 OOD。
- **域 OOD**：成像方式已覆盖，但组织、疾病、物种、标本、医院或采集协议不同。域 OOD 与模态 OOD 分开报告。
- **不能判定 benchmark 可用**：本地 payload、标签、分组、split 或任务定义不完整。这不改变它在论文中体现出的潜在模态新颖性。

## 20TB 来源证据

训练 run 配置 `outputs/01_training_runs/HS6_L_robust_biosafe256_gb1024_lr1e4_wu3_tw30_nosig_e61x4098_20tb_route2_r0r9_8x3090qi_20260927/config.yaml` 将训练来源混合为 9% `webds_micro_100k`、21% `deepcad_5t_v1`、70% route2 nested shards。下表是 route2 这一个组件的固定来源池容量，不是按 epoch 的抽样比例；route2 的 `final_selection/composition_acceptance.json` 记录 19.99997 TB 选中量：

| 来源标签 | 字节数 | 约十进制 TB | 证据解释 |
|---|---:|---:|---|
| 明场 | 3,824,695,950,340 | 3.825 | 常规透射明场/brightfield 已覆盖 |
| H&E 病理明场 | 2,750,745,673,728 | 2.751 | 1,748,877 个病理样本；额外来源脚本明确以 TCGA diagnostic SVS 切片 |
| 荧光 | 3,678,751,670,292 | 3.679 | 常规荧光显微已覆盖 |
| 多模态（逐 OID 规则） | 7,999,996,443,136 | 8.000 | 多来源/多标记；不能据此推断其包含任一特定新模态 |
| 光片 | 634,521,632,918 | 0.635 | light-sheet/LSFM 类别明确出现 |
| 电镜 | 529,470,769,776 | 0.529 | route2 将其明确归为 EM 类别；约 160 个源样本 |
| Phase/DIC | 76,876,578,998 | 0.077 | 普通相差/DIC 有少量覆盖；不等于已见数字全息定量相位 |
| 共聚焦 | 59,991,263,236 | 0.060 | 明确受限后的少量共聚焦样本 |
| 超分 | 6,886,824,774 | 0.0069 | 仅约 0.035% 容量，视作极弱 exposure，不视作充分 SIM 预训练 |

TCGA 病理 tiles 的生成脚本 `/mnt/deepcad_nfs/xzj/pathology_he_512/src/pathology_public_download/slice_tcga_he_512.py` 只列出 `TCGA-COAD`、`TCGA-READ`、`TCGA-LUAD`、`TCGA-LUSC` SVS 项目。故有直接证据证明 H&E 明场、TCGA 病理图像已见，但这个 TCGA 子集不含 TCGA-BRCA。route2 在显微来源中还明确标注明场、宽场荧光、光片、扫描透射电镜、相差/DIC 和共聚焦。来源字段未提供逐图完整的公开数据集名称，因此 modality hit 不等于确认与某候选共享相同图像。

另外 30% microscopy mixture 不是空白：`deepcad_5t_v1/mix_spec.json` 记为 30% replay `webds_micro_100k` + 70% 新 4TB microscopy。其原始筛选汇总 `filter_drop_confocal_summary.json` 明确保留 1,849,733 个明场、888,702 个荧光、99,903 个光片、101,952 个 Phase/DIC 和 206,808 个“超分”样本，confocal 被排除；这证明标准光学显微、光片、相位和一定超分辨来源在完整混合中有额外 exposure。`webds_micro_100k` 本身也是微观图像源。该来源审计没有列出乳腺 H&E/TCGA-BRCA family。注意 700k 抽样清单记录的是 source shard/key 而非 dataset 标签，因此不能由家族汇总推出具体候选图像是否原样重叠。

## 逐数据集判断

状态针对**模态覆盖**；括号内另列域变化或当前数据可用性。

### Classification

| 数据集 | 论文/原始数据说明的模态 | 相对 20TB 的判断 | 证据与限制 |
|---|---|---|---|
| FILM | mid-IR photothermal-modulated fluorescence、光谱/高光谱 | **强模态 OOD** | 不是普通荧光；新颖点是中红外吸收调制荧光。27 个独立年龄样本，适合作小样本观察，不适合宣称稳定大样本分类。原文：[Nature Methods 2026](https://www.nature.com/articles/s41592-026-03090-1)。 |
| Brightness_demixing | SMLM 单分子定位与 brightness 混合解混 | **近域/专门采集 OOD** | 与荧光相邻，但定位显微的稀疏单分子事件不是常规宽场/共聚焦帧。缺少独立事件真值，标签若由 brightness 阈值生成会循环。原始 Zenodo：[15862477](https://doi.org/10.5281/zenodo.15862477)。 |
| ColonCancerHSI | 结肠癌高光谱组织成像 | **潜在模态 OOD，当前不可判 benchmark** | 20TB 没有可核验的 HSI 来源标签；本地仅有门户/链接信息，无图像 payload。 |
| RIED | ECL/CL/BL 化学发光反应成像与超分辨重建 | **模态新颖，但不适合该分类任务** | 反应发光不是训练池常规明场/荧光；本地只有三个 acquisition，数据集自身的结构与成像方式混杂，宜做 restoration 方法测试。原文/数据：[Nature 2026](https://www.nature.com/articles/s41586-026-10889-7)、[Zenodo](https://doi.org/10.5281/zenodo.20328840)。 |
| HMI_LUSC | H&E 组织的可见光高光谱显微成像 | **潜在模态 OOD，当前不可判 benchmark** | 原论文使用 450–750 nm、61 bands 的 wavelength-scanning HMI；20TB 3-channel 表示不能证明见过高光谱立方体。本地图像 payload 缺失。原文：[Scientific Data](https://www.nature.com/articles/s41597-026-06766-7)。 |
| Altair-LSFM | light-sheet fluorescence microscopy，3D | **模态 ID** | 20TB route2 已含约0.635 TB 光片图像；数据集主要是 beads/cell demo，因此物种/结构和三维重建协议仍有域差异，但不是新模态。原文：[eLife](https://elifesciences.org/articles/106910)、[Zenodo](https://doi.org/10.5281/zenodo.18049380)。原数据不构成独立生物分类 benchmark。 |
| SMLDM | 单分子扩散轨迹/合成 Brownian trajectories | **不属于 DINO 图像模态 OOD benchmark** | 主体是轨迹，不是原生显微图像分类；转换为 raster 会另造 proxy。原始数据：[Zenodo](https://doi.org/10.5281/zenodo.15233247)。 |
| DHM_Erythrocyte | 数字全息显微（DHM）原始 hologram + 重建 phase | **强模态/表示 OOD** | 训练池有少量普通相差/DIC，但未发现离轴数字全息与 hologram-to-phase 重建。原始配对数据论文：[Data in Brief 2024](https://doi.org/10.1016/j.dib.2024.110424)。本地 RBC 数据的来源标签、组别和划分仍待核验。 |
| Microscopic_Hyperspectral_Choledoch | 胆管组织的显微 HSI 与 RGB | **潜在模态 OOD，当前不可判 benchmark** | 若使用 HSI cube 是新采集/光谱模态；RGB companion 是近域病理明场。45.4 GB archive 尚未解压，需按原论文确认数据和分组。 |

### Segmentation

| 数据集 | 论文/原始数据说明的模态 | 相对 20TB 的判断 | 证据与限制 |
|---|---|---|---|
| BBBC039 | U2OS Hoechst 荧光细胞核 | **模态 ID** | 常规荧光；另有细胞系/药筛域变化。官方：[BBBC image sets](https://bbbc.broadinstitute.org/image_sets)。 |
| CREMI | serial-section EM | **模态 ID** | 20TB 已有约0.529 TB EM 家族数据；具体 EM acquisition/神经突触任务是明显域差异。原始挑战：[CREMI](https://cremi.janelia.org/)。 |
| SIIM_ACR_Pneumothorax | 胸部 X-ray radiograph | **强模态 OOD** | route2 影像来源是显微/数字病理，未见放射线源。数据/任务：[SIIM-ACR challenge](https://www.kaggle.com/c/siim-acr-pneumothorax)。 |
| Annotated_QPI_Cell | quantitative phase microscopy / QPI | **方法级近域 OOD** | 普通 Phase/DIC 有约0.077 TB，但定量相位（无染料、物理相位量）不是普通相差。当前本地 segmentation 路径缺失。原始数据：[Zenodo 5153251](https://doi.org/10.5281/zenodo.5153251)。 |
| CoNSeP | H&E 结直肠癌病理切片 | **模态 ID，域近邻** | H&E/明场训练域已大量出现；癌种、核类别和数据集来源仍是 held-out pathology domain。原论文 HoVer-Net：[arXiv](https://arxiv.org/abs/1812.06499)。 |
| CRAG | H&E 结直肠腺癌 WSI tiles | **模态 ID，域近邻** | 同 H&E 病理成像；组织/医院/腺体分割任务为域差异。原数据项目/论文：[CRAG](https://github.com/XiaoyuZHK/CRAG-Dataset_Aug_ToCOCO)。 |
| EPFL_Hippocampus_EM | hippocampus serial EM | **模态 ID** | 20TB 有 EM 曝光；神经组织和高分辨 serial section 仍属专门域。原始来源：[EPFL technical report](https://infoscience.epfl.ch/record/161954)。 |
| BBBC006 | U2OS 多焦平面/离焦荧光图像 | **模态 ID，采焦分布近域** | 荧光已见；离焦/z-stack 形成清晰的成像质量 shift。官方：[BBBC006](https://bbbc.broadinstitute.org/BBBC006)。 |
| BBBC007 | Drosophila Kc167 荧光细胞 | **模态 ID** | 常规荧光；物种/细胞系和标注域变化。官方：[BBBC007](https://bbbc.broadinstitute.org/BBBC007)。 |
| BBBC008 | HT29 细胞多通道荧光 | **模态 ID** | 常规荧光及多通道标记；非新成像物理。官方：[BBBC008](https://bbbc.broadinstitute.org/BBBC008)。 |
| BlastoSPIM | mouse embryo 3D light-sheet fluorescence | **模态 ID** | route2 已含约0.635 TB 光片模式；胚胎、3D体数据和核实例任务构成强域/任务 shift。原文/数据：[Development 2024](https://journals.biologists.com/dev/article/151/21/dev202817/362603/Nuclear-instance-segmentation-and-tracking-for)、[项目页](https://blastospim.flatironinstitute.org/html/)。 |
| ISBI2012_EM | serial-section transmission EM | **模态 ID** | 训练池有扫描透射 EM 标签；神经突触 segmentation 域不同。原挑战/数据：[ISBI 2012 EM](https://web.archive.org/web/20160406014920/http://brainiac2.mit.edu/isbi_challenge/)。 |
| GlaS_MICCAI2015 | H&E colon gland histology | **模态 ID，域近邻** | H&E 病理模态已见；腺体/结肠域变化。原挑战：[GlaS MICCAI 2015](https://warwick.ac.uk/fac/cross_fac/tia/data/glascontest/)。 |
| SNEMI3D | serial-section EM | **模态 ID** | EM 已有直接覆盖；3D connectomics 与标注任务仍是域差异。原挑战：[SNEMI3D](https://snemi3d.grand-challenge.org/)。 |
| Yeast_in_Microstructures | time-lapse yeast cell microscopy in microfluidic traps | **模态近邻/ID** | 原论文定位为显微图像分割；route2 明场/相差类来源覆盖相邻成像，微流控结构、酵母和长时序是新域。原文/代码：[EMBC 2023 dataset](https://github.com/ChristophReich1996/Yeast-in-Microstructures-Dataset)。 |

### Regression

| 数据集 | 论文/原始数据说明的模态 | 相对 20TB 的判断 | 证据与限制 |
|---|---|---|---|
| BioSR | structured-illumination fluorescence microscopy (SIM), LR/HR reconstruction pairs | **近域；SIM-specific 覆盖证据不足** | 20TB route2 里“超分”类约6.9 GB，另一个 microscopy 预训练源有206,808个泛称超分辨样本；故它不是从未见过超分辨显微，但本地训练清单未证明这些样本具体是 SIM。常规荧光已见，SIM 条纹/频域重建可作为 acquisition-method shift。原数据：[DL-SR/BioSR](https://github.com/qc17-THU/DL-SR)。 |
| BBBC005 | simulated fluorescence cell images | **ID，且已在 formal v3** | 仿真荧光模态熟悉，当前 OOD 回归目录是重复副本，不新增 OOD cell。官方：[BBBC005](https://bbbc.broadinstitute.org/BBBC005)。 |
| IDCIA | marker-based fluorescence microscopy cell images | **模态 ID，域近邻** | 训练含宽场/共聚焦荧光；marker、条件、成像日期不同。论文定义了 condition shift，但缺 donor/specimen ID，协议需保持 proposed。原文：[arXiv 2411.08992](https://arxiv.org/html/2411.08992v1)。 |
| CellFMCount | DAPI fluorescence microscopy cell counting | **模态 ID** | 常规核荧光属于明确已见模式；样本来源/计数冲突是可执行性 blocker。原文：[arXiv 2511.19351](https://arxiv.org/html/2511.19351v1)。 |
| FPM_INR | Fourier ptychographic microscopy, multi-angle intensity reconstruction | **强方法级模态 OOD** | 训练源没有明确 FPM/LED-array Fourier reconstruction 标签；普通 brightfield 标签不能当作 FPM 已见。当前只有2个确认 paired GT，native reconstruction benchmark 暂不可执行。原方法：[FPM-INR](https://github.com/hwzhou2020/FPM_INR)。 |
| RSNA_Bone_Age | pediatric hand radiography / X-ray | **强模态 OOD** | 医用 radiography 完全不同于 20TB 显微图像来源。原挑战：[Radiology 2018](https://doi.org/10.1148/radiol.2018180736)。 |
| VGG_Cell_Counting | synthetic rendered cell-count images | **不能据现有文件判定** | 本地 raw/extracted 当前为空；没有可用图像供模态和任务审计。7 月 scan 已过期。 |

### Detection

| 数据集 | 论文/原始数据说明的模态 | 相对 20TB 的判断 | 证据与限制 |
|---|---|---|---|
| RSNA_Pneumonia | chest X-ray radiographs | **强模态 OOD** | 20TB 训练来源中未见放射影像；task 与 image domain 都不同。原挑战：[RSNA](https://www.rsna.org/artificial-intelligence/ai-image-challenge)。 |
| NuCLS | TCGA breast cancer H&E whole-slide histology, 40× | **模态 ID；组织/癌种域 OOD** | 原论文：[NuCLS](https://arxiv.org/abs/2102.09099) 说明为 TCGA 乳腺癌 H&E。训练池确有2.751 TB TCGA H&E，但本地 tile 脚本只取 `TCGA-COAD/READ/LUAD/LUSC`，不含 BRCA。因此不能称 NuCLS 为模态 OOD；它可作为 breast pathology domain-held-out benchmark。是否有精确 slide overlap 仍须拿逐 slide manifest 比对。 |
| BCCD | brightfield peripheral blood smear | **ID/近域基线** | 20TB 明场约3.825 TB，且有 blood/brightfield microscopy 的来源域；不应作为新显微模态 OOD。血涂片标本/染色和 hematology domain 可作独立域 shift。原始发布：[BCCD GitHub](https://github.com/Shenggan/BCCD_Dataset)。 |
| BBBC041 | brightfield Giemsa-stained malaria blood smears | **ID/近域基线** | 与 BCCD 同属明场血涂片，模态已见；疟原虫感染阶段、染色和机构来源可作 disease/domain shift。官方原始数据说明：[BBBC041](https://bbbc.broadinstitute.org/BBBC041)。 |
| Spheroid_LSFM | 3D light-sheet fluorescence microscopy, spheroid nuclei | **模态 ID；样本/任务域 OOD** | 训练池含光片模式约0.635 TB；球状肿瘤细胞、3D centroid detection 仍是新域和原生检测任务。原文：[PAC-MAP](https://doi.org/10.1016/j.compbiomed.2024.109561)、[Zenodo](https://zenodo.org/records/11636385)。 |

## 建议的 OOD 结论

1. **可保留为强成像模态 OOD 代表**：RSNA_Pneumonia、RSNA_Bone_Age（X-ray）；FPM_INR（Fourier ptychography）；FILM（mid-IR photothermal fluorescence）；DHM_Erythrocyte（数字全息）。后两者需先完成数据来源/组别审计，FPM 还需补足 paired GT。
2. **高光谱候选有科学新颖性但现在不可作 benchmark 结论**：ColonCancerHSI、HMI_LUSC、Microscopic_Hyperspectral_Choledoch 的潜在 HSI 与 RGB companion 必须分开；当前本地 payload 不齐。
3. **应从“模态 OOD”降为 ID/近域或域 OOD**：NuCLS、BCCD、BBBC041、CREMI/EPFL/ISBI/SNEMI EM、Altair-LSFM、BlastoSPIM、Spheroid_LSFM、CoNSeP/CRAG/GlaS、BBBC 荧光系列。新鲜任务或新疾病标签可以形成 domain/task generalization，不足以声称新模态。
4. **当前不适合监督 DINO 分类 OOD 的目录**：RIED（成像方式与样本类别混杂且 acquisition 太少）、SMLDM（以轨迹而非图像为主）、Brightness_demixing（无独立 event truth）。

以上结论是模态与数据来源审计，不是模型性能结论。正式称为“相对 20TB 的 OOD”时，报告需分别写 `modality-OOD`、`domain-OOD`、`task-OOD`，并对有可能共享原图的病理数据执行 slide/source ID 交集检查。
