# 2026-09-28 新方法完整 v4 补齐与评测扩容

用户本轮明确要求完整v4，以及本机8卡每卡1个test、单卡3090、deepcad空卡。沿用已授权的4个方法checkpoint和后续固定间隔checkpoint，不按test挑选。

沿用现有53个非CTC/OOD task-dataset cells的固定teacher、batch、split、seed、evaluator和特征配置；补齐CTC、X-ray、Cryo使每checkpoint目标为56个cells。队列task数和cell数分开计数（retrieval同时产生clustering；segmentation包含所有预算/seeds/rotations）。全部有效之前不发布完整v4总分；LC25000仍如协议标记provisional。

本机8张5090已有约18GiB/卡的20TB训练，保留训练，在剩余显存允许时每卡叠加1个正式test。单3090使用空闲cpu15，恢复已有共享盘挂载，不复制checkpoint/data。deepcad目前空闲2/3卡，2用于评测，3恢复Adaptive；项目最多4张distinct GPU。不触碰外来训练。正式batch/分辨率不因资源变化而改变。原子mkdir跨机领任务，状态附带host，禁止跨机解释PID。

CTC复用已有native full20-domain runner与缓存，固定5fold、20domain、B8、50epochs、HoVerNet固定head/linker、官方commit。先完成2D/3D oracle格式检查和输入/source固定，再为每个新checkpoint注册完整任务；完整结果须20domain/5fold和有效TRA/SEG校验，绝不使用count proxy。

OOD复用已有X-ray/Cryo实现与固定recipe，记录共享盘上全部输入选择与hash：X-ray124volumes×8slices=992；Cryo固定4project各20000particle=80000；ID三来源各1000，seed0，70/30训练参考/ID测试，kNN10。这些数量是既有协议固定选择，不临时缩样本。保持B64/BF16/final CLS+patch均值、224crop/256resize、p0.5/p99.5；teacher加载和数据完整性未通过不启动。

CTC/OOD代码及全套Rules复制到独立快照，记录Git状态和全文件SHA256，在所有执行机启动前核验。只复用同一共享盘路径与链接；不修改活跃训练代码，不移动大文件。回归调参另列train-only CV扩展，不能替代固定alpha=1主表。

Adaptive长程从完整16103恢复，单rank DDP、batch128、累积8、global1024。鉴于两次占卡中断导致未保存进度丢失，恢复运行改为每61步保存完整训练状态；正式eval仍每488步。此为容错保存频率调整，不改变loss、样本或优化参数；loader/RNG恢复仍不声称位级连续。
