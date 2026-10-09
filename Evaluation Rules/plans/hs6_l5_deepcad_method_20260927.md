# 2026-09-27 deepcad：继续训练与新的方法验证

用户本轮明确要求：在 deepcad 空卡继续 method 探索、进一步训练可行方法。沿用既有授权，不重新从零训练 Vanilla/Gram。此次计划在执行前记录；最多使用4张distinct GPU，不占外部用户的1/6/7卡，也给旧GPU4评测worker留位。初始空闲0/2/3/4/5均3MiB，选择0/2用于续训、3/5用于新方法。所有checkpoint/data直接读取共享盘，不跨机复制。

## 轨道A：首版Adaptive的长程延续

从首版adaptive_formal/ckpt/15127/checkpoint.pth完整恢复model、optimizer、EMA、恢复器统计和gate，历史anchor仍12687。独立输出，不覆盖旧结果。预定训练到20007，保存15615/16103/16591/17079/17567/18055/18543/19031/19519/20007；不按test分数挑选checkpoint。DDP BF16，有效batch1024；先测每卡batch256是否可用，OOM后仅回退128，再64；batch调节是训练资源选择而非测试协议变更。若改变微批大小，明确报告恢复器每微步更新频率变化，不能称位级连续轨迹。保留原代码行为，优先每卡128的已验证布局。

训练源：/mnt/huawei_deepcad/dinov3_adaptive_continue_snapshot_20260927。默认2卡×128×累积4。最多100个全状态保存，保证本轮不触发自动删除。每30秒写资源遥测，发现选定GPU有外来PID则仅停止本次拥有的子进程。训练前和退出写环境/代码hash；失败保留原输出。

## 轨道B：从M=20007接入双参照约束的短验证

原假设缺口：一般线性可恢复不保证当前特征的余弦/局部差分关系；首版检测和RxRx3退步。已失败的rank4误差谱/Fisher方向选择不重启、不调网格。

候选方案：保留E=12687的Adaptive恢复约束，同时使用冻结M=20007约束归一化全局特征和局部邻域差分的可恢复性。M恢复映射严格为分块正交映射，允许坐标旋转但不允许任意拉伸。它不计算token两两Gram矩阵；固定原输出宽度、单backbone推理。

理论动机：Q正交且单位特征恢复残差为epsilon_i，则两样本余弦差的绝对值不超过epsilon_i+epsilon_j；历史邻居margin超过相应误差和时排序可保留。对局部差分用相同条件分析，但这不是检测AP不退化定理。先做数值性质检查、已有TRAIN特征bank的分组验证、有限DDP工程pilot，再投入488步；未通过不启动后续长程。

固定pilot参数和判据在METHOD_DESIGN.md中锁定，测试集不参与选择权重或checkpoint。后续mid正式训练保持新optimizer这一混杂显式可见；复用旧Vanilla/Gram轨迹作为参考，不称完全匹配reset对照。新方法训练不使用未来L=29279 teacher。

## 评测

仍采用union-v4（25分类、4回归、7检索、7聚类、7分割、3检测代理、1CTC、2OOD）。只运行READY组件，未准入保留BLOCKED；full_v4_aggregate_allowed=false。分类严格均值排除provisional LC25000；dense native-final/E20,E50/3seeds，PanNuke3rotations；冻结batch64、dense32、检测8不因设备改变。回归train-only CV扩展单列。不给未完成结果填零，不把patch-F1写作AP。

本轮先启动训练。评测在训练阶段释放的同一组GPU上排队，每卡至少5个已准入test，在测得的GPU/主存峰值允许时继续补位到70%以上；不为凑占用改变协议、不与训练争抢显存。若5个任务的已知峰值不安全，记录资源限制并使用可安全组合，不伪造占用。新snapshot全source/Rules哈希，环境版本，teacher key、checkpoint/config/split hashes均登记后才执行test。

输出：outputs/01_training_runs/hs6_l5_deepcad_method_20260927；研究记录和机制验证：outputs/00_reports/deepcad_method_20260927；评测根：outputs/02_eval_runs/hs6_l5_deepcad_method_v4_20260927。


执行修订：外来用户先后使用0、3号卡，最终选择A=2/4两卡、B=5单卡DDP（累积8，仍global1024）。继续遵守最多4卡，目前实际3卡。A恢复器在独立v2快照修复canonical/DDP key映射，原短尝试显式invalid。有效A输出adaptive_continue_v2_relocated。细节与理论/固定参数见outputs/00_reports/deepcad_method_20260927/METHOD_DESIGN.md。
