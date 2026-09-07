# D-scale e8 last 10-shot 补格（2026-09-04）

只补用户标出的分类缺口。不拷 `checkpoint.pth`。full linear 已有则 skip。

## 格子

| dataset | 缺 10-shot | 机 |
|---|---|---|
| blood / tissue / cyclops | 无（已齐） | — |
| nct-crc-he | S+/B/L 全部 D；H+ 0.1M | S+/B/L → nfs；H+ → H100 GPU 2,3 |
| chammi-allen-task1 | B 0.1M、B 1M | nfs |
| chammi-allen-task2 | S+/B/L 全部 D；H+ 0.1M | 同上 |
| bbbc005 / NCT retrieval | 无 10-shot 协议 | 不测；表改为只报 full |

ckpt = D-scale e8 last：823 / 1639 / 4103 / 8199。

## 协议

- frozen classification，**batch 64**，不按卡改 batch
- 抽特征 → k=5/10 × seed 0/1/2 → 立刻删 npz
- skip：已有 `axis=data` last、k=10、seed 0/1/2
- nfs 8×5090：每卡叠 2 路，`FREE_MIN=8000`
- H+ 0.1M 只在 H100 原机；本计划本机不启 H+

## 不拷

不 rsync/scp 任何 `checkpoint.pth`。只写 `kshot.csv` / 改 lattice。
