# KINE-EXP-002 · 因果消融（7 天，单卡）

状态：立刻开跑
硬件：RTX 5070 Ti 12GB
基座：KINE-EXP-001 ckpt-step10000

## 为什么现在必须做

10k 步：FUT-1 0.823→0.842，TEMP/MOT/EVT/EMB 下跌。继续只降 loss = 牺牲物理理解。

## 唯一自变量

冻结编码器。预测器前加 Intervention Head。
A 原预测器 / B 随机干预标签 / C 对齐干预标签。

## 通过线

C 的 EVT-1 ≥ 0.58 且高于 A；CAU 区分度比 A 高 ≥ 0.08；FUT 回撤 ≤ 0.03；B 不得显著优于 C。
失败也公开。
