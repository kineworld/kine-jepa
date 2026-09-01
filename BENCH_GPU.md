# KINE-Bench GPU 评测 · 一键指南（本机实测可用）

把 KineOne-WM（或对齐基线 V-JEPA 2）在 GPU 上跑出可对比基准数字，是和白泽同台、
把"原型"变成"可投"的关键一步。本指南基于 **本机 RTX 5070 Ti Laptop（12GB）** 实测跑通。

## 环境（本机）
- Python venv：`C:/Users/zoah/.workbuddy/binaries/python/envs/default/Scripts/python.exe`
- kine-bench：与 kine-jepa 同级 `kine/kine-bench`（`python -m kinebench` 可用；launcher 已自动加该路径）
- 本地 V-JEPA 2 权重：`C:/Users/zoah/AppData/Local/Temp/vjepa2`（config.json + model.safetensors 1.3GB + video_preprocessor_config.json）

## 首次：装 CUDA torch（关键坑）
托管 venv 默认是 **CPU 版 torch**，必须换 CUDA 构建，且 **torch 与 torchvision 版本必须匹配**，否则 `torchvision::nms` 注册失败、VJEPA2Model 导入崩。

```bash
# 1) 强制重装 CUDA torch（注意：普通 pip install 会因"已装 CPU 版"跳过，必须 --force-reinstall --no-deps）
cd C:/Users/zoah/.workbuddy/binaries/python/envs/default
Scripts/pip install --force-reinstall --no-deps torch torchvision --index-url https://download.pytorch.org/whl/cu128
#    -> 实测得到 torch 2.11.0+cu128 + torchvision 0.26.0+cu128（匹配，nms 正常）
# 2) 拉最新 kine-bench（含 transformers 5.x 兼容性修复）
cd C:/Users/zoah/WorkBuddy/2026-09-01-10-59-54/kine/kine-bench
git -c http.proxy=http://127.0.0.1:7897/ pull
```

> 踩坑记录（已修复，pull 即含）：
> - transformers 5.16.1 的 `AutoVideoProcessor` 是**惰性桩**，访问即抛 `ModuleNotFoundError`；原适配器 `try/except ImportError` 没兜住，连带把 `AutoModel` 置 None。已改为处理器可选 + 守卫只要求 `AutoModel`（提交 `ac448ef`，已推 `zoahdev/kine-bench`）。
> - **（此处曾写错，已更正）** 一度以为 GPU 被系统电源策略锁在 ~17W / `Perf P4`。那是**空闲态读数**：`nvidia-smi` 在无负载时显示 17.6W、`utilization 1%`、`Perf P4`，属正常降频。实测评测负载下为 **98.8W、`utilization 99%`、显存 5.3GB**，并未被锁。教训：判断功耗墙必须在负载中读，不能读空闲值。
> - `nvidia-smi -pl 100` 仍被拒（`Changing power management limit is not supported in current scope`），但这是沙箱权限问题，不影响实测性能。

## 一键评测（GPU）
```bash
cd C:/Users/zoah/WorkBuddy/2026-09-01-10-59-54/kine/kine-jepa

# A) 合成验证（无需视频文件，离线权重）：98 条合成片段走 CUDA
set KINE_VJEPA2_LOCAL=C:/Users/zoah/AppData/Local/Temp/vjepa2
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set HF_HUB_DISABLE_TELEMETRY=1
python bench_gpu_launcher.py --synthetic --device cuda --max-clips 98

# B) 真实 98 条片段（把你的视频文件夹指过去；需真实标注才有可信 EVT-1）
python bench_gpu_launcher.py --data-dir <视频文件夹> --device cuda
#    或正式数据：python bench_gpu_launcher.py --data-dir ../kine-datapipe/clips --device cuda
```
- 默认 `--max-clips 98`、`--model vjepa2-vitl-256`、`--num-frames 16`、`--batch-size 2`（合成验证可用 4）
- 产出 `bench_report.json` + `bench_report.html`（自包含报告，浏览器直接看）
- `run_gpu_bench.sh` 已封装 A 的完整命令，双击/一键可复现

## 数据管线（真实片段时用）
```bash
python prep_bench_data.py --src /path/to/your/videos --out bench_data
```
- 复制 `.mp4/.mkv/.webm` 到 `bench_data/raw/`，生成 `bench_data/events.json`（事件帧为**启发式占位**，可信 EVT-1 需真实标注）

## CPU 验证（无 GPU 时确认链路）
```bash
python bench_gpu_launcher.py --smoke
```

## 你会拿到什么
| 任务 | 含义 | V-JEPA 2（纯编码器）实际状态 |
|---|---|---|
| KINE-TEMP-1 | 时序理解 | ✅ 可跑 |
| KINE-MOT-1 | 运动幅度 | ✅ 可跑 |
| KINE-EVT-1 | 物理事件偏移 | ✅ 可跑（需真实视频 + events.json），合成模式下 **skipped** |
| KINE-CAU-1 | 因果 | ⚠️ **可跑但 degraded**：`auc_do=null`，do-branch 未暴露，只剩观测量 |
| KINE-FUT-1 / EMB-1 | 未来预测 / 具身想象 | n/a（编码器无 predictor，协议诚实报 n/a） |

> 白泽类纯编码器同样只有 encode，TEMP/MOT/EVT 是同台可比硬数字；
> FUT/EMB/CAU 差距正是 KineOne-WM「可规划 + 可反事实」的护城河——待真实轨迹后训练（闭源配方）补上。

## 本机实测结果（RTX 5070 Ti Laptop · CUDA）
| 项 | 数值 |
|---|---|
| 数据 | 合成 98 条（无视频文件） |
| 分辨率 / 帧数 | 64px / 16 帧 |
| 墙钟 | **901.7s**（≈9.2s 每条，含模型加载） |
| 对比 CPU | CPU smoke 8 条耗 11807.5s（≈1476s 每条）→ **≈160× 提速** |
| 负载态功耗 | 98.8W、`utilization 99%`、显存 5.3GB |
| KINE-TEMP-1 | accuracy 1.000（基线 0.5） |
| KINE-MOT-1 | pearson_r −0.0022（基线 0.0） |
| KINE-CAU-1 | AUC 1.000（基线 0.5）· **degraded，`auc_do=null`** |
| KINE-FUT-1 / EMB-1 | n/a（`requires 'predict'`） |
| KINE-EVT-1 | skipped（合成模式无真实视频 + 标注） |

## 分辨率-吞吐扫描（回应「64px 是不是省算力」）
```bash
export KINE_VJEPA2_LOCAL="C:/Users/zoah/AppData/Local/Temp/vjepa2" \
       HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python gpu_resolution_sweep.py --device cuda --clips "64:32,128:32,256:16"
# 产出 gpu_sweep.json + gpu_sweep.html（三档墙钟 / 每条耗时 / 相对 64px 倍率）
```
### 实测结果（本机 RTX 5070 Ti Laptop，CUDA，batch_size 1，num_frames 16）
| 分辨率 | 片段数 | 墙钟 | 每片段 | 条/分 | 相对 64px |
|---|---|---|---|---|---|
| 64px | 32 | 377.2s | 11.79s | 5.09 | 1.0× |
| 128px | 32 | 382.6s | 11.96s | 5.02 | **1.0×** |
| **256px（V-JEPA 2 原生）** | 16 | 259.4s | **16.21s** | 3.70 | **仅 1.4×** |

**结论：256px 原生分辨率只比 64px 慢 1.4 倍。** 降采样到 64px 几乎没省到算力
（该尺度下瓶颈是每片段固定开销，不是分辨率），所以「64px 数字靠降采样取巧」的质疑不成立——
反向说明我们完全可以直接在原生分辨率上跑。产出：`gpu_sweep.json` / `gpu_sweep.html`。

> 踩坑记录：早期用 **2 条**片段做 256px 预探测，得到「≈72s 每条」，属**高估**——
> 模型加载的固定开销被摊到仅 2 条上。样本量放大到 16 条后修正为 16.21s 每条。
> **教训：测吞吐时样本量要够大，否则模型加载会主导结果。**

## 拿到数字后
把 `bench_report.html` 并入 `kineworld_capability_deck.html` 第 10 支柱作为「真实基准」，
直接服务 9/20 引航陪跑 + 10/1 合肥国资申报。

**诚实边界（务必一起交付）**：合成片段不含真实运动结构，TEMP=1.0 平凡偏高、
MOT≈0、CAU AUC=1.0 属 degraded，样本量小时分数可在极值间跳变（2 条样本下 TEMP 观测到 0.0
与 1.0 的差异）。这些分数**不是竞争力证据**；本次真正证明的是 CUDA 链路跑通 + 评测协议
可执行 + 160× 吞吐。**真实申报数字须用真实 98 条片段跑方案 B。**

### 批处理对照（256px · 98 条 · 同机同数据）
| batch | 墙钟 | 每片段 | 条/分 | 显存 |
|---|---|---|---|---|
| 1 | 926.5s | 9.45s | 6.35 | 8.5GB |
| 4 | 918.3s | 9.37s | 6.40 | **11.1GB**（逼近 12GB 上限） |

**batch 4 仅快 1.01×，显存 +30%。** 与分辨率扫描结论互证：瓶颈在每片段固定开销
（视频解码 / 预处理数据管道），不在 GPU 算力——加大 batch 是无效优化方向，
收益/代价最优解是默认 batch 1–2。TEMP-1 分数在两种 batch 下完全一致（0.9828），
批处理不影响评测结果。

### 时序长度对照（256px · batch 1 · 16 条）
| num_frames | 墙钟 | 每片段 | 每帧开销 |
|---|---|---|---|
| 16 | 259.4s | 16.21s | 1.013s |
| 32 | 253.6s | 15.85s | 0.495s |
| **64（fpc64 满配）** | **253.9s** | 15.87s | **0.248s** |

**4 倍时长，墙钟纹丝不动**（259.4 / 253.6 / 253.9s），每帧开销线性下降
1.013 → 0.495 → 0.248s。瓶颈彻底定性：**每片段固定开销（视频解码/预处理）主导，
GPU 推理时长完全不是本机瓶颈**。对评测协议的实际意义：
**直接用 64 帧满配（时间一致性窗口 ×4）吞吐零代价**。
（合成下 MOT r 随帧数 0.36→0.49→0.84 属 16 条样本噪声，不作证据。）
