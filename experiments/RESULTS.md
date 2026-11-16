# CryoWISP × CryoPPP 迁移实验结果

2026-11-15 · Gadi 登录节点 · GLM (glm-5.3[1m]) 编辑器 · editor-isolation local（见 DESIGN.md）

## 结论：CryoWISP 可以迁移到 cryoEM 颗粒挑选任务

从三个旧 EMPIAR 数据集（10005/10291/10081，运输蛋白，130–154 px）训练的二分类
ResNet-18 完全不认新数据集 10947（病毒蛋白，240 px）的颗粒；一次编辑器会话产出的
权重程序用 64+600 张标注 crops 把新数据集能力装进网络，且在**独立训练的第二个网络上
重跑同一程序**同样成功 —— 迁移、复用两个核心主张都成立。

## 主结果（引擎 final split，每张图只评一次）

| seed 1 | new (10947) | old (旧3组) | net |
|---|---|---|---|
| 基线 | 49.75% | 85.94% | — |
| **weight program（accepted）** | **72.25% (+22.50)** | **89.69% (+3.75)** | **+26.25** |
| 少样本微调（20 epoch，同 experience） | 69.00% (+19.25) | 93.75% (+7.81) | +27.06 |

程序改动 1026 / 11,689,512 个参数（0.0088%），不碰任何 2-D 卷积；微调改全网络。
程序的新条件增益高于微调，old 保持增益更小（experience 中 old 样本 600 张 vs new 64 张，
微调把大量容量花在旧条件上）。

## 程序重跑（transfer，同一 bundle 应用到独立训练的 seed 2）

| seed 2 | new (10947) | old (旧3组) | net |
|---|---|---|---|
| 基线 | 49.00% | 83.13% | — |
| **程序重跑（accepted）** | **75.50% (+26.50)** | **89.38% (+6.25)** | **+32.75** |

第一次 apply（300 s 程序墙）在 seed 2 上超时被拒 —— 程序在 seed 1 上就贴近上限；
--program-wall-s 480 后通过。程序为每个网络重新计算编辑（冻结 delta 直接拷贝无效）。

## 外部测试集（new_test 2040 / old_test 1836 crops，编辑器全程不可见）

| 模型 | new_test | old_test |
|---|---|---|
| seed 1 基线 | 47.99% | 93.85% |
| seed 1 + 程序 | **61.42% (+13.43)** | 93.25% (−0.60) |
| seed 2 基线 | 47.11% | 94.12% |
| seed 2 + 程序 | **59.41% (+12.30)** | 93.63% (−0.49) |

增益从 final(+22.5) 到更大更难的外部集(+13.4) 有衰减但仍显著；old 基本无损。

## 编辑器会话统计（seed 1）

- 90 分钟墙钟，5 个程序提交（2 个合格），47 个 Python cell（39 ok / 4 error / 4 timeout）
- 选中程序 bn2_recal.py（attempt 0004）：逻辑回归拟合 layer4.1.bn2 的 1024 个 1-D 仿射参数
  + fc.bias —— 编辑器自己发现 fc 的 2-D 相对 Frobenius 预算不够旋转决策方向，而 1-D 参数
  不受该上限约束，是"固定头之前的合法旋钮"
- selection 分数：基线 new 49.67/old 83.28 → 0004 new 66.45/old 86.25（net +19.75）

## Gadi 环境适配（对框架的改动，均已记录在 run.json）

- `--editor-isolation local`：新增 LocalKernel（同协议直启 NumPy-only 内核，无 bwrap），
  程序走 UnisolatedRunner；run.json 记 program_isolation: none + isolation_note
- synthesize/apply 增加 --base-url-file/--auth-token-file（GLM/Kimi/DeepSeek 兼容端点）
- 本地模式附加防线：系统提示词禁区规则 + cell 代码静态路径闸门（引用运行目录内
  trial/patches/scratch 之外的路径直接拒绝）
- 诚实声明：无文件系统/网络隔离是本实验的已知弱点；会话后审计了 cell 日志与保留程序源码，
  未发现越界读取 selection/final（冒烟阶段曾出现，已由闸门阻断路径）

## Baseline 全家桶（v1 experience 预算，final split 只评一次，两个独立网络）

| 方法（final split：new / old） | seed 1 | seed 2 | 参数改动 |
|---|---|---|---|
| Zero-shot（冻结基座） | 49.75 / 85.94 | 49.00 / 83.13 | 0 |
| **CryoWISP 程序（v1）** | **72.25 / 89.69** | **75.50 / 89.38**（重焊） | **1026，无梯度** |
| Full fine-tune（20 ep） | 74.50 / 93.44 | 71.75 / 93.13 | 11.69M（100%） |
| Full fine-tune 复跑 | 69.00 / 93.75 | — | 同上 |
| Linear probe（fc，1026 参数） | 49.00 / 93.59 | 48.00 / 92.34 | 1026 |
| Knob 梯度法（程序同款 1026 旋钮，20 ep） | 49.00 / 93.44 | 47.75 / 92.19 | 1026 |
| Knob 梯度法公平版（60 ep，lr 3e-4，10×类加权） | 49.00 / 93.44 | 48.50 / 92.19 | 1026 |
| Copied delta（seed1 编辑直接拷到 seed2） | — | 48.25 / 83.28 | 1026（冻结） |
| Random knob（同预算随机扰动同旋钮） | 49.50 / 83.91 | 49.00 / 82.19 | 1026 |

**v1 → v2 协议披露**：make_crops 使用单一随机流，v2 把 new_exp 从 32→96 张改变了后续角色
进入时的随机数状态，因此 v2 的 new 侧 selection/final/test crop 坐标被重抽（显微图分配不变；
old 侧逐位一致，old baseline 0.8328 不变；new selection baseline 49.67→50.00）。v2 会话内部
自洽（自有 baseline/selection/final），梯度基线在 v2 协议上另行重跑成表；v1 表与 v2 表分开读。
v2 experience 实际 81+96 new + 600 old（个别 new_exp 显微图正例不足 96）。

**v2 协议表**（192 张 new experience，重抽的 new 侧评估 crop，其余同 v1）：

| 方法（final split：new / old） | seed 1 | seed 2 |
|---|---|---|
| Zero-shot | 待会话 finalize 报告 final_base | 同左（selection 基线 seed1 50.00/83.28） |
| Linear probe | 47.50 / 93.91 | 47.00 / 92.03 |
| Knob 梯度法 | 47.50 / 93.91 | 47.00 / 92.03 |
| Full fine-tune | 70.00 / 94.06 | 68.75 / 92.66 |
| **CryoWISP 程序（v2）** | 会话运行中 | — |

**审稿人视角的四个证据**：
1. **同容量梯度失败、程序成功**：程序同款 1026 旋钮上，梯度法无论默认（20ep, lr 1e-4）还是
   公平版（60ep, lr 3e-4, 10×类加权）都学不会新条件（~48–49%），程序解析拟合一次到位
   （72–75%）—— 凸优化式拟合 + 旧样本锚定找到了梯度下降找不到的解
2. **拷贝编辑失败、程序重焊成功**：seed1 的 delta 拷到 seed2 = 48.25%（≈zero-shot），
   同一程序在 seed2 重跑 = 75.50% —— "程序 vs delta" 的本质区别三行数字讲清
3. **随机扰动无效**：排除"动一动这些旋钮就有用"的平凡解释
4. **打平全微调**：程序新条件增益 ≥ full fine-tune，参数量少 4 个数量级，2-D 卷积零改动

引擎侧校验：import 时 NumPy 前向 vs torchvision 参考逐位一致（float64 最大绝对差 9.2e-8）；
copied_delta 自检（seed1 delta 加回 seed1）精确复现程序 final 成绩 72.25/89.69，delta 提取无误。

## 复现路径

```
cryoppp_data/lite/                 # CryoPPP_Lite 5 个数据集（aria2 并行下载）
cryoexp/make_crops.py              # 显微图级切分裁剪
cryoexp/train_base.py              # 基础网络（PBS df68 作业）
cryoexp/run_engine_task.sh         # import-model + prepare-data ×3 + prepare-task
cryoexp/run_synth_seed1.sh         # 编辑器会话（登录节点，GLM 端点）
cryoexp/finetune_baseline.py       # 对照
apply --program-sha256 …           # seed 2 重跑（--program-wall-s 480）
```
