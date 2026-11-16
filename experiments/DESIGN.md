# CryoWISP × CryoPPP 迁移实验设计

日期：2026-11-15 · 环境：NCI Gadi（登录节点无 userns，bwrap 不可用）

## 任务定义
把 cryoEM 颗粒挑选转成 **patch 二分类**（background=0 / particle=1，按 ImageFolder 字母序），
输入 224×224 RGB crops（box = 1.5×颗粒直径，双线性缩放到 224）。
迁移问题：基础网络在旧数据集上训练后，用 CryoWISP 权重程序从少量标注
crops 安装"新数据集颗粒识别"能力，同时保持旧数据集表现。

## 数据（CryoPPP_Lite，来自 calla.rnet.missouri.edu/cryoppp_lite/）
| 角色 | EMPIAR | 蛋白类型 | 颗粒直径 | 显微图 |
|---|---|---|---|---|
| old×3 | 10005 | TRPV1 运输蛋白 | 142 px | 29 张有标注 |
| old×3 | 10291 | 运输蛋白 | 130 px | 300 |
| old×3 | 10081 | 运输蛋白 | 154 px | 300 |
| new | 10947 | 病毒蛋白 | 240 px | 400 |

裁剪：正例 = 专家坐标中心；负例 = 随机中心，与任何粒子/假阳性中心距离 ≥ box/2 + d/2。
显微图级切分（同一张图不跨角色），种子 7：
- base_train：旧 3 组各 ~2000/类（ImageNet 预训练起点微调）
- old_exp/old_sel/old_fin：旧 3 组各 100/类/角色（引擎 old 条件）
- old_test：旧 3 组各 ~300/类（自有评估）
- new_exp 32/类 · new_sel 150/类 · new_fin 200/类（引擎 new 条件）· new_test 1000/类（自有评估）

## 引擎任务（application/skill 路径）
- 模型：FP32 torchvision ResNet-18 state_dict（2 类头），import-model 校验 + NumPy 前向奇偶校验
- prepare-data ×3 → prepare-task：--retention-drop 0.01 --min-new-gain 0.01
- synthesize：编辑器 = Claude Code CLI → GLM 端点（open.bigmodel.cn/api/anthropic，
  模型 glm-5.3[1m]，凭证文件 ~/.config/cryowisp/{glm_base_url,glm_auth_token}，mode 600）
- **editor-isolation local**（本分支新增）：Gadi 全节点 user.max_user_namespaces=0，
  bwrap 不可用。LocalKernel 直启 NumPy-only 内核（-I、内存 rlimit、cell 预算、协议校验、
  虚拟时钟），程序经 UnisolatedRunner（同驱动/时钟/上限，无文件系统隔离）。
  run.json 如实记录 program_isolation: none + isolation_note。
- 预算：checks 8 · cells 100 · wall 5400 s · program-wall 300 s · threads 2（实测 NumPy 前向
  2 线程最快 ~0.4 s/图；多线程反而变慢）

## 对照
1. base 模型直接评估（new_sel/new_fin/new_test/old_*）
2. 少样本微调（finetune_baseline.py）：同样只用 experience（664 crops），selection 挑 epoch，final 只评一次
3. weight program（synthesize 产物）
4. 迁移：同程序 bundle apply 到 seed 2 独立训练的网络（--program-sha256 无沙箱复用）

## 关键路径
- 数据：cryoppp_data/lite/<id>/，crops 在 cryoexp/data/crops/
- 模型：cryoexp/models/model_seed{1,2}.pt
- 引擎：cryoexp/engine/seed<k>/{model,experience.npz,selection.npz,final.npz,task}
- 会话：cryoexp/engine/seed1/synth（editor 会话）、…/reuse_seed2（迁移）
