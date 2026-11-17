# CryoWSP

**Cryo-EM Weight-Space Programming: new picking skills welded into frozen networks, one program at a time**

CryoWSP 把 WeightCode 的权重空间编程（WSP, Weight-Space Programming）范式变成 cryoEM
颗粒挑选的新方法：主干网络永久冻结（cryo），新数据集/新颗粒类型的识别能力由一个
LLM 综合出的**权重程序**"安装"进去 —— 像低温焊接（weld），而不是像微调那样把整个
网络熔掉重铸。接缝极小：v1 实测只改 0.009% 的参数，任何 2-D 卷积零改动。

**名字**：CryoWSP = Cryo-EM + **W**eight-**S**pace **P**rogramming，与范式名 1:1 映射
（WeightCode 论文与 effl-lab 仓库均以 WSP 指代该范式）。先前曾用名 CryoWISP
（WISP backronym）已弃用——直白胜过巧妙，审稿人零解码成本。
机制上"焊接"保留为第三阶段的意象：冻结主干 + 加性权重补丁（引擎产物本身就是
`weightcode.patch.v0` 的 add_dense 补丁）。

## 方法五段（Freeze–Show–Weld–Prove–Re-weld）

1. **Freeze（冻结）** — 在若干旧 EMPIAR 数据集上训练颗粒/背景二分类 ResNet-18，之后永久冻结。
   显微图级切分保证新旧数据互不泄漏。
2. **Show（示样）** — 专家只需在新数据集的几张显微图上点标注：~百张量级的正负 crop
   （v1 用 64，v2 用 192）+ 旧数据锚定 crop 构成 experience。
3. **Weld（焊接）** — 一个冻结的 LLM 编辑器（GLM/Claude）读取网络权重与前向代码，写出
   可执行的*焊接程序*：在编辑预算（≤5% 条目、2-D 张量相对 Frobenius ≤0.5）内计算加性
   权重修改。v1 中编辑器自主发现关键接缝：**fc 头的 2-D 预算不足以旋转决策方向，但
   BatchNorm 的 1-D 仿射参数（γ/β）不受 2-D 上限约束**，于是用逻辑回归在示样上拟合
   layer4.1.bn2 的 1024 个 γ/β + fc.bias —— 无梯度训练。
4. **Prove（验证）** — 固定选择规则：每个候选在 selection split 上打分（新条件增益 +
   旧条件保持），达标者中最优冻结，final split 只评一次；不达标则保留原网络（base-kept）。
5. **Re-weld（重焊）** — 保留的焊接程序是文件。在**任何**独立训练的兼容网络上重跑，
   程序为新网络重新计算编辑（冻结 delta 直接拷贝无效）；改一次，处处可装。

## 与现有方法的差别

| | 微调/crYOLO 类 | CryoWeld |
|---|---|---|
| 新数据集适配 | 全网络梯度训练（熔化） | 权重级加性编辑（焊接），主干冻结 |
| 标注需求 | 通常数千级 | 百张量级 expert crops |
| 改动规模 | 100% 参数 | ~0.01% 参数，2-D 卷积零改动 |
| 可复用性 | 每个网络重训 | 同一程序在任何兼容网络上重跑 |
| 可验证性 | 训练曲线 | 预算内检查 + 一次性 final 验证 + 审计日志 |

v1 结果（CryoPPP 10005/10291/10081 → 10947）：final +22.5/+3.75 pt（优于同预算微调的
+19.3），独立网络重焊 +26.5/+6.25 pt，外部测试集 +12~13 pt 且旧条件 −0.5 pt 内。

## 命名备选（均已查重）

- **CryoWSP**（选定，曾用名 CryoWISP 已弃用）— 与 WSP 范式名 1:1 映射，零解码成本
- ~~IcePick~~ — Langlois & Frank 2014 已用（低对比度挑选器，J Struct Biol）
- ~~CryoWeld~~ — 保留为机制隐喻（焊接 vs 熔化），未用作方法名；NY 一家消防公司同名，不同领域
