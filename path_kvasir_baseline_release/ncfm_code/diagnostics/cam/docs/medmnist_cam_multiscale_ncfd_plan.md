# MedMNIST 可解释性与多尺度 NCFD 正式方案与代码审计

## 0. 文档目的

这份文档同时服务两件事：

1. **把后续研究路线定清楚**：先做什么、为什么先做、哪些实验必须先跑、哪些改动先不要做。
2. **把 NCFM 相关代码分支差异审清楚**：原版、clean MedMNIST 版、token-local 版、raw-patch 版、以及 `2026_dd_med_v3` 之间到底有什么关系，哪些改动值得借鉴，哪些不应该直接复用。

本方案以当前干净仓库 **`NCFM_medmnist_clean`** 为工作基线，目标是：

- 保持 **纯 NCFM 算法主体尽量不动**
- 先完成 **MedMNIST 上的可解释性审计（CAM）**
- 再做 **面向小尺寸医学图像的多尺度 NCFD 改进**
- 所有结论都建立在干净对照和可复现实验上

---

## 1. 当前仓库与分支关系审计

## 1.1 审计对象

本次需要区分以下 5 个代码基线：

1. **上游原版 NCFM**  
   - GitHub: `https://github.com/gszfwsb/NCFM.git`

2. **当前工作基线：`NCFM_medmnist_clean`**  
   - 目标：纯 NCFM + MedMNIST 数据接口
   - 不引入 local token / local patch / AUC-best 等算法分支

3. **`patchncfm/token_local_github_e9c66fd`**  
   - 医学数据 + AUC + local token 分支

4. **`patchncfm/current_raw_patch_worktree`**  
   - 在 token-local 分支基础上，改成 raw local patch 分支
   - 并带有大量 sweep/generated 产物

5. **`2026_dd_med_v3`**  
   - 一个独立 benchmark 仓库
   - 内嵌了 `methods/NCFM`
   - 不是 NCFM 主仓库的干净 fork

---

## 1.2 关系结论

### A. `NCFM_medmnist_clean`

这是当前最适合作为研究起点的基线，因为它的定位非常清楚：

- 保留原版 NCFM 的核心算法主体
- 只补 MedMNIST 数据接口
- 不把 local token/raw patch 历史实验逻辑混进来

当前我们已经明确保持不动或尽量不动的核心算法文件是：

- `NCFM/NCFM.py`
- `NCFM/SampleNet.py`
- `condenser/compute_loss.py`
- `condenser/Condenser.py`

当前 clean 版主要新增/修改的是：

- `data/medmnist.py`
- `data/transform.py`
- `utils/utils.py`
- `condenser/condense_transfom.py`
- `config/ipc10/*.yaml`
- `requirements.txt`
- `utils/experiment_tracker.py`（仅修 top-k 兼容）

### B. `token_local_github_e9c66fd`

这是一个 **“医学数据 + local token 匹配 + AUC 评估”** 分支。

主要特征：

- 新增 MedMNIST/PCam 数据集支持
- 新增 AUC 计算与按 AUC 选 checkpoint 的逻辑
- 在 NCFM 核心 loss 中加入 local token 分支
- 配置字段使用：
  - `use_local_token_ncfm`
  - `local_token_block_count`

该分支的价值：

- 说明作者已经尝试过“局部结构增强”的方向
- 对 MedMNIST 数据接入、统计量、配置组织方式有参考价值

该分支的问题：

- 把数据适配、评估协议、算法改动混在一起
- 命名已经和我们想要的“干净 multi-scale NCFD”不一致
- 不适合直接作为当前主线基线

### C. `current_raw_patch_worktree`

这是一个 **“医学数据 + raw local patch matching + sweep 产物”** 分支。

它相对 token-local 分支的关键变化是：

- `NCFM/NCFM.py` 中增加：
  - `_patchify_images(...)`
  - `local_patch_match_loss(...)`
  - `global_local_patch_match_loss(...)`
- `NCFM/SampleNet.py` 中新增：
  - `MultiScaleSampleNet`
- `condenser/compute_loss.py` 中改成了：
  - 同时返回 `loss_total / loss_global / loss_local`
  - 支持全局与局部 sampling net 分开对抗更新
- `utils/experiment_tracker.py` 中加入了：
  - `loss_global_data`
  - `loss_local_data`
  - 对应绘图逻辑
- `condense/condense_script.py`、配置文件里加入：
  - `use_local_patch_ncfm`
  - `local_patch_block_counts`
  - `lambda_local`
  - `lambda_global`

这个分支的价值：

- 它证明“局部匹配”这条路在工程上是可接入的
- 它已经试过把 local 分量拆成可记录、可 sweep、可对抗训练的结构
- 它对我们后续实现 `global + local` 有很强的参考价值

这个分支的问题也很明显：

1. **不是干净最小实现**
   - 它不是只改 loss，而是联动改了 sampling net、script、tracker、config、notebook

2. **局部匹配对象是 raw image patches**
   - 这和我们当前更想做的 **feature-map local NCFD** 不一样
   - raw patch 方案更接近输入空间匹配，不够“纯 NCFD 表征空间”

3. **历史产物很多**
   - `config/generated/**`
   - `notebooks/**`
   - 旧字段残留
   - 硬编码旧路径的 shell/sweep 产物

4. **命名包袱大**
   - `use_local_patch_ncfm`
   - `local_patch_block_counts`
   - `lambda_local`
   - 这些不适合直接成为 clean 主线接口

### D. `2026_dd_med_v3`

`2026_dd_med_v3` 不是 NCFM 主仓库的 clean fork，而是一个 benchmark 仓库。

其特点：

- 根目录是 benchmark/eval 框架
- `methods/NCFM` 是内嵌的一份 NCFM 副本
- 真正的 NCFM 相关定制主要发生在根目录 benchmark 代码里，而不在 `methods/NCFM` 主体内部

它的研究价值：

- 可以借鉴其 benchmark/eval protocol 设计
- 可以参考它如何把 NCFM 对齐到统一评估框架

但它不适合作为我们的主开发基线，因为：

- 它不是为“纯 NCFM + MedMNIST + 新损失研究”组织的
- benchmark 包装层会干扰我们做算法级因果分析

---

## 1.3 审计结论：应该以哪个仓库为主线

**主线应固定为 `NCFM_medmnist_clean`。**

原因：

- 干净
- 边界清晰
- 对照简单
- 最适合做“问题诊断 -> 方法改进 -> 消融验证”

其他分支只作为“参考来源”，不要直接 merge：

- `token_local_github_e9c66fd`：参考 MedMNIST/PCam/AUC 接入经验
- `current_raw_patch_worktree`：参考 local loss 接入工程方式
- `2026_dd_med_v3`：参考 benchmark 协议与跨框架评估思路

---

## 2. 当前 clean 基线的工程事实与风险

## 2.1 Clean 基线当前已经完成什么

当前 `NCFM_medmnist_clean` 已经实现：

- MedMNIST 数据集包装
- 动态/统一 transform 接入
- loader 兼容 NCFM 原生数据接口
- condense 后 synthetic evaluation transform 接入
- 若干 2D MedMNIST config
- 二分类 top-k 兼容修复

也就是说：

> 它已经是一个“纯 NCFM + MedMNIST 数据接口”的干净起点。

---

## 2.2 现在还必须先处理的风险：RGB 28×28 ConvNet 兼容

这是当前最重要的工程前置条件之一。

在 clean repo 现状下，**ConvNet 对 28×28 RGB 输入存在结构兼容风险**：

- `gray28` 可跑
- `rgb32` 可跑
- `rgb28` 存在维度不匹配风险

问题根源在 [models/convnet.py](../models/convnet.py#L137-L172)：

- 原版实现对 `im_size == 28` 的假设，本质上是服务灰度 MNIST 风格输入
- `pathmnist` / `bloodmnist` 是 **28×28 RGB**
- 所以在 evaluator backbone 选择 ConvNet 时，必须先确认 flatten 维度与 classifier 输入维度严格一致

这件事不属于算法改进，而属于：

> **MedMNIST RGB 28 兼容修复 / 模型尺寸 sanity**

它必须作为正式计划里的 **P0.5**，排在 CAM 和多尺度 NCFD 之前。

---

## 2.3 CAM 的另一个工程事实：默认 evaluation 不一定保留 checkpoint

当前评估入口是 [evaluation/evaluation_script.py](../evaluation/evaluation_script.py)。

它会：

- 读 condensed dataset
- 构造 syndataloader
- 调用 `synset.evaluate(...)`

但当前流程并不天然保证：

- 训练出来的 evaluator checkpoint 一定保留
- 真实数据训练与合成数据训练的 evaluator 都有稳定可复用的 checkpoint 命名规范

所以如果我们要做 CAM，不能只依赖现在的 evaluation 主流程；必须补充**独立 CAM 入口**。

建议新增：

- `evaluation/cam_utils.py`
- `evaluation/run_cam.py`

并支持最小参数：

- `--dataset`
- `--data_dir`
- `--model`
- `--checkpoint`
- `--train_source real|synthetic`
- `--split train|val|test`
- `--num_samples`
- `--target_layer`
- `--save_dir`

---

## 3. 研究总目标

我们接下来做两件事，但必须严格分阶段：

1. **CAM / 热力图可解释性**
2. **多尺度 NCFD 损失**

其中：

- CAM 是 **分析工具 / 可解释性模块**
- 多尺度 NCFD 是 **算法改进模块**

这两者绝对不能一开始混做。

---

## 4. 总体研究原则

## 4.1 两条线分开推进

### 阶段 A：先做 CAM，不改算法

目标：

- 看纯 NCFM 在 MedMNIST 上学到了什么
- 判断 synthetic-trained evaluator 是否真的忽略病灶局部区域
- 给后续 multi-scale NCFD 是否必要提供证据

### 阶段 B：再做多尺度 NCFD

目标：

- 仅在确认 CAM 暴露出局部注意力问题之后
- 再引入最小算法改动：`global + local`
- 用性能、可解释性、稳定性共同验证收益

---

## 4.2 保持对照清晰

必须始终区分这三类对象：

1. **真实数据训练的 evaluator**
2. **合成数据训练的 evaluator**
3. **condensation 自身的损失设计**

对应地，CAM 和性能实验必须明确写清楚训练来源：

- real train -> real test
- synthetic train -> real test
- multi-scale synthetic train -> real test

---

## 4.3 第一阶段只做 2D MedMNIST

第一轮研究只覆盖：

- 2D MedMNIST
- 优先：`bloodmnist`, `pathmnist`, `pneumoniamnist`
- 先不碰 3D MedMNIST
- 先不引入 AUC-best checkpoint 逻辑
- 先不复用 local token / raw patch 历史方案
- 先不同时上 SSIM / Dice / MS-SSIM / 边缘损失

原因很简单：

- 3D 不是“顺手扩展”问题，而是另一套数据、网络、特征图、CAM 和 decode 设计
- 如果第一版就混进去，结论会完全变脏

---

## 5. CAM 可解释性模块：正式设计

## 5.1 CAM 的研究问题

我们希望用 CAM 回答三个问题：

1. 真实数据训练的模型关注区域是否合理？
2. 纯 NCFM 合成数据训练的模型关注区域是否偏移？
3. 后续多尺度 NCFD 是否让 synthetic-trained 模型的注意力更接近 real-trained 模型？

---

## 5.2 第一版方法：Grad-CAM

第一版只做 **Grad-CAM**。

原因：

- 对 CNN evaluator 最稳妥
- 实现成本最低
- 可读性最好
- 足够支撑第一轮问题诊断

后续若需要，再补：

- Grad-CAM++
- Eigen-CAM

但不应进入 phase 1 必需项。

---

## 5.3 CAM 挂载层选择

这件事不能写死，要根据 evaluator backbone 决定。

### 如果 evaluator 是 ResNet 类

优先挂：

- 最后一个卷积 stage
- 例如 `layer4[-1]`

### 如果 evaluator 是 ConvNet 类

优先挂：

- 最后一层仍保留空间分辨率的 conv feature

### 原则

- 不能挂到 FC 层
- 不能挂到 global pooling 之后
- 必须保留空间维度

---

## 5.4 CAM 的输入样本设计

每个数据集至少做三组：

1. **真实训练 / 真实测试**
2. **纯 NCFM 合成训练 / 真实测试**
3. **multi-scale NCFD 合成训练 / 真实测试**

每组都要覆盖：

- 预测正确样本
- 预测错误样本
- 高置信度错误样本
- 置信度接近决策边界的样本（可选）

建议每个数据集第一轮取：

- 50~100 个 test 样本
- 各类尽量均衡

---

## 5.5 CAM 输出格式

建议目录：

- `results/cam/<dataset>/<model_source>/`

其中 `model_source` 例如：

- `real_train`
- `pure_ncfm_syn_train`
- `multiscale_ncfm_syn_train`

每个样本至少保存：

- 原图
- 热力图
- overlay 图
- 预测类别
- 真实类别
- 置信度
- 样本索引

另存一个汇总表：

- `results/cam/<dataset>/<model_source>/summary.csv`

字段建议：

- `index`
- `split`
- `y_true`
- `y_pred`
- `confidence`
- `correct`
- `cam_entropy`
- `topk_activation_ratio`
- `cam_path`
- `overlay_path`

其中 `cam_entropy`、`topk_activation_ratio` 可以先作为可选字段预留。

---

## 5.6 CAM 工程落点

建议新增：

- `evaluation/cam_utils.py`
- `evaluation/run_cam.py`

### `cam_utils.py` 建议包含

- target layer 解析
- hook 注册与注销
- Grad-CAM 核心实现
- 热力图 resize / normalize / overlay
- 保存图像与 csv 的通用函数

### `run_cam.py` 建议负责

- 读取 checkpoint
- 构建 dataset / dataloader
- 选样本
- 执行前向与 backward
- 保存结果
- 输出 summary

尽量不要把 CAM 塞进现有 condensation 主流程。

---

## 5.7 CAM 的定量补充指标

MedMNIST 没有标准病灶 segmentation mask，所以 phase 1 主要还是**定性分析**。

但为了后续论文化，建议从第一版开始就把几个弱定量指标预留好：

1. **CAM entropy**  
   - 看注意力是否过于发散

2. **Top-k activation ratio**  
   - 看显著激活区域是否过大

3. **Center-of-mass distance**（可选）  
   - 如果某些数据集的病灶通常位于一定区域，可构造弱先验

4. **Inter-model similarity**（可选）  
   - 比较 real-trained 和 synthetic-trained 模型热力图的一致性

---

## 6. 多尺度 NCFD：正式设计

## 6.1 核心问题定义

原始 NCFD 在小尺寸医学图像上可能存在三个问题：

1. **局部病灶信息被全局特征淹没**
2. **原始频率采样对 28×28 小图不匹配**
3. **结构/相位信息的重要性可能被低估**

因此我们的目标不是“推翻 NCFM”，而是：

> 在尽量不改原始框架的前提下，把全局 NCFD 扩展成更适合小图医学结构的损失。

---

## 6.2 第一版的边界：先做 `global + local`，不先做 SSIM

建议按三层递进：

### B1：`global NCFD + local NCFD`

形式：

- `L = L_global + lambda_local_ncfd * L_local`

这是第一版最应该做的内容。

### B2：小图像超参数适配

扫：

- `num_freqs`
- `alpha`
- 如有需要再看采样分布参数

### B3：SSIM 或结构正则

仅在 B1/B2 已经证明局部结构收益存在后，再考虑加入：

- `L = L_global + lambda_local * L_local + lambda_ssim * L_ssim`

SSIM 不是第一版必须项。

---

## 6.3 我们第一版要做的 local NCFD，不等于历史 raw patch 分支

这里要明确区分：

### 历史 raw patch 分支做的是什么

见 `current_raw_patch_worktree` 中 [NCFM/NCFM.py](../../patchncfm/current_raw_patch_worktree/NCFM/NCFM.py)：

- 它把 **输入图像** 切成 patches
- 对每个 patch 计算局部匹配
- 使用 `local_patch_match_loss(...)`

### 我们当前推荐做的是什么

我们更推荐第一版做：

> **feature-map local NCFD**

即：

1. 从用于 NCFD 的 backbone feature tensor 出发
2. 保留空间维度 `[B, C, H, W]`
3. 在 feature map 空间做 local partition
4. 对每个局部块做 NCFD
5. 最后在块间平均

这样更适合当前“尽量保持纯 NCFM 表征逻辑”的目标。

---

## 6.4 local NCFD 的第一版定义

建议写成：

- `L_local = mean_k NCFD(f_real^(k), f_syn^(k))`

其中：

- `f_real^(k)`：真实数据在第 `k` 个局部块上的 feature
- `f_syn^(k)`：合成数据在第 `k` 个局部块上的 feature

### 分块策略建议

不要一开始就把“4×4 最优”当真理。

建议第一轮消融：

- `1x1`（等于无局部块）
- `2x2`
- `4x4`
- `7x7` 或按 feature map 每个空间点单独算

重点：

- 更推荐按 feature map 切，不是按原图 28×28 切
- 如果某个 backbone 输出正好是 7×7，逐位置局部匹配是一个自然 first version

---

## 6.5 第一版多尺度 NCFD 的工程落点

建议尽量把改动收缩到：

- [condenser/compute_loss.py](../condenser/compute_loss.py)
- 新增一个辅助文件（可选）：`condenser/local_ncfd.py`
- 少量配置字段

尽量先**不要**改：

- `NCFM/NCFM.py`
- `NCFM/SampleNet.py`
- `condenser/Condenser.py`

这是和 `current_raw_patch_worktree` 的最大区别：

- raw patch 分支已经把局部损失扩展到了核心匹配函数、采样器、tracker 和多个 script
- 我们第一版更希望把实现限制在 **loss 计算层**，这样最干净、最容易 review，也最容易回退

---

## 6.6 第一版建议的配置字段

建议只引入最少字段：

- `use_local_ncfd: true|false`
- `local_ncfd_grid: 2 | 4 | 7`
- `lambda_local_ncfd: 0.6`
- `num_freqs: 256 | 512 | 1024`
- `alpha_for_loss: 0.3 | 0.4 | 0.5`
- `beta_for_loss: 0.7 | 0.6 | 0.5`（若需要显式拆分幅度/相位权重）

不要直接沿用历史字段：

- `use_local_token_ncfm`
- `use_local_patch_ncfm`
- `local_patch_block_counts`

因为这些字段会把旧分支语义带回来，影响 clean 主线可读性。

---

## 7. 必须先做的预实验

这一部分是最关键的。

在正式实现 multi-scale NCFD 之前，必须先用预实验确认：

- baseline 是可复现的
- CAM 问题是真实存在的
- 小图参数敏感性是否显著
- local loss 是否真的比 global-only 更好

---

## 7.1 P0：数据与 loader sanity

目标：确认当前 clean repo 的 MedMNIST 数据管线是稳定的。

检查项：

- train/test split 正常
- label 类型与 shape 正常
- grayscale / RGB 都能进入 loader
- normalize 统计量正确加载
- condense/eval transform 一致

数据集先覆盖：

- `bloodmnist`
- `pathmnist`
- `pneumoniamnist`

输出：

- batch shape
- label shape
- 样本可视化
- transform 输出范围

---

## 7.2 P0.5：模型尺寸 sanity（必须先做）

目标：确认当前 evaluator backbone 对 MedMNIST 输入尺寸是结构可用的。

这是正式方案新增的硬前置项。

至少验证：

- `gray28`
- `rgb28`
- `rgb32`

重点关注：

- ConvNet flatten 维度
- classifier 输入维度
- 训练前向是否报 shape mismatch

如果 `rgb28` 不稳，优先把这件事作为 **兼容修复** 完成，然后再进入 CAM 和 baseline。

这是工程兼容问题，不属于算法改进。

---

## 7.3 P1：纯 NCFM MedMNIST baseline 跑通

目标：确认 clean repo 的 baseline 可复现。

每个数据集先做：

- 小 IPC：`1 / 10`
- 单卡跑通
- evaluator 正常训练与评估

输出：

- baseline accuracy
- 训练日志
- condensed 样本可视化

没有 baseline，后面 CAM 和 multi-scale 都没法解释。

---

## 7.4 P2：真实数据训练参考模型

目标：建立 real-trained reference model。

做法：

- 用真实 train 数据训练 evaluator backbone
- 在 real test 上评估
- 保存 checkpoint
- 作为 CAM 参考模型

输出：

- accuracy / F1（如需要）
- checkpoint
- 第一批 CAM 结果

---

## 7.5 P3：纯 NCFM 合成数据训练模型 + CAM 审计

目标：验证“全局对齐淹没局部诊断信息”的假设是否成立。

关注点：

- 性能是否明显低于 real-trained
- CAM 是否更分散
- 是否出现背景/边缘激活
- 是否主要捕捉全局亮暗模式，而非病灶区域

如果这一步看不出明显注意力偏移，多尺度 NCFD 的必要性必须重新评估。

---

## 7.6 P4：原始 NCFD 小图像超参数敏感性

目标：先验证是不是仅靠调小图参数就能解决相当一部分问题。

先不加 local loss，只扫原始 global NCFD 参数：

- `num_freqs = 128, 256, 512, 1024`
- `alpha = 0.3, 0.4, 0.5`

输出：

- accuracy
- 收敛速度
- 稳定性
- CAM 变化（若可负担）

如果这一步已经明显改善，说明问题不只是局部结构缺失，也和频率设计有关。

---

## 7.7 P5：local NCFD 最小原型

目标：只验证 local block matching 是否带来收益。

第一轮建议只试：

- baseline global only
- global + local(2x2)
- global + local(4x4)

固定其他参数，保持最小对照。

输出：

- accuracy 对比
- 训练耗时
- 数值稳定性
- CAM 是否更聚焦

---

## 7.8 P6：是否需要 SSIM

目标：判断结构正则是否真的提供附加收益。

仅在 local NCFD 已证明有效后再做：

- global + local
- global + local + ssim

观察：

- 性能
- 视觉质量
- 是否过度平滑
- 是否更难调参

SSIM 不应提前进入第一版主线。

---

## 7.9 P7：跨数据集稳健性复核

目标：避免方法只对一个数据集有效。

建议在以下数据集做复核：

- `bloodmnist`
- `pathmnist`
- `pneumoniamnist`

如果第一轮效果成立，再扩到：

- `retinamnist`
- `breastmnist`
- `dermamnist`

---

## 8. 实验矩阵建议

## 8.1 Phase 1：问题诊断矩阵

| 实验 | train source | loss | 目的 |
|---|---|---|---|
| A1 | real | supervised only | 建立参考模型 |
| A2 | pure NCFM synthetic | global NCFD | 建立 clean baseline |
| A3 | pure NCFM synthetic | global NCFD | 做 CAM 审计 |

---

## 8.2 Phase 2：参数敏感性矩阵

| 实验 | num_freqs | alpha | local |
|---|---:|---:|---|
| B1 | 1024 | 0.5 | no |
| B2 | 512 | 0.5 | no |
| B3 | 256 | 0.5 | no |
| B4 | 256 | 0.4 | no |
| B5 | 256 | 0.3 | no |

---

## 8.3 Phase 3：local NCFD 矩阵

| 实验 | global | local grid | lambda_local_ncfd |
|---|---|---|---:|
| C1 | yes | none | 0.0 |
| C2 | yes | 2x2 | 0.4 |
| C3 | yes | 2x2 | 0.6 |
| C4 | yes | 4x4 | 0.4 |
| C5 | yes | 4x4 | 0.6 |

建议第一轮不要把 grid 和 lambda 同时扫得太大，否则实验量失控。

---

## 8.4 Phase 4：结构正则矩阵（可选）

| 实验 | local | ssim |
|---|---|---|
| D1 | yes | no |
| D2 | yes | yes |

---

## 9. 代码实现路线

## 9.1 第一优先：CAM 代码骨架

建议先实现：

- `evaluation/cam_utils.py`
- `evaluation/run_cam.py`

并约定：

- checkpoint 读取方式
- target layer 命名方式
- 输出目录规范
- summary.csv 字段规范

这是最应该先落地的代码。

---

## 9.2 第二优先：兼容修复

如果 `rgb28` ConvNet 的尺寸问题仍存在，优先完成：

- evaluator backbone 的输入尺寸兼容
- flatten/classifier 维度校验
- 至少 `bloodmnist/pathmnist/pneumoniamnist` 能稳定前向训练

这件事比 multi-scale NCFD 更优先。

---

## 9.3 第三优先：local NCFD 最小实现

建议第一版实现路线：

1. 在 `compute_loss.py` 后追加 local 分量
2. 若代码过长，再拆到 `condenser/local_ncfd.py`
3. 不改 `NCFM/NCFM.py`
4. 不改 `NCFM/SampleNet.py`
5. 不引入 MultiScaleSampleNet
6. 不引入 raw image patch matching

目标是：

> 第一版 multi-scale NCFD 只体现为 loss 层的最小增量改动。

---

## 10. 暂不建议做的事情

第一版先不要和主线绑在一起的内容包括：

- 3D MedMNIST
- local token 历史分支复用
- raw patch 历史分支直接 merge
- AUC-best checkpoint 逻辑
- 同时上 SSIM / MS-SSIM / Dice / 边缘损失
- 重写 backbone
- 重写 sample net
- 同时做跨架构 benchmark 扩展

这些都会让第一轮结论失真。

---

## 11. 当前正式主计划

如果把当前路线收敛成一句话，就是：

> 以 `NCFM_medmnist_clean` 为唯一主线，先完成 MedMNIST 兼容性与 Grad-CAM 可解释性审计，确认纯 NCFM 是否存在局部病灶关注不足；若确认存在，再以最小 loss 层改动实现 feature-map local NCFD，并通过性能、稳定性和 CAM 对齐度共同验证其收益。

---

## 12. 下一步执行建议

最合理的顺序是：

1. **先补 P0.5：RGB 28 模型尺寸 sanity**
2. **实现 CAM 代码骨架**
3. **跑 real-trained / pure-NCFM-synthetic-trained 的第一轮 CAM**
4. **确认问题后再做 local NCFD 设计与实现**
5. **最后再评估是否需要 SSIM**

如果按研究效率排序，当前最应该立刻进入实现的是：

- `evaluation/run_cam.py`
- `evaluation/cam_utils.py`

因为 CAM 先回答“问题是否真实存在”，再决定算法是否值得改。
