# MedMNIST 医学结构诊断与 NCFM 改进路线

更新时间：2026-05-20

本文档记录当前阶段已经完成的实验、可解释性分析、医学结构诊断结论，以及下一步算法改进路线。当前阶段的原则是：

- 暂不改训练代码
- 将 `B_minmax_ncfm_psi` 固定为论文式 NCFM-core / SOTA baseline
- 不再把 A/B/C 配置胜负作为主研究问题
- 先弄清医学图像蒸馏中“哪些结构重要、B 丢了什么”
- 再设计面向医学结构的 NCFM 改进

实验主目录：

```text
/data/zengqiang/experiments/ncfm_medmnist_ablation_20260519
```

本地工作基线：

```text
D:\Project\2026_4_methods\NCFM_medmnist_clean
```

---

## 1. 当前研究主线重新定位

### 1.1 之前的问题

最开始我们比较了 A/B/C 三个配置：

| 组别 | 名称 | 含义 |
|---|---|---|
| A | `A_pure_ncfd_wopsi` | 纯 NCFD / 固定频率采样，不使用 sampling net ψ |
| B | `B_minmax_ncfm_psi` | 论文式 min-max NCFM，使用 sampling net ψ |
| C | `C_code_default_enhanced` | 代码默认增强版，不开 ψ，但使用更多频率和 calibration |

但是现在研究问题已经不应该停留在：

> A/B/C 谁高 1 个点？

因为这会把注意力放到配置比较上，而不是医学图像蒸馏的本质问题上。

### 1.2 现在的主问题

现在应将问题改成：

> 以论文式 NCFM-core B 作为当前 SOTA 基线，在 MedMNIST 上，真实数据训练模型依赖哪些局部或结构性信息？B 蒸馏得到的数据是否保住了这些医学判别结构？如果没有，缺失的是哪类结构？应该加什么局部或结构约束来补？

因此后续主线是：

1. 固定 B 作为 NCFM-core baseline
2. 用 real-trained evaluator 作为参考
3. 分析 real CAM 与 B synthetic-trained CAM 的差异
4. 归纳医学数据结构特点
5. 设计 NCFM 的医学结构增强版本

---

## 2. 当前实验资产

### 2.1 已完成 formal A/B/C 结果

结果文件：

```text
reports/formal_ablation_summary.csv
reports/formal_ablation_summary.md
reports/formal_ablation_summary.json
```

formal 结果如下：

| Dataset | A ACC | B ACC | C ACC | 当前观察 |
|---|---:|---:|---:|---|
| pneumoniamnist | 92.15 | 91.03 | 94.23 | C 最高，B 也很强 |
| bloodmnist | 91.29 | 90.24 | 91.03 | A 略高，B 接近 |
| pathmnist | 81.06 | 78.11 | 83.09 | C 最高，B 明显低 |

注意：这个表不是后续主线重点。它只说明：

- B 是论文核心 NCFM，但不是所有 MedMNIST 设置下的最高配置
- C 作为代码默认增强版，在 PathMNIST / PneumoniaMNIST 上表现更强
- 但 C 的 CAM 不一定更接近 real-trained evaluator

### 2.2 已完成 CAM 资产

CAM 目录：

```text
results/cam/<dataset>/<group>/
```

已完成 12 组：

```text
bloodmnist/real_train
bloodmnist/ipc10_A_pure_ncfd_wopsi
bloodmnist/ipc10_B_minmax_ncfm_psi
bloodmnist/ipc10_C_code_default_enhanced

pathmnist/real_train
pathmnist/ipc10_A_pure_ncfd_wopsi
pathmnist/ipc10_B_minmax_ncfm_psi
pathmnist/ipc10_C_code_default_enhanced

pneumoniamnist/real_train
pneumoniamnist/ipc10_A_pure_ncfd_wopsi
pneumoniamnist/ipc10_B_minmax_ncfm_psi
pneumoniamnist/ipc10_C_code_default_enhanced
```

每组：

- 100 个 test 样本
- 原图
- CAM heatmap
- overlay
- `summary.csv`

总计：

- 12 个 CAM summary
- 约 3600 张 CAM 图像

### 2.3 已执行 notebooks

Notebook 目录：

```text
notebooks/
reports/notebook_html/
```

已经执行并导出 HTML：

| Notebook | 状态 | 说明 |
|---|---|---|
| `00_formal_metrics_overview_executed.ipynb` | 已执行 | formal 指标总览 |
| `01_cam_summary_spatial_bias_executed.ipynb` | 已执行 | CAM entropy / top-k / spatial bias |
| `02_cam_similarity_to_real_executed.ipynb` | 已执行 | real-vs-synthetic CAM similarity |
| `03_cam_gallery_and_class_prototypes_executed.ipynb` | 已执行 | CAM 汇总与 class prototype 可视化框架 |
| `04_occlusion_sensitivity_executed.ipynb` | 已执行 | 目前只是 occlusion 函数骨架，没有实际遮挡结果 |
| `05_pathmnist_c_low_similarity_visual_analysis_executed.ipynb` | 已执行 | PathMNIST C 低相似高置信样本可视化 |

重要说明：

`04_occlusion_sensitivity_executed.ipynb` 目前没有实际 occlusion 实验输出。它只定义了：

- hot-region mask
- cold-region mask
- random-region mask
- occlusion drop 汇总函数

因此当前不能声称“遮挡实验已经证明热点区域有因果重要性”。这一步仍需补跑。

---

## 3. B 作为 NCFM-core baseline 的配置

B 组配置文件位于：

```text
configs/<dataset>/ipc10_B_minmax_ncfm_psi.yaml
```

核心参数：

```yaml
network:
  net_type: convnet
  norm_type: instance
  depth: 3
  width: 1.0

condense:
  ipc: 10
  num_premodel: 20
  niter: 20000
  iter_calib: 0
  sampling_net: true
  num_freqs: 1024
  dis_metrics: NCFM
  factor: 2
  alpha_for_loss: 0.5
  beta_for_loss: 0.5
  decode_type: single
  teacher_model_epoch: 20

optimization:
  optimizer: adamw
  lr_img: 0.01
  mom_img: 0.5
  lr_sampling_net: 0.001
```

B 的定位：

> B 是论文式 min-max NCFM：开启 sampling net ψ，用 1024 个 frequency arguments，不使用 calibration。

后续把 B 称为：

```text
NCFM-core / NCFM-SOTA baseline
```

---

## 4. 当前 CAM 聚合结果

### 4.1 CAM summary 总览

来自 `03_cam_gallery_and_class_prototypes_executed.ipynb` 与 `reports/cam_summary_grouped.csv`。

| Dataset | Group | CAM Acc | Mean Conf | Entropy | Top10 Mass |
|---|---|---:|---:|---:|---:|
| bloodmnist | real_train | 0.93 | 0.8878 | 0.7166 | 0.2981 |
| bloodmnist | B | 0.86 | 0.8430 | 0.7305 | 0.4548 |
| pathmnist | real_train | 0.89 | 0.8819 | 0.8156 | 0.4345 |
| pathmnist | B | 0.76 | 0.7753 | 0.7789 | 0.4312 |
| pneumoniamnist | real_train | 0.88 | 0.9328 | 0.8652 | 0.3945 |
| pneumoniamnist | B | 0.93 | 0.9164 | 0.7141 | 0.4153 |

解释：

- `Entropy` 越高，CAM 越分散
- `Top10 Mass` 越高，说明最热 10% 像素承载更多 CAM 质量
- 这些指标不是病灶真值，只反映模型当前预测依赖的区域

### 4.2 B 与 real-trained 的 CAM similarity

来自 `reports/cam_similarity_to_real_grouped.csv`。

| Dataset | Pearson | Spearman | Cosine | Top10 IoU | Top10 Dice |
|---|---:|---:|---:|---:|---:|
| bloodmnist | 0.0681 | 0.0410 | 0.4406 | 0.0820 | 0.1299 |
| pathmnist | 0.0641 | 0.0543 | 0.4006 | 0.0928 | 0.1438 |
| pneumoniamnist | 0.0085 | 0.0171 | 0.4173 | 0.0599 | 0.1053 |

核心结论：

> B 虽然是 NCFM-core baseline，但它的 synthetic-trained CAM 与 real-trained CAM 的热点重合度整体较低。

特别是 top10 Dice 只有约 0.10 到 0.14，说明：

- B 可以让 evaluator 取得不错性能
- 但它学到的空间证据并不高度等同于 real-trained evaluator
- 高 ACC 不代表模型“看了同一块医学结构”

---

## 5. 医学数据结构特点诊断

### 5.1 医学信息不一定集中在图像中心

spatial bias 结果：

| Dataset | Group | Edge Mass | Center Mass | Corner Mass |
|---|---|---:|---:|---:|
| pathmnist | real_train | 0.5498 | 0.1978 | 0.1119 |
| pathmnist | B | 0.4815 | 0.2416 | 0.0934 |
| bloodmnist | real_train | 0.4154 | 0.3232 | 0.0563 |
| bloodmnist | B | 0.3633 | 0.3785 | 0.0512 |
| pneumoniamnist | real_train | 0.4621 | 0.2499 | 0.0716 |
| pneumoniamnist | B | 0.4345 | 0.2594 | 0.0643 |

观察：

- PathMNIST real-trained 模型明显更依赖边缘/外周组织区域
- BloodMNIST real-trained 同时看细胞中心形态与外周结构
- PneumoniaMNIST 更像大范围肺野结构，不是单点局部热点

因此不能简单认为：

> 医学图像关键结构都在中心。

更合理的判断是：

> 医学图像的判别结构可能出现在组织边界、细胞轮廓、肺野外周纹理、局部细胞形态等多个尺度和空间位置。

### 5.2 三个数据集的结构类型不同

#### PathMNIST：局部组织纹理与组织边界型

PathMNIST 是当前最适合作为 local / structure-aware NCFM 主战场的数据集。

证据：

- B formal ACC 只有 78.11，明显低于 C 的 83.09
- B 与 real 的 top10 Dice 只有 0.1438
- real-trained edge mass 高达 0.5498
- 低相似高置信样本显示 B/C 都可能在看与 real 不同的局部区域

诊断：

> PathMNIST 的关键判别信息很可能来自组织结构、组织边界、局部纹理模式。纯全局 NCFD 容易保住粗粒度统计，但未必保住真实模型依赖的组织结构位置。

#### BloodMNIST：细胞形态与局部纹理型

BloodMNIST 是第二主战场。

证据：

- B formal ACC 为 90.24，接近 A/C，但 CAM 与 real 重合仍低
- B 低相似高置信子集 Pearson 为 -0.149
- B selected 子集 top10 Dice 只有 0.0169
- B selected 子集 confidence 约 0.992

诊断：

> B 在 BloodMNIST 上可能学到了足以分类的细胞局部纹理或颜色形态，但这些区域与 real-trained evaluator 的关注区域不一致。它适合验证“细胞形态保真”和“前景结构保真”。

#### PneumoniaMNIST：大范围肺野结构型

PneumoniaMNIST 更适合作为全局结构对照。

证据：

- B formal ACC 为 91.03，C 可到 94.23
- real CAM entropy 为 0.8652，比 B 的 0.7141 更分散
- real 和 B 的 edge/center 差异没有 PathMNIST 那么强

诊断：

> PneumoniaMNIST 可能更依赖肺野整体结构、大范围纹理或全局分布。它不适合作为“极局部病灶被淹没”的最强证据，但适合作为方法不能过度局部化的对照。

---

## 6. B 基线丢失的信息

### 6.1 丢失一：真实模型的局部关注一致性

B 与 real-trained CAM 的 top10 Dice：

| Dataset | B vs Real Top10 Dice |
|---|---:|
| bloodmnist | 0.1299 |
| pathmnist | 0.1438 |
| pneumoniamnist | 0.1053 |

这说明：

> B 学到的判别区域与真实数据训练模型的判别区域不够一致。

注意这不是说 B 完全失败，而是说：

- B 的 global NCFD 可以让 synthetic evaluator 有较好性能
- 但它没有显式约束局部结构位置
- 因此可能学到另一套 shortcut 或替代性纹理模式

### 6.2 丢失二：高置信正确但结构错位

B 专用诊断目录：

```text
reports/real_vs_b_medical_structure_diagnosis/
```

每个数据集都筛选了：

```text
B 高置信 + 预测正确 + 与 real CAM 低相似
```

子集结果：

| Dataset | Subset | n | Conf | Pearson | Top10 Dice | Edge | Center | Corner |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| pathmnist | selected | 12 | 0.9925 | -0.0013 | 0.0686 | 0.3967 | 0.3056 | 0.1034 |
| bloodmnist | selected | 12 | 0.9921 | -0.1490 | 0.0169 | 0.4108 | 0.3803 | 0.0506 |
| pneumoniamnist | selected | 12 | 0.9978 | -0.0228 | 0.0232 | 0.4191 | 0.2929 | 0.0493 |

核心现象：

> B 可以非常自信地预测正确，但 CAM 与 real-trained 模型几乎不重叠。

这正是医学蒸馏中非常重要的问题：

> synthetic data 可能保住了分类可用信号，但没有保住真实医学判别结构。

### 6.3 丢失三：diffuse / edge-heavy / center-heavy / low-hot-mass 多种错位模式

从 PathMNIST C 的低相似样本分析可以看到类似模式；B 专用 montage 也显示类似现象。常见错位类型包括：

1. **edge-heavy**
   - CAM 偏边缘、外周、图像边界
   - 可能是组织边界，也可能是裁剪/背景 shortcut

2. **corner-biased**
   - CAM 偏四角
   - 更可疑，容易是位置偏置或边缘伪特征

3. **center-heavy**
   - CAM 集中在中心纹理块
   - 不一定错误，但如果与 real CAM 不重合，说明 B 可能使用替代性纹理

4. **diffuse**
   - CAM 分散
   - 可能说明模型只学到粗粒度统计，没有稳定局部证据

5. **low-hot-mass / near-empty CAM**
   - 热点很弱或 CAM 退化
   - 不适合直接做医学解释，但说明解释稳定性存在问题

### 6.4 丢失四：类条件局部结构不稳定

在 PathMNIST C 低相似样本中，低相似案例集中于少数类：

| Class | Count |
|---:|---:|
| 0 | 8 |
| 1 | 6 |
| 3 | 1 |
| 4 | 2 |
| 8 | 1 |

虽然这是 C 的分析，但它提示了一个重要方向：

> 局部结构丢失可能不是均匀发生在所有类别，而是集中于某些组织形态更难蒸馏的类别。

这对 B 的后续分析也重要，后面应进一步做：

- B 的 class-wise CAM prototype
- B 的 class-wise low-similarity rate
- class-wise occlusion sensitivity

---

## 7. 对现有 notebooks 的判断

### 7.1 `03_cam_gallery_and_class_prototypes_executed.ipynb`

这个 notebook 已经有用，主要贡献是：

- 汇总 12 组 CAM 的 acc / confidence / entropy / top10 mass
- 提供 class prototype 和 gallery 的可视化函数

当前它的文字输出主要是 CAM group summary。下一步建议扩展它：

- 自动保存每个 dataset/group/class 的 prototype 图
- 对比 real vs B 的 class-wise prototype
- 输出 class-wise CAM entropy / top10 mass / edge mass

### 7.2 `04_occlusion_sensitivity_executed.ipynb`

这个 notebook 目前只是框架，尚未真正跑 occlusion。

它定义了：

- hot region 遮挡
- cold region 遮挡
- random region 遮挡
- probability drop 汇总函数

但当前没有：

- `occ.csv`
- hot/random/cold drop 结果
- dataset/group 级 occlusion 表

所以当前不能把 occlusion 作为已完成证据。

下一步应补：

```text
pathmnist real_train
pathmnist B
bloodmnist real_train
bloodmnist B
pneumoniamnist real_train
pneumoniamnist B
```

每组 50 到 100 个样本即可。

关键指标：

| 指标 | 含义 |
|---|---|
| `hot_drop` | 遮挡 CAM 最热区域后，正确类概率下降 |
| `random_drop` | 遮挡随机区域后的概率下降 |
| `cold_drop` | 遮挡最低响应区域后的概率下降 |

如果：

```text
hot_drop >> random_drop > cold_drop
```

则说明 CAM 热点有更强判别因果性。

### 7.3 `05_pathmnist_c_low_similarity_visual_analysis_executed.ipynb`

这个 notebook 很有价值，但它分析的是 C，不是 B。

它说明：

- C 可以高置信预测正确
- 但与 real CAM 严重不一致
- 低相似样本中存在 edge-heavy / center-heavy / diffuse / low-hot-mass 多种模式

因此它不是主 baseline 分析，但提供了重要启发：

> 高性能配置也可能依赖与 real-trained 模型不同的空间证据。

这对 B 的意义是：

> 后续不能只用 ACC 判断改进是否有效，还必须看 real-vs-synthetic CAM similarity 和 occlusion 因果性。

---

## 8. 下一步最推荐的算法改进方向排序

### 第一优先：Feature-map Local NCFD

这是最推荐的第一版算法改进。

动机：

- B 是全局 NCFD / NCFM-core
- 现有证据显示 B 的局部关注与 real-trained 不一致
- PathMNIST / BloodMNIST 更依赖局部组织和细胞结构

形式：

```text
L = L_global_NCFD + lambda_local * L_local_NCFD
```

其中：

- `L_global_NCFD` 保持 B 的原始 NCFM 主损失
- `L_local_NCFD` 在 feature map 局部块上计算
- 第一版只在 loss 层实现，不改 `NCFM/NCFM.py` 和 `SampleNet.py`

推荐第一轮消融：

| 实验 | 说明 |
|---|---|
| B | 原始 NCFM-core |
| B + local 2x2 | feature map 二分块 |
| B + local 4x4 | feature map 四分块 |
| B + local 2x2, lambda=0.4 | 轻局部 |
| B + local 2x2, lambda=0.6 | 中等局部 |

第一主数据集：

```text
pathmnist IPC10
```

第二复核：

```text
bloodmnist IPC10
```

全局结构对照：

```text
pneumoniamnist IPC10
```

### 第二优先：CAM-guided Local Weighting

动机：

- real CAM 可以作为弱局部教师
- B 与 real CAM overlap 低
- 可以让局部匹配更重视 real-trained evaluator 关注区域

可能做法：

1. 用 real-trained evaluator 生成 CAM prior
2. 将 CAM prior 转成 spatial weight
3. 对 feature-map local matching 加权

风险：

- 可能过拟合到某一个 evaluator 的 CAM
- CAM 本身不是病灶真值
- 必须先验证 real CAM 的稳定性和 occlusion 因果性

因此它排第二，不建议第一版就上。

### 第三优先：Foreground / Edge-bias Aware Constraint

动机：

- 部分低相似样本出现 edge-heavy / corner-biased
- 可能存在背景、边缘、裁剪 shortcut

可能做法：

- 前景 mask consistency
- 背景区域 CAM mass penalty
- edge/corner bias penalty
- tissue foreground weighted matching

风险：

- edge 不一定是坏东西，PathMNIST real-trained 本身就 edge_mass 高
- 医学组织边界可能是真实判别结构
- 不能粗暴惩罚所有 edge attention

因此它更适合作为辅助项。

### 第四优先：Class-conditional Structure Prototype

动机：

- 低相似样本可能集中于部分类别
- 不同类别可能依赖不同组织/细胞结构

可能做法：

- per-class CAM prototype
- per-class feature prototype
- class-aware local NCFD
- class-wise local weighting

风险：

- 工程更复杂
- 第一版容易把变量引入太多

建议作为第二阶段。

### 第五优先：SSIM / Pixel Structure Regularization

动机：

- 医学图像结构相似性重要

但暂不建议第一版使用。

原因：

- SSIM 引入像素空间约束
- 容易造成平滑
- 会混淆 local NCFD 的贡献
- 可能和 NCFD 表征空间目标冲突

建议：

> 只有在 feature-map local NCFD 已证明有效后，再考虑 SSIM。

---

## 9. 下一步实验设计建议

### 9.1 先补 Occlusion

在改代码前，最应该补的是 occlusion。

目的：

> 确认 CAM 热点是否真的对预测有因果影响。

优先跑：

| Dataset | Group |
|---|---|
| pathmnist | real_train |
| pathmnist | B |
| bloodmnist | real_train |
| bloodmnist | B |

可选：

| Dataset | Group |
|---|---|
| pneumoniamnist | real_train |
| pneumoniamnist | B |

输出：

```text
reports/occlusion/occlusion_samples.csv
reports/occlusion/occlusion_grouped.csv
reports/occlusion/occlusion_summary.md
```

### 9.2 再实现最小 Local NCFD

确认 occlusion 后，再进入代码改动。

第一版只做：

```text
B + feature-map local NCFD
```

不做：

- raw patch
- local token
- SSIM
- MultiScaleSampleNet
- CAM-guided loss

### 9.3 评估指标必须包含可解释性指标

不能只看 ACC。

正式表应该包含：

| 类别 | 指标 |
|---|---|
| 分类性能 | ACC, AUC macro OvR, Macro-F1, Balanced ACC |
| 二分类额外 | Sensitivity, Specificity, AUPRC |
| CAM 一致性 | Pearson, Spearman, Cosine, Top10 IoU, Top10 Dice |
| 空间偏置 | Edge Mass, Center Mass, Corner Mass |
| 因果验证 | Hot Drop, Random Drop, Cold Drop |

只有当方法同时满足：

```text
性能不下降或提升
CAM 更接近 real-trained
低相似高置信样本减少
hot_drop 更合理
```

才能说它真的改善医学蒸馏结构保真。

---

## 10. 版本管理建议

在进入代码改动前，应先做版本冻结。

建议在 `NCFM_medmnist_clean` 中建立清晰版本：

### 10.1 当前 clean + analysis 版本

建议分支：

```text
analysis/medmnist-structure-diagnosis
```

建议 tag：

```text
v0.1-medmnist-clean-cam-analysis
```

这个版本只包含：

- MedMNIST 数据适配
- metrics
- CAM 工具
- notebooks
- 当前分析文档
- 不包含 local NCFD 改动

### 10.2 后续算法改动版本

建议新分支：

```text
exp/local-ncfd-feature-map
```

这个分支再开始改：

- `condenser/compute_loss.py`
- 可选 `condenser/local_ncfd.py`
- config 字段
- tracker 记录 local/global loss

### 10.3 不要直接在当前分析分支上改算法

原因：

- 当前分支是证据链基线
- local NCFD 是新方法
- 两者必须清楚分开，方便回滚、审计、写论文

---

## 11. 当前一句话结论

当前 evidence chain 支持的核心判断是：

> 医学图像蒸馏的关键问题不是单纯提高 NCFM 的全局分布对齐能力，而是让 synthetic data 保住真实模型依赖的局部结构性判别信息。论文式 NCFM-core B 在 PathMNIST 和 BloodMNIST 上已经暴露出高置信但局部结构错位的问题，因此第一优先的改进方向应是 feature-map local NCFD；在此之后再考虑 CAM-guided weighting、foreground/edge-bias regularization 和 class-conditional structure prototypes。

---

## 12. 立即下一步

建议下一步不改 NCFM 算法，先补：

```text
Occlusion sensitivity for real_train vs B
```

最小执行集：

```text
pathmnist real_train
pathmnist B_minmax_ncfm_psi
bloodmnist real_train
bloodmnist B_minmax_ncfm_psi
```

如果 hot-region 遮挡明显比 random/cold 更影响预测，就可以正式进入：

```text
feature-map local NCFD
```

如果 occlusion 不支持 CAM 热点因果性，则应先改进解释分析，而不是急着改 loss。
