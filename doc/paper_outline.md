# WeldMind 论文写作大纲

> 这是一份**论文规划文档**（不是论文正文），用于和 `papaer.docx` 配套迭代。
> 主要回答四件事：**核心主旨**、**实验设计**、**对比 baseline**、**预期结果**。

---

## 1. 核心主旨（One-sentence thesis）

**在焊缝缺陷分类这种"标注昂贵、专家依赖、样本稀缺"的工业场景下，把"生成一段缺陷描述"作为视觉-语言模型（VLM）的辅助目标，能让分类器在极少量标注样本下就达到传统 CNN/ViT 需要数千张样本才能达到的精度。**

把这句话拆成三个可量化的卖点（论文 Abstract 应该都点到）：

1. **小数据可行（Few-shot）**：50 张/类（200 张总量）就能达到 ≥ 95% 准确率；
2. **方法新颖**：加权多任务损失（label tokens 加权），把生成的"理由"当作 implicit supervision；
3. **优于传统强基线**：在同等数据规模下，准确率比 ResNet/ViT/DINOv2 高 X-Y 个点（X、Y 待实验填）。

---

## 2. 写作时反复强调的主张（Take-home messages）

读者读完应该带走的三句话——所有图表、所有实验、所有写作都为这三句服务：

> **TM1 — 标注昂贵问题**：焊缝缺陷的"金标准"标签需要持证 NDT/RT 检查员，每张图按行业价 5–15 美元、且耗时数分钟。任何能把"必需样本量"压低一个数量级的方法都有直接经济价值。
>
> **TM2 — 描述是廉价 supervision**：相比标注一个 label，让 GPT-4o-mini 看图给一段半-技术化描述，每张 ~$0.0003、几秒钟。这是用前沿 LLM "蒸馏" 出的廉价 supervision，可大规模复制。
>
> **TM3 — 生成训练 ≠ 生成评估**：我们的方法在**训练时**学着说描述，但**部署时**只要 label 输出。描述只起到"训练时的脚手架"作用，不增加推理负担。

---

## 3. Overview / 方法框架

```
                ┌─────────────────────────────────────────────────────┐
                │           RIAWELC 24,407 张 (公开数据集)            │
                │     CR | PO | LP | ND     原始 224×224 灰度 PNG     │
                └─────────────────────────────────────────────────────┘
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        │ STAGE 1: 描述生成 (一次性)   │                              │
        │ GPT-4o-mini + structured     │                              │
        │ prompt → 每张图一段          │                              │
        │ inspection report            │                              │
        │ (~1500 字符, JSON-keyed)     │                              │
        └──────────────────────────────┘                              │
                                       │                              │
                                       ▼                              │
        ┌─────────────────────────────────────────────────────────┐   │
        │ STAGE 2: 多任务微调  (Qwen3-VL-4B + LoRA)               │   │
        │                                                         │   │
        │  Input: 图像 + 指令 ("分类 + 描述")                     │   │
        │  Target: "<label>\n\n<description>"                     │   │
        │                                                         │   │
        │  Loss = Σ_t w_t · CE(logits_t, target_t) / Σ_t w_t      │   │
        │       w_t = λ_lbl on label tokens                       │   │
        │       w_t = 1.0   on description tokens                 │   │
        │                                                         │   │
        │  vision tower frozen, LoRA on language projections      │   │
        └─────────────────────────────────────────────────────────┘   │
                                       │                              │
                                       ▼                              │
        ┌─────────────────────────────────────────────────────────┐   │
        │ STAGE 3: 推理 (轻量)                                    │   │
        │  生成 ≤ 16 tokens (label only)                          │   │
        │  正则解析为 4 类之一                                    │   │
        └─────────────────────────────────────────────────────────┘   │
                                                                      │
                                       ◄──────────────────────────────┘
                                       │ 对比: 同数据量下的传统分类器
                                       ▼
                         ┌─────────────────────────┐
                         │ ResNet50 / ViT / DINOv2 │
                         └─────────────────────────┘
```

**核心 contributions**（论文 Section 3 该写的）：

- **C1**：提出 "GPT-generated description as auxiliary supervision" 的 VLM 微调范式（数据是"白嫖" GPT，标注预算只用在 label 上）。
- **C2**：加权 cross-entropy 设计，让 label tokens 的权重 λ 可调（典型 λ=3 表现最好）。
- **C3**：在 RIAWELC 上系统地展示了"小数据 + VLM + 辅助生成 > 大数据 + 传统 CNN"。

---

## 4. 论文结构（建议章节顺序）

| 章节 | 内容 | 重点 |
|---|---|---|
| 1. Introduction | 焊缝检测的工业重要性；标注成本问题；现有方法在 small-N 下不足 | 引出 **TM1** |
| 2. Related Work | (a) 焊缝缺陷分类传统方法 (b) VLM in industrial inspection (c) Knowledge distillation from LLMs | 突出"前人没把 LLM 蒸馏用到 NDT" |
| 3. Method | 数据预处理 + GPT 描述生成 + 加权多任务 loss + 推理 | 数学公式 + 框图 |
| 4. Experimental Setup | RIAWELC 数据 + 子集采样协议 + baselines + 实现细节 | 强调"每条配置都是 fair comparison" |
| 5. Results | 三个层次的实验（见下文 §5） | 突出 **TM2** 和 **TM3** |
| 6. Analysis | 失败案例 + 消融 + λ 敏感度 + 描述质量 vs 分类准确率相关性 | 说明 "为什么" 有效 |
| 7. Conclusion | 回到 TM1/TM2/TM3 + 工业部署建议 + 局限 | 给读者明确的"可以怎么用" |

---

## 5. 实验设计（这是论文的"心脏"）

### 5.1 数据集和划分

- **RIAWELC** 公开数据集：24,407 张 → 训练 15,863（65%）/ 验证 6,102（25%）/ 测试 2,441（10%）
- 训练用 stratified subset：**每类 {50, 200, 1000} 张** = 200 / 800 / 4000 张
- 验证集（6,102）用于调参和 sweep 结果汇报
- 测试集（2,441）只用一次，论文表里报最终数字

### 5.2 三组实验

每组都报 **Top-1 Accuracy + macro F1 + per-class F1**。

#### **EXP-1: 内部消融**（核心实验，证明 TM2）

| 变体 | 训练目标 | λ_lbl |
|---|---|---|
| **V_cls** | label only | n/a |
| **V_joint** | label + description, 等权 | 1.0 |
| **V_weighted** | label + description, label 加权 | **3.0** |

在 3 个 subset size 上各跑一遍 → **9 个 runs**。

**期待图表**：折线图，X = 每类样本数（log），Y = 准确率；3 条线分别是 V_cls / V_joint / V_weighted。

**期待结果**：
- 在 50/类：V_weighted ≈ 95%，V_joint ≈ 93%，V_cls ≈ 88% — **gap 最大、最有说服力**。
- 在 1000/类：三条线接近，但 V_weighted 还能领先 0.5–1 个点。
- 在更大数据上（如果加全量）：三条线收敛到 ~99%，**说明优势在小数据**（对应 TM1）。

#### **EXP-2: 与传统方法对比**（证明 TM1，刚性卖点）

选 3 个具有代表性的强 baseline，在同样的 subset 上训练：

| Baseline | 类型 | 为什么选它 |
|---|---|---|
| **B1: ResNet-50（ImageNet 预训练）** | 经典 CNN | RIAWELC 原论文就用 CNN，工业界用得最多 |
| **B2: ViT-B/16（ImageNet-21k 预训练）** | 经典 Transformer | 公平体现"现代视觉骨干" |
| **B3: DINOv2-ViT-B/14 + Linear Probe** | 自监督视觉表征 | 当前 SOTA 通用视觉表征，no-language 阵营最强 |

每个 baseline 在 {50, 200, 1000}/类上各训练一次，记最佳验证集准确率。

**期待图表**：同一张折线图叠加上 V_weighted 这一条 → 6 条线（3 个 baseline + V_cls + V_joint + V_weighted），或者画两张分开展示。

**期待结果（论文需要这个 gap 才有故事）**：

| 每类样本 | ResNet-50 | ViT-B/16 | DINOv2 (LP) | **Ours (V_weighted)** |
|---:|---:|---:|---:|---:|
| 50 | ~70–80% | ~75–82% | ~83–88% | **~93–96%** ✨ |
| 200 | ~85–90% | ~88–92% | ~90–94% | **~96–98%** |
| 1000 | ~94–97% | ~95–98% | ~96–98% | ~98–99% |

50 张/类的那一行是论文 Abstract 的"卖点数字"。

#### **EXP-3: 分析性实验**（证明"为什么"有效，呼应 TM3）

每一项都是一段 narrative，配一张小图：

- **3a. λ 敏感度**：在 50/类上扫描 λ ∈ {1, 2, 3, 5, 10}，画准确率曲线；预期 λ=3 附近最优。
- **3b. 描述质量 vs 分类准确率**：把 GPT 描述按 ROUGE 自相似度（description vs gen_text 两遍生成的一致性）分箱，看分类准确率是否随描述一致性上升 → 间接验证"高质量描述帮助更大"。
- **3c. 推理时的"开启/关闭描述"**：把 V_weighted 在推理时分别用 (i) max_new_tokens=16 只取 label, (ii) max_new_tokens=512 让它先生成描述再 label。后者如果更好 → "chain-of-thought" 解释；如果相当 → 说明描述只在训练时起作用（**这是 TM3**）。
- **3d. 失败案例**：随机挑 ~10 张错分的图片，人工对比 V_cls vs V_weighted，看 V_weighted 修正了哪些类型的错误（典型期待：CR vs LP 易混淆，描述提示"裂纹的方向性"帮模型区分）。

### 5.3 实现细节（论文要写进 Appendix）

- Qwen3-VL-4B，bf16，flash-attn-2，LoRA r=16/α=32 on language projections，vision tower 冻结
- AdamW，lr=2e-4，cosine schedule，warmup 3%，gradient checkpointing
- per_device_batch=2, grad_accum=4（effective batch=8）
- 单卡 H20-NVLink 96GB，~4.5h 完成全部 9 runs
- Baselines 用相同 GPU、相同 epoch budget 训练，公平比

---

## 6. 期待最强的图（论文 Headline Figure）

```
        准确率 (%)
   100 ┤                                    ●──── Ours V_weighted
       │                              ●─────       (~96 at 50/cls)
    95 ┤                       ●──────
       │                ●──────
    90 ┤         ●─────
       │   ●─────                            ▲──── DINOv2 LP
    85 ┤                            ▲─────────
       │                     ▲──────
    80 ┤              ▲──────                ■──── ViT-B/16
       │       ▲──────              ■─────────
    75 ┤             ■──────────────                 ✦──── ResNet-50
       │       ■──────              ✦──────────
    70 ┤───────────────────✦───────
       │       ✦──────────
    65 ┤
       └─────┴──────────┴──────────┴──────────┴──→ 每类样本数 (log)
           50         200       1000      4000
```

X 轴 log 刻度。读者一眼看到：**所有数据量上 ours 都在最上面，且在 50/类时 gap 最大**——这就是论文的"sales pitch"。

---

## 7. Related Work 要 cite 的关键文献

按主题分组写 Section 2：

**A. 焊缝缺陷分类（领域 baseline）**
- Totino et al. 2022 ICMECE — RIAWELC 数据集原文 (必 cite)
- Perri et al. 2023 Manufacturing Letters — 与 RIAWELC 配套的 CNN 方法 (必 cite)
- Bacioiu et al. 2019 NDT&E Int. — TIG 焊缝缺陷传统 CNN
- Hou et al. 2019 — 焊缝 X-ray 图像深度学习综述

**B. VLM 在工业检测**
- Qwen2-VL / Qwen2.5-VL / Qwen3-VL 技术报告
- LLaVA-Med、SkyEyeGPT 等领域适配的 VLM 工作
- 缺陷检测领域用 CLIP 的工作（PaDiM-CLIP 等）

**C. 用 LLM 蒸馏作为 supervision**
- Self-instruct / Alpaca 范式 (LLaMA 阵营)
- Distilling step-by-step (Hsieh 2023)
- Visual-CoT / SciCap 之类"图像描述当 supervision"

**D. Few-shot industrial vision**
- WinCLIP, ZeroAD 等零/少样本缺陷检测

---

## 8. 论文里 NOT 该做的事

写作时最容易"画蛇添足"，提前给自己设红线：

- ❌ **不报生成质量指标（ROUGE/BLEU）**：val/test 没有 reference 描述；强行算只能算 description vs gen_text 自相似度，意义有限，且容易让审稿人问"如果生成质量不好为啥分类还提升？" — 反而模糊主旨。
- ❌ **不试图论证 GPT 描述本身的正确性**：它就是个 noisy teacher signal，**重点是它有没有帮到分类**，而不是"描述写得多准"。
- ❌ **不做超大模型对比**（GPT-4V、Claude vision 直接零样本分类）：那是另一篇论文的故事，会模糊"小数据微调"这条主线。如果审稿人非要，放一行 in-text 数字即可。
- ❌ **不堆超过 4 个 baseline**：每多一个 baseline = 多一组训练 + 多一段写作，边际收益递减。

---

## 9. 待你定的几个细节

写正文之前最好先和 advisor 对齐：

1. **目标会议/期刊**：影响版式、长度、是否需要双盲。
   - 候选：IEEE Trans. Industrial Informatics / NDT&E International / Computers in Industry / ICCV Workshop
2. **3 个 baseline 是否就这 3 个**：ResNet / ViT / DINOv2 是我推荐的，但如果你想换成 CLIP linear probe / EfficientNet / SwinTransformer 也可以。
3. **是否做 EXP-3 全部 4 项**：3a 和 3c 我强烈推荐，3b 和 3d 可选。
4. **是否需要在 test split (2441 张) 上做最终对比表**：这是论文表 1 必备，但需要把所有 9 个 adapter + 3 个 baseline 都在 test 上跑一遍 eval。
5. **第二轮的 gen_text 字段是否要用上**：可以当作"廉价 augmentation"，在训练时随机 30% 概率换成 gen_text 而非 description，看是否再涨 0.5 个点。如果时间紧可以不做。

---

## 10. 写作时间预估

| 阶段 | 工时 | 备注 |
|---|---|---|
| EXP-1 (9 runs) | 5h (已规划) | notebook 已写好 |
| EXP-2 (3 baselines × 3 sizes = 9 runs) | 8–12h | 需要写 baseline 训练脚本 (PyTorch + timm) |
| EXP-3 (3a 必做, 3c 必做) | 6–8h | λ 扫描 + chain-of-thought 实验 |
| Test split 最终评估 | 2h | 在所有训练好的 checkpoint 上重跑 cell 9 |
| 写作 (8 节) | 25–35h | 含图表绘制和参考文献整理 |
| **总计** | **~50–70h** | 大概 2 周专注工作 |

---

**TL;DR**：论文的"魂"是**"50 张/类的小数据，靠 GPT 描述 + 加权多任务，VLM 微调能赢传统 CNN 10–20 个点"**。所有实验、图表、写作都围绕这个核心结论组织；任何不直接支持这个结论的内容都应该砍掉或挪到 Appendix。
