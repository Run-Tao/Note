---
title: Level 8 深度学习中的线性代数
type: concept
subject: linear-algebra
level: Level 8
tags:
  - mathematics
  - linear-algebra
  - deep-learning
created: 2026-08-23
source: Learning_Roadmap.md Level 8 推荐资料
---

# Level 8 深度学习中的线性代数

> 这是整个知识库的**收官之章**。前 7 个 Level 建立的"数据即向量、变换即矩阵"（Level 1）、"矩阵 = 线性变换"（Level 3）、"内积度量相似度"（Level 4）、"SVD 与低秩近似"（Level 6）在这里全部汇合：神经网络的前向传播是一串矩阵乘法，反向传播是雅可比矩阵的链式乘积，注意力是内积与 softmax 的复合，卷积是一张稀疏的 Toeplitz 矩阵，微调则是一场低秩近似。
>
> 知识主线：**张量与批量计算 → 雅可比矩阵与反向传播 → 谱性质与稳定训练 → 注意力机制 → 卷积的矩阵形式 → 嵌入与检索 → 生成模型 → LoRA 低秩微调**

## 学习目标

完成本章后，你应该能够：

- 看懂神经网络前向 / 反向传播背后的线代结构：前向 = 矩阵乘法复合，反向 = 雅可比矩阵的链式乘积；
- 掌握张量运算、广播、批量矩阵乘法（batch matmul）的形状规则与直觉；
- 用雅可比矩阵写出链式法则的矩阵形式，并解释"为什么反向传播高效"；
- 用谱范数（最大奇异值）解释梯度爆炸 / 消失，理解正交初始化与谱归一化；
- 读懂注意力公式 $Q=XW_Q,\ K=XW_K,\ V=XW_V$、$\mathrm{softmax}(QK^\top/\sqrt{d_k})$，并说清 $\sqrt{d_k}$ 缩放的原因；
- 从 Toeplitz 矩阵视角理解卷积 = 稀疏矩阵乘法，理解感受野的累积；
- 理解词嵌入、向量检索（RAG）、VAE / 扩散模型、LoRA 背后的线代本质。

> [!abstract] 本章的"应用密度"
> 按 config.md 第 5 节要求，每个核心概念都对应一个**具体**的 AI 应用（模型名 + 环节 + 公式出处）。本章是四个方向（ML / DL / CV / NLP）的收敛点——学完它，前面所有 Level 的工具都有了"用武之地"。

---

## 第一部分：张量与批量计算

### 1. 张量：高维数组

**定义**：深度学习语境下的**张量（tensor）**就是"高维数组"：0 维是标量（scalar），1 维是向量，2 维是矩阵，3 维及以上统称张量。每个张量有一个 shape，例如 $(B, T, d)$ 表示"批次大小为 $B$、序列长度 $T$、每个元素是 $d$ 维向量"。

> [!warning] 常见误区
> 误区："张量 = 数学里那个张量（多重线性映射）"。→ 正确：深度学习（PyTorch / TensorFlow）里的 tensor 只是**多维数组**，借用了数学名词，不要求理解张量分析。

**AI 联系**：一条样本是向量（Level 1），一批样本是矩阵，一批"图像 + 通道 + 批量"就是四维张量 $(B, C, H, W)$（CV 的标准格式）；一条句子是一维序列，一批句子是 $(B, T, d)$（NLP 的标准格式）。**张量 = 给数据加"批量"和"结构"两个轴之后的矩阵**。

### 2. 广播运算（broadcasting）

**规则**：两个张量做逐元素运算时，形状从**最后一个维度开始对齐**：每个维度要么相等，要么其中一个是 1，则大小为 1 的维度被"拉伸"到另一个的大小；不满足则报错。

- 例 1（加偏置）：$X$ 形状 $(B, T, d)$，偏置 $\mathbf{b}$ 形状 $(d,)$ → 对齐后 $(B, T, d) + (1, 1, d) \to (B, T, d)$，每个位置都加上同一个 $\mathbf{b}$。这正是线性层 $Y = XW^\top + \mathbf{b}$ 的向量化写法。
- 例 2（外积式相加）：$(B, T, 1) + (B, 1, d) \to (B, T, d)$——让"序列位置"与"特征"两个方向分别广播。

> [!tip] 直觉
> 广播的语义是"把一个数组沿某个轴复制、与另一个对齐后逐元素运算"。**实现上并不真的复制**（用 stride 技巧共享内存），所以它既省代码又省内存。

**应用（具体）**：几乎所有深度学习框架的 LayerNorm / BatchNorm、注意力掩码相加都用广播；Transformer 里把注意力掩码加到 $QK^\top$ 上，形状 $(B, 1, T, T)$ 广播到 $(B, H, T, T)$（多批多头）。

### 3. 批量矩阵乘法（batch matmul）

**定义**：对矩阵乘法增加一个"批量轴"：$X$ 形状 $(B, N, D)$，$W$ 形状 $(D, M)$，则

$$
X \otimes W = (XW) \quad\text{形状}\ (B, N, M)
$$

即对批次里的**每一个样本**执行同一个线性变换 $W$。更一般地，两个张量都可以带批量轴：$(B, N, D) \otimes (B, D, M) \to (B, N, M)$。

**直觉**：矩阵乘法本来就是"批量计算"（Level 1 的旧结论）——一批向量同时被同一个矩阵变换；batch matmul 只是把"一批向量"升级为"一批矩阵"。GPU 上这是最底层的优化算子（GEMM），性能远超 Python 循环，这也是**为什么深度学习的代码全是矩阵运算而不是 for 循环**。

**应用（具体）**：Transformer 前向传播中，Q、K、V 的计算就是对整批序列一次完成：$X$ 形状 $(B, T, d)$，$W_Q$ 形状 $(d, d_k)$，$Q = XW_Q$ 形状 $(B, T, d_k)$——这里 $B$ 维就是批量矩阵乘法在起作用；PyTorch 的 `torch.matmul` / `torch.einsum` 正是这个算子。

### 4. 神经网络的张量视角

把整个网络看成复合函数：设第 $l$ 层是线性变换 + 非线性激活 $\mathbf{h}_l = \sigma(W_l \mathbf{h}_{l-1} + \mathbf{b}_l)$，则

$$
f(\mathbf{x}) = f_L \circ f_{L-1} \circ \cdots \circ f_1(\mathbf{x})
$$

- 前向传播（forward pass）= 按顺序执行这一串矩阵乘法与逐元素非线性；
- 每个线性层就是 Level 1–3 反复强调的"变换即矩阵"；激活函数（ReLU、softmax 等）是唯一的非线性来源；
- 训练的目标是找到一组权重矩阵 $\{W_l\}$ 使损失最小——这正是下一部分要解决的"梯度怎么算"。

---

## 第二部分：反向传播与雅可比矩阵

### 5. 从标量链式法则到多变量

标量情形（Level 0 微积分）：$z = g(y), \ y = f(x) \Rightarrow \frac{dz}{dx} = \frac{dz}{dy}\frac{dy}{dx}$。

神经网络里输入、输出都是向量，**偏导之间如何复合？**答案是：把"每个输出对每个输入的偏导"排成矩阵，链式法则就变成矩阵乘法——这就是雅可比矩阵登场的时刻。

### 6. 雅可比矩阵（Jacobian matrix）

**定义**：映射 $f: \mathbb{R}^n \to \mathbb{R}^m$，其雅可比矩阵 $J_f \in \mathbb{R}^{m \times n}$ 的第 $(i, j)$ 个元素是

$$
(J_f)_{ij} = \frac{\partial f_i}{\partial x_j}
$$

即**第 $i$ 个输出对第 $j$ 个输入的偏导**。$J_f$ 描述"输入扰动 $\Delta \mathbf{x}$ 时输出如何一阶变化"：$f(\mathbf{x} + \Delta\mathbf{x}) \approx f(\mathbf{x}) + J_f \Delta\mathbf{x}$——它是多变量函数在一点处的"最佳线性近似"，这正是线性代数的视角。

> [!example] 线性映射的雅可比是常数矩阵
> 若 $f(\mathbf{x}) = W\mathbf{x}$，则 $J_f = W$。**一个矩阵变换的雅可比就是它自己**——这解释了为什么深度网络里梯度公式中到处是 $W$ 和 $W^\top$。
> 标量损失 $L$（$m=1$）时，$J_L$ 是 $1 \times n$ 行向量，其转置就是梯度 $\nabla L \in \mathbb{R}^n$。约定：本文梯度一律取列向量。

### 7. 链式法则的矩阵形式

**定理（多变量链式法则）**：若 $z = g(f(\mathbf{x}))$，则

$$
\boxed{\ J_{g \circ f}(\mathbf{x}) = J_g\big(f(\mathbf{x})\big) \cdot J_f(\mathbf{x})\ }
$$

两个雅可比**按这个顺序**做矩阵乘法。注意矩阵乘法不交换（Level 1），所以**求导顺序不能随意调换**——这是多变量与标量链式法则的本质区别。

> [!example] 两层网络的梯度
> 设 $\mathbf{y} = W\mathbf{x}$，损失对 $\mathbf{y}$ 的梯度为 $\mathbf{g} = \partial L/\partial \mathbf{y}$（列向量），由链式法则：
> - 对输入：$\dfrac{\partial L}{\partial \mathbf{x}} = W^\top \mathbf{g}$（验证：$y_i = \sum_j W_{ij}x_j$，$\partial L/\partial x_j = \sum_i g_i W_{ij} = (W^\top \mathbf{g})_j$）；
> - 对权重：$\dfrac{\partial L}{\partial W} = \mathbf{g}\,\mathbf{x}^\top$（**外积**，秩至多 1 的矩阵）。
>
> 这两个公式是"手写神经网络"（如 Karpathy 的 micrograd）里每一层都在执行的运算。

### 8. 为什么反向传播 = 高效的链式法则

链式法则给了计算梯度的"自由"，但**求值顺序决定效率**：

- **朴素做法（forward-mode）**：逐个参数计算 $\partial L/\partial W_{ij}$，需要对每个参数做一次前向，成本与参数个数成正比；
- **反向传播（backpropagation，reverse-mode）**：从标量损失出发，一次把梯度向量沿计算图**从后往前**传回；每一层只做"矩阵 × 向量"（向量-雅可比积，VJP），**从不构造完整的雅可比矩阵**（$m \times n$ 太大，也根本不需要）。

> [!tip] 一句话
> 反向传播 = **对"标量损失"这一特殊情形，选择最划算的求值顺序的链式法则**：前向一次、反向一次，梯度全到手。成本 ≈ 前向的 1–2 倍，与参数个数无关。
> 应用（具体）：PyTorch 的 `autograd`、TensorFlow 的 `GradientTape`、Karpathy micrograd 全部是 reverse-mode 的实现；训练 175B 参数的 GPT 级模型正是靠这一点才可能完成梯度计算。

---

## 第三部分：权重矩阵的谱性质

### 9. 谱范数与 Lipschitz 常数

**定义（回顾 Level 6）**：矩阵 $A$ 的**谱范数（spectral norm）**是其最大奇异值

$$
\|A\|_2 = \sigma_{\max}(A) = \max_{\|\mathbf{x}\|_2 = 1} \|A\mathbf{x}\|_2
$$

**几何直觉**：$A$ 把单位球映成椭球，谱范数就是**最长半轴的长度**——$A$ 最多能把一个单位向量拉长多少。奇异值正是椭球各半轴长（Level 6 的 SVD 结论）。

**Lipschitz 常数**：映射 $f$ 的 Lipschitz 常数 $L$ 满足 $\|f(\mathbf{u}) - f(\mathbf{v})\| \le L \|\mathbf{u} - \mathbf{v}\|$。对线性映射 $f(\mathbf{x}) = W\mathbf{x}$，**最小的 Lipschitz 常数恰是 $\sigma_{\max}(W)$**。多个层复合时（回顾 Level 6 的奇异值乘法界）：

$$
\sigma_{\max}(W_L \cdots W_1) \le \sigma_{\max}(W_L) \cdots \sigma_{\max}(W_1)
$$

### 10. 梯度爆炸 / 消失的谱解释

深度线性网络 $\mathbf{h}_L = W_L \cdots W_1 \mathbf{x}$ 的梯度（第二部分）正比于矩阵乘积 $W_1^\top \cdots W_L^\top$。由第 9 节的乘积界：

- 若**每层** $\sigma_{\max}(W_l) > 1$：乘积的谱范数以指数增长 → **梯度爆炸（exploding gradients）**；
- 若**每层** $\sigma_{\max}(W_l) < 1$：乘积趋于零矩阵 → **梯度消失（vanishing gradients）**。

数值感：$1.05^{100} \approx 131$，$0.95^{100} \approx 0.006$——100 层之后要么放大百倍要么缩到零。ReLU 网络略有不同：设 $z_l = W_l \mathbf{h}_{l-1}$，该层雅可比是 $J_l = \operatorname{diag}(\mathbf{1}_{\{z_l > 0\}})\, W_l$——相当于**保留 $W_l$ 中对应激活非零的行、其余行清零**，谱被激活的子集缩放，但指数级的结论不变。

> [!example] 三个"谱处方"
> - **梯度裁剪（gradient clipping）**：把梯度向量的范数压到阈值，直接截断爆炸；
> - **残差连接（residual connection）**：$h_{l+1} = h_l + F(h_l)$ 的雅可比是 $I + J_F$，恒等部分谱为 1，缓解消失——ResNet、Transformer 都在用；
> - **谱归一化（Spectral Normalization，SN-GAN，Miyato et al. 2018）**：每层除以 $\sigma_{\max}(W)$ 强制 Lipschitz 常数 ≤ 1，用于约束 GAN 判别器的平滑性，稳定对抗训练。

### 11. 正交初始化回顾

**正交初始化（orthogonal initialization）**：把 $W$ 初始化为正交矩阵（或正交矩阵的缩放）。正交矩阵奇异值全为 1（Level 4），因此 $\sigma_{\max} = 1$，前向传播不放大信号、反向传播不放大梯度——在深度线性网络中能严格避免指数级缩放（Saxe et al., 2013）。

对比：Xavier / He 初始化不做"奇异值归一"，而是控制**随机矩阵谱的方差**，让信号在层间"方差守恒"——同一思想的统计版本。

---

## 第四部分：注意力机制

### 12. Q / K / V 投影的矩阵形式

输入 $X \in \mathbb{R}^{T \times d}$（$T$ 个 token，每个 $d$ 维），注意力机制先做三个**线性投影**（出处："Attention Is All You Need"，Vaswani et al., 2017，公式 1）：

$$
Q = XW_Q, \qquad K = XW_K, \qquad V = XW_V
$$

其中 $W_Q, W_K \in \mathbb{R}^{d \times d_k}$，$W_V \in \mathbb{R}^{d \times d_v}$ 是可学习权重矩阵。

**直觉**：$Q$（query，查询）问"我在找什么"，$K$（key，键）答"我是什么"，$V$（value，值）是"我携带的内容"。三个投影 = 把每个 token 的表示映射到三个不同的子空间（Level 3：矩阵就是线性变换；三个不同的变换对应三种语义视角）。整个前向 = 三次批量矩阵乘法（第一部分）。

### 13. softmax(QKᵀ/√d_k) 与缩放的原因

**注意力公式**（完整版）：

$$
\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

- $QK^\top \in \mathbb{R}^{T \times T}$：第 $(i,j)$ 个元素是 $q_i \cdot k_j$，即第 $i$ 个 token 与第 $j$ 个 token 的**内积相似度矩阵**；
- 除以 $\sqrt{d_k}$ 后按行做 softmax（每行归一化成非负且和为 1 的概率分布）；
- 最后右乘 $V$：每行的输出 = 用相似度分布对 $V$ 的各行做加权平均。

**为什么缩放 $\sqrt{d_k}$**（出处：论文 3.2.1 节原文的动机 + 概率论证）：若 $q$、$k$ 的每个分量是独立的零均值、单位方差随机变量，则内积 $q \cdot k$ 的方差为 $d_k$（和的方差 = 方差之和），标准差为 $\sqrt{d_k}$。当 $d_k$ 很大（如 512）时，不缩放的内积动辄几十的量级，softmax 输入进入**饱和区**：最大的那个元素压倒一切，输出接近 one-hot。而 softmax 的雅可比是

$$
\frac{\partial\,\mathrm{softmax}(\mathbf{z})}{\partial \mathbf{z}} = \operatorname{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top
$$

当 $\mathbf{p}$ 接近 one-hot 时这个矩阵**趋近于零矩阵** → 梯度消失，模型学不动。除以 $\sqrt{d_k}$ 把内积方差拉回 $O(1)$，softmax 保持在"有梯度"的区域。**缩放是梯度健康度问题，不只是数值精度问题。**

### 14. 内积相似度的线代本质

$q \cdot k = \|q\|\|k\|\cos\theta$（Level 1 / 4）：内积同时编码"方向对齐程度"（$\cos\theta$）与"长度"（$\|q\|\|k\|$）。注意力用的就是**内积作相似度**——与余弦相似度同源，只是不除范数（缩放因子由 $\sqrt{d_k}$ 承担）。

两个结构性的观察：

- 经过 softmax 的注意力矩阵是**行随机矩阵**（非负、行和为 1），所以"注意力输出 = 对 $V$ 各行的凸组合"——输出停留在 $V$ 的行张成的空间中（Level 3 列空间视角）；
- 未经过 softmax 的 $QK^\top = XW_QW_K^\top X^\top$ 的秩**不超过 $d_k$**——相似度矩阵天然是低秩的（Level 6 的回响）。

**多头注意力（multi-head）**：把 $d$ 维切成 $h$ 个块，每块一组 $(W_Q, W_K, W_V)$ 独立做注意力再拼接——块对角结构，让不同头关注不同的子空间模式。

> [!example] 具体应用
> Transformer 编码器（BERT、GPT、ViT）每一层的自注意力：$Q=K=V=X$（同一个输入序列），公式即第 13 节；交叉注意力（seq2seq、扩散模型的条件注入）则让 $K,V$ 来自另一个序列。**整个注意力机制 = 线性投影 + 内积 + 归一化 + 凸组合**，没有超出本书的任何一个概念。

---

## 第五部分：卷积的矩阵形式与位置编码

### 15. 一维卷积 = Toeplitz 矩阵

一维卷积（valid 模式，核长 3）：$y_i = w_0 x_i + w_1 x_{i+1} + w_2 x_{i+2}$。写成矩阵 $\mathbf{y} = C\mathbf{x}$：

$$
C = \begin{pmatrix} w_0 & w_1 & w_2 & 0 & 0 \\ 0 & w_0 & w_1 & w_2 & 0 \\ 0 & 0 & w_0 & w_1 & w_2 \end{pmatrix}
$$

- **Toeplitz 矩阵（Toeplitz matrix）**：每条（从左上到右下的）对角线上元素全相等——卷积核"滑动"的本质；
- 每行只有核长 $K$ 个非零：**稀疏、带状**；
- 平移等变性：卷积与"整体移位"可交换（忽略边界）——这正是 CNN 对位置变化鲁棒的线代根源。

二维多通道卷积对应**块 Toeplitz 结构**（BTTB），逐块看还是这个模式。

### 16. 卷积 = 稀疏矩阵乘法与实现

> [!warning] 常见误区
> 误区："卷积就是卷积核在图上滑动，跟矩阵乘法无关。" → 正确：卷积**就是**矩阵乘法，只不过对应的矩阵是稀疏 Toeplitz 矩阵。GPU 上的实现正是"矩阵乘法化"。

实际实现（cuDNN 等）从不显式构造 $C$（$C$ 太大，内存爆炸），而是用 **im2col**：把每个感受野 patch 展开成一行，堆成矩阵 $X_{\text{col}}$，然后执行一次标准 GEMM：$Y = W_{\text{col}} X_{\text{col}}$，其中 $W_{\text{col}}$ 是"输出通道 × 核尺寸×输入通道"的权重矩阵。**卷积的前向 = 一次（大）矩阵乘法**，与批量矩阵乘法共用同一套优化过的 GEMM 内核。

### 17. 感受野（receptive field）

**定义**：输出中一个元素所依赖的输入区域大小。卷积矩阵每行只有 $K$ 个非零，但**多层相乘后"带宽"逐层累积**：$L$ 层核长 $K$、步长 1 的网络，感受野

$$
\mathrm{RF} = 1 + L(K - 1)
$$

每多一层，每个输出元素能"看到"的输入范围向两边各扩 $(K-1)/2$。空洞卷积（dilated convolution）把核长等效为 $(K-1)d + 1$（$d$ 为膨胀率），感受野增长更快。

**应用（具体）**：ResNet 深层输出感受野覆盖整张图，是它能做分类的关键；空洞卷积用于 DeepLab 语义分割（在不大幅增加参数的前提下扩大感受野）。

### 18. 位置编码（positional encoding）

**问题**：自注意力中 $QK^\top$ 只依赖 token 两两之间的内积，**对 token 的顺序置换是等变的**——把一句话的词序打乱，注意力输出不变（只换行）。模型必须被"喂"位置信息。

**正弦编码**（Vaswani et al., 2017，第 3.5 节）：对位置 $\mathrm{pos}$、维度 $2i$ 与 $2i+1$：

$$
\mathrm{PE}(\mathrm{pos}, 2i) = \sin\!\left(\frac{\mathrm{pos}}{10000^{2i/d}}\right), \qquad
\mathrm{PE}(\mathrm{pos}, 2i+1) = \cos\!\left(\frac{\mathrm{pos}}{10000^{2i/d}}\right)
$$

**线代本质**：令 $\omega_i = 10000^{-2i/d}$，每一对 $(\sin(\mathrm{pos}\,\omega_i),\ \cos(\mathrm{pos}\,\omega_i))$ 是单位圆上的点。位置平移 $k$ 后，由三角恒等式 $\sin(a+b)=\sin a\cos b+\cos a\sin b$、$\cos(a+b)=\cos a\cos b-\sin a\sin b$ 得

$$
\begin{pmatrix}\sin((\mathrm{pos}+k)\omega_i)\\ \cos((\mathrm{pos}+k)\omega_i)\end{pmatrix}
= \underbrace{\begin{pmatrix}\cos(k\omega_i) & \sin(k\omega_i)\\ -\sin(k\omega_i) & \cos(k\omega_i)\end{pmatrix}}_{R(k\omega_i)}
\begin{pmatrix}\sin(\mathrm{pos}\,\omega_i)\\ \cos(\mathrm{pos}\,\omega_i)\end{pmatrix}
$$

$R(k\omega_i)$ 是二维旋转矩阵（正交矩阵，行列式为 $1$），所以 $\mathrm{PE}_{\mathrm{pos}+k} = R(k)\,\mathrm{PE}_{\mathrm{pos}}$，其中 $R(k)$ 是**分块对角矩阵**、每块是一个旋转——**相对位移 = 一个线性变换（旋转）**，模型容易学会"按相对位置加权"，这正是论文选择正弦编码的动机。学习式位置嵌入则是查一张"位置 × 维度"的嵌入表（第 19 节的矩阵查找思路）。

---

## 第六部分：嵌入与检索

### 19. 词嵌入矩阵

**定义**：**嵌入矩阵（embedding matrix）** $E \in \mathbb{R}^{V \times d}$（$V$ 为词表大小，$d$ 为嵌入维度）。第 $i$ 个词的嵌入 = 第 $i$ 行（或列），等价于 one-hot 向量 $\mathbf{e}_i$ 乘嵌入矩阵：$\mathbf{w}_i = E^\top \mathbf{e}_i$（回顾 Level 1："一条样本 = 一个特征向量"；查表 = 稀疏矩阵乘法）。

- **word2vec**（Mikolov et al., 2013）：用词向量与上下文向量的**内积**建模共现概率；理论结果（Levy & Goldberg, 2014）显示负采样训练等价于对（平移后的）PMI 矩阵做隐式分解——**又是 SVD / 矩阵分解**（Level 6）；
- **线性语义结构**：词向量满足"国王 − 男人 + 女人 ≈ 女王"——嵌入空间里语义关系对应向量加减（Level 3：嵌入空间是向量空间，加法封闭）；
- **应用（具体）**：BERT / GPT 的第一层就是 token embedding 查表（$V \approx 50$k、$d \approx 768$ ～ 4096 的嵌入矩阵）。

### 20. 向量数据库与余弦相似度（RAG）

**检索任务**：给定查询，从大规模语料中找出最相关的片段。把查询和语料都嵌入成向量后，检索 = **高维空间最近邻搜索**。

**余弦相似度（cosine similarity）**（回顾 Level 4 的角度与投影）：

$$
\cos(\mathbf{q}, \mathbf{k}) = \frac{\mathbf{q} \cdot \mathbf{k}}{\|\mathbf{q}\|\,\|\mathbf{k}\|}
$$

它与**向量长度无关、只比方向**——归一化的内积。实际系统常把嵌入归一化（$\|\mathbf{q}\| = \|\mathbf{k}\| = 1$），此时余弦相似度 = 内积，可直接用高效的矩阵乘法批量算相似度（又回到批量 matmul）。

**应用（具体）——RAG（检索增强生成）**：① 用 embedding 模型（如 sentence-transformers / OpenAI embedding API）把文档切块嵌入；② 存入**向量数据库**（FAISS、Milvus 等），用近似最近邻（ANN：HNSW 图索引、IVF-PQ 量化）检索 top-k——之所以用"近似"，是因为精确最近邻在高维是昂贵的（维度灾难，Level 3）；③ 把 top-k 片段拼进 prompt 交给 LLM 生成。整条流水线 = 嵌入（线性投影）+ 内积相似度 + 近邻搜索，没有超出 Level 1–4 的概念。

---

## 第七部分：生成模型中的线性代数（简短直觉）

### 21. VAE：协方差与重参数化

变分自编码器（VAE）让编码器输出隐变量 $\mathbf{z}$ 的分布参数：均值 $\boldsymbol{\mu}$ 与协方差 $\Sigma$，假设 $\mathbf{z} \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma)$（回顾 Level 5 / 7：高斯分布完全由 $(\boldsymbol{\mu}, \Sigma)$ 决定）。

**重参数化技巧（reparameterization trick）**：

$$
\mathbf{z} = \boldsymbol{\mu} + \Sigma^{1/2}\boldsymbol{\varepsilon}, \qquad \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, I)
$$

**线代本质**：标准高斯噪声 $\boldsymbol{\varepsilon}$ 经**仿射变换**（线性映射 + 平移）变成目标高斯——随机性全部藏在 $\boldsymbol{\varepsilon}$ 里，$\boldsymbol{\mu}$ 与 $\Sigma$ 的梯度可以正常反传（否则"从分布采样"这一步不可导）。常见简化：$\Sigma$ 取**对角矩阵**，则 $\mathbf{z}_i = \mu_i + \sigma_i \varepsilon_i$，各维独立缩放（Level 5 对角化的廉价版）；需要完整协方差时用 Cholesky 分解 $\Sigma = LL^\top$（Level 6）。VAE 的 KL 散度项里还出现 $\operatorname{tr}(\Sigma)$ 与 $\log\det\Sigma$——行列式（Level 3）在概率里再次登场。

### 22. 扩散模型中的线性变换

扩散模型（DDPM，Ho et al., 2020）的前向过程逐步加噪：

$$
\mathbf{x}_t = \sqrt{\bar\alpha_t}\,\mathbf{x}_0 + \sqrt{1 - \bar\alpha_t}\,\boldsymbol{\varepsilon}, \qquad \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, I)
$$

**线代本质**：这是**信号与噪声的线性组合**（按方差做"凸组合"：两项系数平方和为 1），所以边际分布 $\mathbf{x}_t \sim \mathcal{N}(\sqrt{\bar\alpha_t}\,\mathbf{x}_0,\ (1 - \bar\alpha_t)I)$——一次加噪就是"缩放数据 + 平移噪声的方差"，每一步都是对标准高斯的线性变换。逆向过程学一个网络 $\boldsymbol{\varepsilon}_\theta(\mathbf{x}_t, t)$ 预测噪声；分数（score）$\nabla_{\mathbf{x}_t}\log p(\mathbf{x}_t) \propto -\boldsymbol{\varepsilon}$ 也是一个线性量。Stable Diffusion 的潜在扩散模型（LDM）先在 VAE 的**线性编码空间**里做扩散——低维线性子空间让生成更高效（Level 3 子空间思想）。

---

## 第八部分：LoRA 与低秩微调

### 23. 低秩近似 W₀ + BA

**回顾 Level 6**：任意矩阵可用 SVD 截断成"秩 $r$ 近似"，秩 = 矩阵携带的有效信息量（奇异值衰减时截断损失很小）。LoRA 把这一思想用到微调上：

**LoRA**（Hu et al., 2021，"LoRA: Low-Rank Adaptation of Large Language Models"）：

$$
W' = W_0 + \frac{\alpha}{r} B A
$$

其中 $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ 是**冻结**的预训练权重，$B \in \mathbb{R}^{d_{\text{out}} \times r}$、$A \in \mathbb{R}^{r \times d_{\text{in}}}$ 是可训练的小矩阵，$r \ll \min(d_{\text{out}}, d_{\text{in}})$（典型 $r = 8$）。

**线代原理**：

- $\operatorname{rank}(BA) \le r$：微调产生的更新矩阵**秩不超过 $r$**，即更新被限制在一个低维子空间里；
- 理论基础：大模型的微调存在**低内在维度**（Aghajanyan et al., 2020）——预训练已经把大部分信息压进权重，微调只需在少数方向修正；
- **训练省资源**：冻结 $W_0$，只有 $BA$ 的参数量是 $r(d_{\text{out}} + d_{\text{in}})$，远小于 $d_{\text{out}} d_{\text{in}}$（优化器状态也大幅缩减）；
- **推理零开销**：$W'\mathbf{x} = W_0\mathbf{x} + \frac{\alpha}{r}B(A\mathbf{x})$，可提前合并成单矩阵 $W_{\text{merged}} = W_0 + \frac{\alpha}{r}BA$，推理时与普通模型完全一样。

**应用（具体）**：LLaMA 系列微调、QLoRA（Dettmers et al., 2023）把 $W_0$ 量化到 4-bit（NF4，量化本身是逐元素/逐块的标量线性映射）再挂 LoRA 适配器，单卡即可微调 65B 模型；SVD 截断压缩权重、剪枝同理——**模型压缩与参数高效微调共享同一个数学内核：低秩近似**。

---

## 常见误区汇总

1. **"张量是数学里的张量"** → 深度学习里的 tensor 只是**高维数组**，与数学张量（多重线性映射）无关；
2. **"反向传播 = 对每个参数分别用链式法则求导"** → 那是 forward-mode（成本 ∝ 参数个数）；反向传播一次把**梯度向量**传回（成本 ∝ 网络大小），且从不构造完整雅可比（$f: \mathbb{R}^n \to \mathbb{R}^m$ 的雅可比是 $m \times n$，一般不是方阵）；
3. **"多变量链式法则 = 标量乘法推广"** → 是**雅可比矩阵乘积**，且**顺序不能交换**（$J_g \cdot J_f \neq J_f \cdot J_g$）；
4. **"谱范数 = 最大特征值"** → 谱范数是**最大奇异值** $\sigma_{\max}$（对非对称矩阵一般 $\neq$ 谱半径 $|\lambda_{\max}|$，且 $\sigma_{\max} \ge |\lambda_{\max}|$）；
5. **"除以 √d_k 只是为了数值稳定 / 防止溢出"** → 主要是**方差归一**：把内积方差从 $d_k$ 拉回 $O(1)$，避免 softmax 饱和导致的**梯度消失**；
6. **"注意力是'纯粹'的矩阵运算，所以是线性的"** → Q/K/V 投影是线性的，但 **softmax 使整体非线性**；注意力矩阵是行随机矩阵，输出是对 $V$ 行的凸组合；
7. **"卷积和矩阵乘法是两回事"** → 卷积就是**稀疏 Toeplitz 矩阵乘法**，GPU 实现（im2col + GEMM）就是把它变成矩阵乘法；
8. **"LoRA 只是省显存的工程技巧"** → 数学本质是**低秩假设**：微调更新集中在秩 ≤ r 的子空间；合并后推理零额外开销。

---

## 知识结构图

```mermaid
graph TD
    A[张量与批量计算] --> B[批量矩阵乘法<br/>前向传播 = 矩阵复合]
    B --> C[线性层 + 激活<br/>深度复合函数]
    C --> D[反向传播<br/>reverse-mode 链式法则]
    D --> E[雅可比矩阵<br/>J_f = ∂f_i/∂x_j]
    E --> F[链式法则矩阵形式<br/>J = J_g · J_f]
    F --> G[梯度爆炸 / 消失<br/>∏ σ_max(W_l)]
    G --> H[谱范数 σ_max<br/>Lipschitz 常数]
    H --> I[正交初始化 / 谱归一化<br/>稳定训练]
    B --> J[注意力机制<br/>Q=XW_Q, K=XW_K, V=XW_V]
    J --> K[softmax(QKᵀ/√d_k)<br/>缩放 = 方差归一]
    C --> L[卷积的矩阵形式]
    L --> M[Toeplitz 矩阵<br/>稀疏、带状]
    M --> N[感受野 + im2col]
    B --> O[嵌入矩阵 E<br/>查表 = one-hot × E]
    O --> P[余弦相似度<br/>向量数据库 / RAG]
    E --> Q[VAE 重参数化<br/>z = μ + Σ^½ ε]
    H --> R[扩散加噪<br/>x_t = √ᾱ_t x_0 + √(1−ᾱ_t) ε]
    Q --> S[LoRA 低秩微调<br/>W₀ + BA, rank ≤ r]
    R --> S
    S --> T[SVD / 低秩近似<br/>Level 6 回顾]
```

## 思考问题

1. 为什么批量矩阵乘法比 Python for 循环快几个数量级？"矩阵化"到底省掉了什么？
2. 用雅可比链式法则推导：$y = W\mathbf{x}$ 时，为什么 $\partial L/\partial \mathbf{x} = W^\top \mathbf{g}$ 而不是 $W\mathbf{g}$？转置从哪里来？
3. 若一个 50 层网络的每层谱范数都是 1.1，粗略估计梯度会放大多少倍？这对应第 10 节的哪个结论？
4. 不缩放时 $d_k = 512$ 的 $q \cdot k$ 标准差是多少？softmax 会进入什么状态，为什么梯度会消失（写出 softmax 的雅可比）？
5. 卷积矩阵每行为什么只有 $K$ 个非零？多层卷积后"带宽"如何累积成感受野公式 $1 + L(K-1)$？
6. 为什么正弦位置编码能做到"相对位置可学习"？$R(k)$ 是什么结构的矩阵（第 18 节）？
7. **收官回顾**：回顾 Level 1–8，说出 Transformer 的注意力机制里同时用到了哪些旧概念（矩阵乘法、内积/角度、投影、行随机矩阵、低秩……）。
8. **收官思考**：Level 1 说"数据即向量、变换即矩阵"。遍历本书：哪些 Level 的概念在 LoRA（$W_0 + BA$）里同时出现？这说明了什么？

## 总结

**Level 8 的知识网络**：

**张量与批量计算（批量 matmul）→ 前向传播 = 矩阵复合 → 雅可比矩阵 → 链式法则矩阵形式 → 反向传播（高效 reverse-mode）→ 谱范数 σ_max（Lipschitz）→ 梯度爆炸/消失的谱解释 → 正交初始化/谱归一化 → 注意力（Q/K/V 投影 + 内积 + softmax 缩放）→ 卷积（Toeplitz 稀疏矩阵）→ 感受野 → 位置编码（旋转矩阵）→ 嵌入（查表 = one-hot × E）→ 余弦相似度检索（RAG）→ VAE 重参数化（协方差）→ 扩散加噪（线性组合）→ LoRA（低秩近似）**

一句话总结：**深度学习是"用线性代数的语言写出来的微分学"——前向是一串矩阵乘法，反向是一串雅可比乘积，稳定性由谱控制，表达能力由秩与维度决定。** Level 8 没有引入任何"新的线性代数"，它只是证明：**前面 Level 1–7 的每一个概念，最终都在现代 AI 系统的心脏里工作。**

> [!note] Level 0–8 收官
> 从 Level 0 的数学语言，到 Level 1 的"数据即向量、变换即矩阵"，再到 Level 8 的 Transformer、CNN、LoRA——这条主线已经闭环：**线代不是深度学习的"前置课程"，而是它自身的语法**。之后的学习（论文、源码、新架构）本质上是"用这套语法读新的句子"。回顾 [[Learning_Roadmap]] 的进度表，更新 Level 8 状态，并把任何遗留疑问按 config.md 规范沉淀回笔记。

> [!question] 未解决
> 问题：注意力缩放因子为什么取精确的 $1/\sqrt{d_k}$，而不是根据训练后 $q,k$ 的实际分布（范数、方差）做更精细的估计？涉及概念：注意力缩放、softmax 饱和、谱性质。我的尝试：方差分析给出 $\sqrt{d_k}$，但该推导假设分量独立同分布且方差为 1，训练后的 Q/K 投影不满足此假设。状态：待解决。

## 推荐资料

- **《深度学习》（花书，Goodfellow et al.）第 2 章**——难度：入门–中等。理由：**必读**。把本书的线代内容与 DL 术语（张量、矩阵运算、范数、特征分解、SVD）权威对照，是衔接线代与 DL 的标准文本。
- **Karpathy micrograd / "Let's build GPT" 系列**——难度：中等（需会读 Python）。理由：用几百行代码手写反向传播与 GPT 的前向/反向，把本文所有矩阵公式变成可运行的代码，是"看得见的链式法则"。
- **3Blue1Brown《神经网络》系列**——难度：零基础。理由：反向传播与梯度下降的几何直觉，配合本笔记第一部分服用效果最佳。
- **"Attention Is All You Need"（Vaswani et al., 2017）第 3 节**——难度：进阶。理由：注意力公式、缩放动机（3.2 节）与正弦位置编码（3.5 节）的**原始出处**，正文短，值得精读矩阵部分。
- **LoRA 论文（Hu et al., 2021）**——难度：中等。理由：低秩微调的原始论文，数学轻、正文短，直接呼应 Level 6 的 SVD 与低秩近似。
- **The Matrix Calculus You Need for Deep Learning（Parr & Howard，选读）**——难度：进阶。理由：雅可比矩阵、梯度布局惯例（行/列约定）的完整梳理，适合消除"转置放哪边"的混乱。

> 若遇到解决不了的疑问，请在笔记中用 `> [!question]` 标记，按 config.md 的规范跟踪解决。

## Related Notes

- [[Level7_Linear_Algebra_in_ML|Level 7 机器学习中的线性代数]] —— 前置基础
- [[Linear Algebra MOC]] —— 知识库总览
- [[Learning_Roadmap]] —— 学习路线
- [[Glossary]] —— 术语表
- [[Mathematical Modeling MOC]] —— 数学建模知识库总览（交叉引用）
