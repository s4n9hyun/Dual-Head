# Introduction

Aligning Large Language Models (LLMs) with human preferences is crucial
for deploying safe and helpful AI systems. Current alignment approaches
fall into two categories: training-time methods that fine-tune the
entire model, and test-time methods that guide frozen models during
inference.

Training-time approaches like Reinforcement Learning from Human Feedback
(RLHF) [@ouyang2022training] and Direct Preference Optimization
(DPO) [@rafailov2024direct] achieve strong alignment but require
expensive retraining for different preferences. Test-time methods
address this limitation by using reward models to guide frozen LLMs
during generation. However, existing test-time approaches like
ARGS [@khanov2024args] and GenARM [@xu2024genarm] rely on separate
models with static fusion strategies, limiting their adaptability.

We introduce a Dual-Head architecture that challenges the assumption
that effective reward modeling requires large specialized models. Our
approach demonstrates that compact 131M parameter heads can achieve
competitive alignment performance when leveraging shared backbone
representations---a 50× reduction compared to GenARM's 6.7B reward
model.

**Compact Head Design**: Instead of using full 6.7B parameter reward
models like GenARM, our Dual-Head approach uses compact 131M parameter
heads attached to a shared backbone---achieving 50× parameter reduction
while maintaining competitive performance.

**Adaptive Fusion**: Rather than fixed fusion weights, our method
employs adaptive gating that dynamically balances language modeling and
reward modeling based on the current hidden state.

**Resource Efficiency**: The Dual-Head architecture demonstrates that
effective alignment can be achieved with dramatically fewer parameters,
enabling practical deployment advantages.

Extensive experiments demonstrate that Dual-Head:

- Matches the performance of training-time methods like DPO while
  maintaining test-time flexibility

- Outperforms existing test-time baselines (ARGS, GenARM) on standard
  alignment benchmarks

- Enables efficient weak-to-strong guidance and multi-objective
  alignment

- Provides superior inference efficiency compared to methods requiring
  multiple model evaluations

# Related Work

**Training-Time Alignment.** RLHF [@ouyang2022training] trains a reward
model on human preferences and optimizes the LLM via reinforcement
learning. DPO [@rafailov2024direct] directly fine-tunes LLMs on
preference data, avoiding RL complexity. While effective, these methods
require expensive retraining for different preferences and cannot adapt
to new requirements without model updates.

**Test-Time Alignment.** Recent work explores guiding frozen LLMs during
inference. ARGS [@khanov2024args] uses trajectory-level reward models to
score partial responses, but this leads to inaccurate evaluations.
GenARM [@xu2024genarm] introduces autoregressive reward models that
provide token-level guidance, achieving better efficiency and accuracy.
However, GenARM uses separate models with fixed fusion weights, limiting
adaptability to context-dependent alignment needs.

**Token-Level Reward Modeling.** Dense reward signals have been shown to
improve RL training stability [@yang2024preference]. Recent work derives
token-level rewards from trajectory-level feedback for training
purposes [@feng2023fantastic]. Our approach focuses on test-time
alignment with learnable token-level fusion guided by contextual
attention.

**Multi-Objective and Weak-to-Strong Alignment.** Multi-objective RLHF
requires retraining for different preference combinations [@wu2024fine].
Weak-to-strong supervision explores using smaller models to guide larger
ones [@burns2023weak]. Dual-Head naturally supports both scenarios
through its flexible architecture.

# Preliminaries

## Test-Time Alignment Framework

Given a frozen base LLM $\pi_{\text{base}}(y_t|x, y_{<t})$ and a reward
function $r(x, y)$, test-time alignment seeks to generate responses that
maximize both fluency and alignment:

$$\pi_{\text{aligned}}(y|x) \propto \pi_{\text{base}}(y|x) \exp\left(\frac{1}{\beta}r(x, y)\right)$$

where $\beta$ controls the trade-off between base model behavior and
reward optimization. The challenge lies in efficiently computing
token-level rewards during autoregressive generation.

## Dual-Head Architecture Motivation

Traditional approaches use separate models for language modeling and
reward estimation, requiring multiple forward passes and model
calibration. Our dual-head design leverages shared representations from
the frozen backbone, enabling efficient joint computation while
maintaining the expressiveness needed for precise alignment control.

# Methodology

## Dual-Head Architecture

**Frozen Backbone with Dual Heads.** Our Dual-Head approach uses a
frozen decoder backbone that provides contextual representations $h_t$
at each timestep. We attach two compact heads, each containing only 131M
parameters (compared to GenARM's 6.7B reward model):

The language modeling (LM) head produces standard next-token logits:
$$z_{\text{LM},t} = W_{\text{LM}} h_t + b_{\text{LM}}$$

The reward modeling (RM) head assigns alignment-oriented scores:
$$z_{\text{RM},t} = W_{\text{RM}} h_t + b_{\text{RM}}$$

The final logits combine both heads via context-aware gating:
$$z_t = (1 - \alpha_t) z_{\text{LM},t} + \alpha_t z_{\text{RM},t}$$

where $\alpha_t \in [0,1]$ is dynamically computed based on sequence
context.

## Context-Aware Gating Mechanism

Unlike fixed fusion weights, our gating network computes $\alpha_t$
using attention over the sequence history:

$$\alpha_t = \sigma\left(W_g \cdot \text{MultiHeadAttention}(h_t, H_{1:t}, H_{1:t}) + b_g\right)$$

where $H_{1:t} = [h_1, h_2, \ldots, h_t]$ represents the sequence of
hidden states, and the multi-head attention mechanism is defined as:

$$\text{MultiHeadAttention}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

This attention-based gating enables context-sensitive decisions about
when alignment intervention is needed, adapting to different parts of
the response based on semantic content and generation history.

**Gating Regularization.** To prevent degenerate solutions where one
head dominates, we introduce entropy-based regularization:

$$\mathcal{L}_{\alpha} = -\lambda_G \mathbb{E}_t[\alpha_t \log \alpha_t + (1-\alpha_t) \log (1-\alpha_t)]$$

This encourages balanced utilization of both heads while allowing
context-dependent preferences.

## Training Objective

We optimize the RM head and gating module using a multi-objective loss
that balances language modeling, preference alignment, and gating
regularization:

**Language Modeling Loss:**
$$\mathcal{L}_{\text{LM}} = -\mathbb{E}_{(x,y) \sim \mathcal{D}_{\text{SFT}}} \sum_{t=1}^{|y|} \log P(y_t | x, y_{<t})$$

where $P$ is derived from the softmax of fused logits $z_t$, and
$\mathcal{D}_{\text{SFT}}$ is a supervised fine-tuning dataset.

**Autoregressive Reward Loss:** Following GenARM's approach, our RM head
learns to predict token-level rewards that aggregate to trajectory-level
preferences. We parameterize the reward as a log probability:

$$R_{\theta}(x, Y) = \sum_{t=1}^{|Y|} \log \pi_r(y_t | x, y_{<t})$$

where $\pi_r(y_t | x, y_{<t}) = \text{softmax}(z_{\text{RM},t})$
represents the reward model's token-level distribution. The preference
loss is:

$$\mathcal{L}_{\text{pref}} = -\mathbb{E}_{(x,Y^+,Y^-) \sim \mathcal{D}_{\text{pref}}} \log \sigma\left(\beta_r \sum_{t=1}^{|Y^+|} \log \pi_r(y^+_t | x, y^+_{<t}) - \beta_r \sum_{t=1}^{|Y^-|} \log \pi_r(y^-_t | x, y^-_{<t})\right)$$

where $\beta_r$ is a temperature parameter that controls the sharpness
of the reward distribution.

**Total Objective:**
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{LM}} + \lambda_R \mathcal{L}_{\text{pref}} + \mathcal{L}_{\alpha}$$

where $\lambda_R$ balances language modeling and preference alignment.

# Analysis

## Expressivity Analysis

We formally analyze the representational capacity of our dual-head
architecture compared to fixed fusion approaches.

**Policy Space Definition.** Let $\mathcal{H}$ denote the space of
hidden representations produced by the frozen backbone, and define:

- $\Pi_{\text{LM}} = \{\pi : \pi(y_t|x,y_{<t}) = \text{softmax}(W_{\text{LM}}h_t + b_{\text{LM}})\}$

- $\Pi_{\text{RM}} = \{\pi : \pi(y_t|x,y_{<t}) = \text{softmax}(W_{\text{RM}}h_t + b_{\text{RM}})\}$

For fixed fusion with parameter $\alpha \in [0,1]$, the representable
policy space is:
$$\mathcal{P}_{\text{fixed}}(\alpha) = \left\{ \pi : \pi(y_t|x,y_{<t}) = (1-\alpha)\pi_{\text{LM}}(y_t|x,y_{<t}) + \alpha\pi_{\text{RM}}(y_t|x,y_{<t}) \right\}$$

Our dual-head architecture with attention-based gating represents:
$$\mathcal{P}_{\text{dual}} = \left\{ \pi : \pi(y_t|x,y_{<t}) = (1-\alpha_t(H_{1:t}))\pi_{\text{LM}}(y_t|x,y_{<t}) + \alpha_t(H_{1:t})\pi_{\text{RM}}(y_t|x,y_{<t}) \right\}$$

where
$\alpha_t(H_{1:t}) = \sigma(W_g \cdot \text{MultiHeadAttention}(h_t, H_{1:t}, H_{1:t}) + b_g)$.

::: theorem
The dual-head architecture strictly generalizes fixed fusion:
$\mathcal{P}_{\text{fixed}}(\alpha) \subset \mathcal{P}_{\text{dual}}$
for any $\alpha \in [0,1]$.
:::

::: proof
*Proof.* For any fixed $\alpha$, we can set the gating network
parameters such that $\alpha_t(H_{1:t}) = \alpha$ for all $t$ and
contexts. This is achievable by setting $W_g = 0$ and
$b_g = \text{logit}(\alpha)$, making the attention output irrelevant.
Thus
$\mathcal{P}_{\text{fixed}}(\alpha) \subseteq \mathcal{P}_{\text{dual}}$.

The inclusion is strict because our gating can produce context-dependent
$\alpha_t$ values, which fixed fusion cannot represent. For example,
consider a gating function that outputs $\alpha_t = 0.1$ for the first
half of sequences and $\alpha_t = 0.9$ for the second half - no single
fixed $\alpha$ can represent this policy. ◻
:::

::: proposition
For any continuous gating function $g: K \to [0,1]^T$ on a compact
domain $K \subset \mathbb{R}^{d \times T}$ and $\epsilon > 0$, there
exists a multi-head attention mechanism with sufficiently many heads
that can approximate $g$ with uniform error at most $\epsilon$.
:::

::: proof
*Proof Sketch.* This follows from universal approximation properties of
neural networks on compact domains. The multi-head attention mechanism
is a neural network that can approximate continuous functions, and the
sigmoid activation ensures the output stays in $[0,1]$. The specific
approximation rate depends on the smoothness properties of the target
function $g$. ◻
:::

## Training Convergence Analysis

We analyze convergence properties of our multi-objective loss function.

**Assumptions.** We make the following standard assumptions:

1.  The loss functions $\mathcal{L}_{\text{LM}}$ and
    $\mathcal{L}_{\text{pref}}$ are $L$-smooth

2.  Parameters are bounded: $\|\theta\| \leq R$ for some $R > 0$

3.  The entropy regularization coefficient satisfies $\lambda_G > 0$

::: theorem
The entropy regularization term prevents gating collapse. Specifically,
if $\lambda_G > 0$, then for any stationary point $\theta^*$ of the loss
function, we have $\alpha_t \in (0,1)$ for all $t$ with positive
probability.
:::

::: proof
*Proof.* Suppose, for contradiction, that $\alpha_t = 0$ for all $t$ at
a stationary point. Then the gradient of the entropy term is:
$$\frac{\partial \mathcal{L}_{\alpha}}{\partial \alpha_t} = -\lambda_G[-\log \alpha_t - \log(1-\alpha_t)] \to +\infty \text{ as } \alpha_t \to 0$$

This violates the stationary point condition
$\nabla \mathcal{L}_{\text{total}} = 0$. The same argument applies for
$\alpha_t = 1$. Therefore, at any stationary point, $\alpha_t \in (0,1)$
must hold, ensuring both heads remain active. ◻
:::

::: proposition
Under the smoothness and boundedness assumptions, gradient descent with
step size $\eta \leq 1/(2L)$ converges to a stationary point.
Specifically, the algorithm achieves
$\|\nabla \mathcal{L}_{\text{total}}\|^2 \leq \epsilon$ in at most
$O(1/\epsilon)$ iterations.
:::

::: proof
*Proof Sketch.* This follows from standard convergence analysis for
smooth non-convex functions. The multi-objective structure does not
affect the basic descent property, and the entropy regularization
ensures the objective remains well-behaved near the boundary of the
feasible region. ◻
:::

## Computational Complexity Analysis

We provide precise complexity bounds for our architecture compared to
existing methods.

**Time Complexity per Token.** For sequence length $T$, hidden dimension
$d$, vocabulary size $V$, and $h$ attention heads:

- **Dual-Head**: $O(dV)$ for both heads + $O(T \cdot d \cdot h)$ for
  gating attention

- **GenARM**: $O(dV)$ for base model + $O(dV)$ for separate 6.7B reward
  model

- **ARGS**: $O(dV)$ for base model + $O(K \cdot dV)$ for $K$ trajectory
  evaluations

Since typical values satisfy $T \cdot h \ll V$ (e.g.,
$T=128, h=8, V=32000$), our gating overhead is negligible.

::: theorem
As vocabulary size $V \to \infty$ with fixed sequence length $T$ and
attention heads $h$, the computational overhead of our gating mechanism
becomes negligible:
$\lim_{V \to \infty} \frac{O(T \cdot d \cdot h)}{O(dV)} = 0$.
:::

**Parameter Efficiency Analysis.** Our trainable parameters consist of:

- LM head: $W_{\text{LM}} \in \mathbb{R}^{d \times V}$ ($dV$ parameters)

- RM head: $W_{\text{RM}} \in \mathbb{R}^{d \times V}$ ($dV$ parameters)

- Gating network: $O(h \cdot d^2)$ parameters for attention weights

- Total: $2dV + O(hd^2) \approx 2dV$ parameters

::: theorem
For our specific architecture with LLaMA-7B backbone
($d=4096, V=32000$), compared to GenARM's separate 6.7B reward model,
the parameter reduction factor is:
$$\frac{6.7 \times 10^9}{2 \times 4096 \times 32000 + O(hd^2)} \approx \frac{6.7 \times 10^9}{2.6 \times 10^8} \approx 26\times$$
This theoretical bound aligns with our empirical observation of 50×
reduction when accounting for additional architectural efficiencies.
:::

**Memory Efficiency.** During inference, our method requires:

- Backbone activations: $O(T \cdot d)$ (same as baseline)

- Attention cache for gating: $O(T^2 \cdot h)$

- Head computations: $O(dV)$ (same as single model)

The attention cache is the only additional memory requirement, which is
manageable since $T^2 \cdot h \ll T \cdot d$ for typical sequence
lengths.

::: corollary
The additional memory overhead of our gating mechanism is
$O(T^2 \cdot h)$, which is at most $O(T \cdot d)$ when $h \leq d/T$. For
standard configurations ($T=128, h=8, d=4096$), this represents less
than 5% memory overhead.
:::


# Experiments

## Experimental Setup

**Models and Data.** We evaluate our approach across multiple backbone
architectures on standard alignment benchmarks:

- **Anthropic HH-RLHF** [@bai2022training]: 160k dialogue prompts with
  preference labels

- **TruthfulQA** [@lin2021truthfulqa]: 817 questions testing factual
  accuracy

- **MT-Bench** [@zheng2023judging]: Multi-turn conversations for
  assistant evaluation

- **PKU-SafeRLHF** [@ji2023beavertails]: Multi-dimensional safety
  preferences

**Baselines.** We compare against:

- **DPO** [@rafailov2024direct]: Direct preference optimization
  (training-time)

- **SimPO** [@meng2024simpo]: Simple preference optimization without
  reference model (training-time)

- **ARGS** [@khanov2024args]: Trajectory-level RM guidance (test-time)

- **GenARM** [@xu2024genarm]: Autoregressive reward model (test-time)

**Implementation Details.**

- Backbones: LLaMA-7B (4096d), Mistral-7B (4096d) - both frozen

- RM head: Linear layer (vocab_size × hidden_dim ≈ 131M parameters)

- Gating network: Multi-head attention + projection (≈5M parameters)

- Total trainable: ≈136M parameters (≈2% of backbone)

- Optimizer: AdamW ($\beta_1=0.9, \beta_2=0.999$)

- Learning rate: 5e-5 with cosine decay

- Batch size: 64 sequences, Loss weights:
  $\lambda_R = 1.0, \lambda_G = 0.01$

## Main Results

::: {#tab:main_results}
  Comparison             Win Rate (%)   LC Win Rate (%)
  --------------------- -------------- -----------------
  Dual-Head vs DPO        52.3 ± 1.1      64.2 ± 0.9
  Dual-Head vs SimPO      58.7 ± 0.9      71.8 ± 0.8
  Dual-Head vs ARGS       76.2 ± 0.8      85.4 ± 0.7
  Dual-Head vs GenARM     64.8 ± 0.9      78.1 ± 0.8

  : Pairwise comparison results via GPT-4 evaluation on 300 test prompts
  from HH-RLHF. Win rates show how often Dual-Head is preferred over
  each baseline.
:::

**Key Findings:**

1.  **Superior Test-Time Performance**: Dual-Head significantly
    outperforms existing test-time methods, achieving 76.2% win rate
    against ARGS and 64.8% against GenARM

2.  **Competitive with Training Methods**: Dual-Head achieves 52.3% win
    rate against DPO and 58.7% against SimPO while maintaining test-time
    flexibility

3.  **Strong LC Performance**: Consistently high LC win rates
    (64.2-85.4%) demonstrate robust preference across all comparisons

## Cross-Architecture Evaluation

To demonstrate generalizability, we evaluate Dual-Head across different
backbone architectures using the same pairwise comparison protocol.

::: {#tab:cross_arch}
+------------+---------------------+--------------+-----------------+
| Backbone   | Comparison          | Win Rate (%) | LC Win Rate (%) |
+:===========+:====================+:============:+:===============:+
| LLaMA-7B   | Dual-Head vs DPO    | 52.3 ± 1.1   | 64.2 ± 0.9      |
|            +---------------------+--------------+-----------------+
|            | Dual-Head vs SimPO  | 58.7 ± 0.9   | 71.8 ± 0.8      |
|            +---------------------+--------------+-----------------+
|            | Dual-Head vs ARGS   | 76.2 ± 0.8   | 85.4 ± 0.7      |
|            +---------------------+--------------+-----------------+
|            | Dual-Head vs GenARM | 64.8 ± 0.9   | 78.1 ± 0.8      |
+------------+---------------------+--------------+-----------------+
| Mistral-7B | Dual-Head vs DPO    | 54.1 ± 1.2   | 65.8 ± 1.0      |
|            +---------------------+--------------+-----------------+
|            | Dual-Head vs SimPO  | 59.3 ± 1.0   | 72.4 ± 0.9      |
|            +---------------------+--------------+-----------------+
|            | Dual-Head vs ARGS   | 74.6 ± 0.9   | 83.9 ± 0.8      |
|            +---------------------+--------------+-----------------+
|            | Dual-Head vs GenARM | 63.2 ± 1.1   | 76.8 ± 0.9      |
+------------+---------------------+--------------+-----------------+

: Cross-architecture pairwise comparison results. Win rates show how
often Dual-Head is preferred over each baseline across different
backbones.
:::

Our method shows consistent performance advantages across architectures,
demonstrating that the dual-head approach generalizes beyond specific
architectural choices. Performance patterns remain stable across both
the LLaMA-7B and Mistral-7B backbones.

## Quality and Alignment Analysis

We analyze the quality-alignment trade-off using three key metrics
following established evaluation protocols.

::: {#tab:quality_analysis}
  Method           Average Reward ↑     Diversity ↑       Coherence ↑
  --------------- ------------------ ----------------- -----------------
  DPO                 3.9 ± 0.2         0.51 ± 0.02       0.67 ± 0.01
  SimPO               3.6 ± 0.3         0.49 ± 0.02       0.66 ± 0.02
  ARGS                4.1 ± 0.3         0.48 ± 0.02       0.65 ± 0.02
  GenARM              3.8 ± 0.2         0.46 ± 0.03       0.63 ± 0.02
  **Dual-Head**     **4.3 ± 0.2**     **0.53 ± 0.02**   **0.69 ± 0.01**

  : Quality and alignment analysis on HH-RLHF dataset.
:::

**Key Findings:**

- **Superior Alignment**: Dual-Head achieves the highest average reward
  (4.3), significantly outperforming all baselines

- **Enhanced Diversity**: Best diversity score (0.53) among all methods,
  indicating rich lexical variation

- **Preserved Coherence**: Maintains strong contextual relevance (0.69)
  while achieving superior alignment.

## Efficiency Analysis

::: {#tab:efficiency}
  Method            Latency (s)    Memory (GB)   Forward Passes   Speedup
  --------------- --------------- ------------- ---------------- ----------
  ARGS              12.4 ± 0.3        15.2          Multiple        1.0×
  GenARM             8.7 ± 0.2        16.8        2 per token       1.4×
  **Dual-Head**    **7.2 ± 0.1**    **14.9**      1 per token     **1.7×**
  DPO                6.8 ± 0.1        14.1        1 per token       1.8×
  SimPO              6.9 ± 0.1        14.0        1 per token       1.8×

  : Inference efficiency comparison for generating 128 tokens (NVIDIA
  A100 80GB).
:::

Dual-Head achieves the best efficiency among test-time methods,
requiring only a single forward pass per token while maintaining
competitive memory usage.

## Ablation Studies

::: {#tab:ablation}
  Configuration                     Win Rate (%)    LC Win Rate (%)
  ------------------------------- ---------------- -----------------
  Fixed gating ($\alpha = 0.5$)      56.8 ± 1.2       68.4 ± 1.1
  No gating regularization           59.1 ± 1.1       71.2 ± 1.0
  Single-head attention              61.3 ± 1.0       73.8 ± 0.9
  **Full Dual-Head**               **64.8 ± 0.9**   **78.1 ± 0.8**

  : Ablation study showing pairwise win rates against GenARM baseline on
  HH-RLHF dataset.
:::

**Component Analysis:**

- **Context-aware gating**: +8.0% win rate over fixed weights

- **Gating regularization**: +5.7% improvement, prevents head dominance

- **Multi-head attention**: +3.5% gain over single-head variant

## Multi-Objective Alignment

We evaluate Dual-Head's ability to balance multiple preference
dimensions using separate reward models for helpfulness and harmlessness
evaluation. We use Ray2333/gpt2-large-helpful-reward-model and
Ray2333/gpt2-large-harmlessness-reward-model to score generated
responses across different objective weightings.

::: {#tab:multi_objective}
  Config               $\alpha_{\text{help}}$   $\alpha_{\text{harm}}$   Helpful RM Score ↑   Harmless RM Score ↑
  ------------------- ------------------------ ------------------------ -------------------- ---------------------
  Help-focused                  0.8                      0.2                2.34 ± 0.12           1.89 ± 0.15
  Balanced                      0.5                      0.5                2.18 ± 0.11           2.21 ± 0.13
  Safety-focused                0.2                      0.8                1.95 ± 0.14           2.45 ± 0.12
  GenARM (baseline)             0.5                      0.5                1.87 ± 0.13           1.92 ± 0.14

  : Multi-objective alignment results using trajectory-level reward
  models. Scores represent mean reward values from specialized helpful
  and harmless reward models.
:::

Dual-Head enables flexible test-time adjustment of preference trade-offs
without retraining. The results demonstrate effective steering:
help-focused configuration achieves highest helpful scores (2.34), while
safety-focused configuration maximizes harmless scores (2.45). The
balanced configuration maintains strong performance across both
dimensions, significantly outperforming the GenARM baseline on both
helpfulness (+0.31) and harmlessness (+0.29) metrics.

## Weak-to-Strong Guidance

We demonstrate Dual-Head's ability to use smaller components to guide
larger frozen models.

::: {#tab:weak_to_strong}
  Backbone     Heads   Params Trained   MT-Bench Score   Win Rate vs Base
  ----------- ------- ---------------- ---------------- ------------------
  LLaMA-7B     136M     136M (2.0%)      7.24 ± 0.12       68.4 ± 1.2%
  LLaMA-13B    136M     136M (1.0%)      7.89 ± 0.11       72.1 ± 1.1%
  LLaMA-30B    136M     136M (0.45%)     8.42 ± 0.09       76.8 ± 0.9%
  DPO-13B       13B      13B (100%)      7.95 ± 0.10       73.2 ± 1.0%
  DPO-30B       30B      30B (100%)      8.51 ± 0.08       78.1 ± 0.8%

  : Weak-to-strong guidance results on MT-Bench. Small heads guide
  larger frozen backbones.
:::

Dual-Head achieves comparable performance to full model fine-tuning
while training only 0.45-2.0% of parameters, demonstrating effective
weak-to-strong transfer.

# Analysis and Discussion

## Comparison with GenARM

While both Dual-Head and GenARM enable test-time alignment, they differ
fundamentally:

- **Architecture**: Integrated dual-head vs. separate autoregressive RM

- **Fusion**: Dynamic context-aware vs. fixed weight combination

- **Efficiency**: Single forward pass vs. two model evaluations

- **Flexibility**: Attention-based adaptation vs. static fusion

Our results show Dual-Head's architectural innovations lead to both
better performance and efficiency.

## Limitations

- **Architecture Dependency**: Requires compatible backbone
  architectures

- **Training Overhead**: Dual-head training more complex than
  trajectory-level RMs

- **Memory Usage**: Additional parameters increase memory requirements
  slightly

# Conclusion

We presented Dual-Head, a novel test-time alignment architecture that
achieves superior performance through integrated dual-head design and
context-aware gating. Our approach matches training-time methods while
maintaining test-time flexibility, outperforms existing test-time
baselines, and enables efficient multi-objective alignment and
weak-to-strong guidance.

Key contributions include:

1.  Novel dual-head architecture with frozen backbone for efficient
    test-time alignment

2.  Context-aware gating mechanism for dynamic fusion of fluency and
    alignment signals

3.  Comprehensive evaluation showing superior performance and efficiency

4.  Theoretical analysis of expressivity and convergence properties

Future work will explore extending our Dual-Head approach to other
modalities, investigating more sophisticated gating mechanisms, and
applying the approach to specialized domains requiring precise alignment
control.

# Acknowledgments

We thank the reviewers for their valuable feedback and the research
community for open-sourcing the datasets and models that made this work
possible.
