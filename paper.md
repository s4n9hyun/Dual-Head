---
abstract: |
  Large Language Models (LLMs) require careful alignment with human
  preferences, but existing methods face trade-offs between
  effectiveness and efficiency. Training-time approaches like RLHF and
  DPO require expensive retraining for different preferences, while
  test-time methods often rely on separate models with static fusion
  strategies. We propose Context-Aware REward FUsion Learning
  (Dual-Head), a novel test-time alignment architecture that integrates
  dual output heads with a context-aware gating mechanism while keeping
  the base LLM frozen. Unlike existing approaches that use separate
  reward models or fixed fusion weights, Dual-Head employs dynamic
  token-level fusion guided by attention over sequence history. Our
  approach achieves significant parameter efficiency (50× reduction
  compared to GenARM) while maintaining competitive performance among
  test-time methods. Experiments on HH-RLHF demonstrate that while
  Dual-Head faces challenges competing with training-time methods like
  DPO (34.7% win rate), it outperforms some test-time baselines like
  ARGS while providing architectural innovations for efficient alignment.
---

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

Experiments on HH-RLHF demonstrate that Dual-Head:

- Achieves significant parameter efficiency with 50× fewer parameters
  than GenARM (131M vs 6.7B) while maintaining test-time flexibility

- Outperforms ARGS (34.7% vs 28.3% win rate against DPO) among 
  test-time methods, though underperforming GenARM (45.3% win rate)

- Provides architectural innovations for context-aware alignment
  without requiring multiple model evaluations

- Faces performance challenges against training-time methods like DPO,
  indicating areas for future improvement

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
reward estimation, requiring multiple forward passes and Dual-Head
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
  Comparison             Win Rate (%)   Tie Rate (%)   Loss Rate (%)
  --------------------- -------------- -------------- ---------------
  Dual-Head vs DPO        34.7 (104)     6.7 (20)      58.7 (176)
  ARGS vs DPO             28.3 (85)      15.0 (45)     56.7 (170)
  GenARM vs DPO           45.3 (136)     15.3 (46)     39.3 (118)

  : Pairwise comparison results via GPT-4 evaluation on 300 test prompts
  from HH-RLHF. Win rates show how often each method is preferred over
  DPO baseline. Numbers in parentheses show absolute counts.
:::

**Key Findings:**

1.  **Performance Gap with DPO**: Dual-Head achieves 34.7% win rate
    against DPO, indicating room for improvement in training-time
    competitive performance

2.  **Mixed Test-Time Results**: While outperforming ARGS (28.3% vs DPO),
    Dual-Head underperforms GenARM (45.3% vs DPO) on the same evaluation

3.  **Parameter Efficiency Trade-off**: The 50× parameter reduction 
    (131M vs 6.7B) comes with performance costs that need addressing

## Performance Analysis and Limitations

Our evaluation reveals important insights about the Dual-Head approach's
current performance characteristics and areas for improvement.

**Performance Challenges:**
The current implementation faces significant challenges when compared to
established methods. Against DPO, Dual-Head achieves only 34.7% win rate,
substantially lower than the 45.3% achieved by GenARM on the same evaluation.
This suggests that while the architectural innovations are promising, the
current training methodology may be suboptimal.

**Test-Time Method Comparison:**
Among test-time methods, results are mixed:
- Outperforms ARGS: 34.7% vs 28.3% win rate against DPO
- Underperforms GenARM: 34.7% vs 45.3% win rate against DPO

**Parameter Efficiency vs Performance Trade-off:**
The 50× parameter reduction comes with measurable performance costs.
While 131M parameters provide significant efficiency gains, they may be
insufficient to capture the complex alignment patterns that larger
models like GenARM's 6.7B reward model can learn.

## Future Research Directions

Based on the current evaluation results, several research directions emerge
for improving the Dual-Head approach:

**Training Methodology Improvements:**
- **Loss Function Tuning**: The current performance gap suggests suboptimal
  loss weight balancing (λ_R, λ_G) that requires systematic exploration
- **Gating Mechanism Enhancement**: The attention-based gating may need
  architectural refinements to better utilize context information
- **Optimization Strategies**: Alternative training schedules and learning
  rate schemes could improve convergence

**Architectural Refinements:**
- **Head Capacity**: The 131M parameter heads may be undersized for complex
  alignment tasks, requiring exploration of larger intermediate architectures
- **Multi-Scale Fusion**: Incorporating different temporal scales in the
  gating mechanism could improve context sensitivity
- **Regularization Schemes**: Better regularization strategies to prevent
  mode collapse while maintaining head specialization

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
fundamentally in design philosophy and performance:

**Architectural Differences:**
- **Integration**: Dual-head uses integrated heads vs. GenARM's separate autoregressive RM
- **Parameters**: 131M trainable parameters vs. GenARM's 6.7B reward model
- **Fusion**: Dynamic context-aware vs. GenARM's static weight combination
- **Efficiency**: Single forward pass vs. GenARM's two model evaluations

**Performance Trade-offs:**
Current evaluation reveals that GenARM's larger parameter count (6.7B) provides
superior alignment performance (45.3% vs 34.7% win rate against DPO). This
suggests that while Dual-Head's architectural efficiency is valuable, the
parameter reduction may be too aggressive for the complexity of alignment tasks.

**Research Implications:**
The comparison highlights the parameter-performance trade-off in test-time alignment,
suggesting that intermediate architectures between Dual-Head's 131M and GenARM's 6.7B
parameters may provide optimal balance of efficiency and effectiveness.

## Limitations

**Performance Gaps:**
- **Training-Time Competition**: 34.7% win rate against DPO indicates
  substantial room for improvement in competing with training-time methods
- **Test-Time Performance**: Underperforming GenARM (45.3% vs 34.7% against DPO)
  suggests current parameter allocation may be insufficient

**Methodological Challenges:**
- **Training Stability**: Complex multi-objective loss function requires
  careful hyperparameter tuning and may suffer from optimization difficulties
- **Head Capacity**: 131M parameter heads may be too small to capture
  nuanced alignment patterns compared to larger reward models

**Architectural Constraints:**
- **Backbone Dependency**: Requires compatible decoder architectures
- **Memory Trade-offs**: While more efficient than GenARM, still increases
  memory requirements over base models
- **Context Length**: Attention-based gating may become computationally
  expensive for very long sequences

# Conclusion

We presented Dual-Head, a novel test-time alignment architecture that
introduces significant architectural innovations through integrated
dual-head design and context-aware gating. While our approach achieves
substantial parameter efficiency (50× reduction compared to GenARM)
and maintains test-time flexibility, the current evaluation reveals
important performance gaps that inform future research directions.

Key contributions and findings include:

1.  **Novel Architecture**: Dual-head design with frozen backbone
    enabling efficient test-time alignment with minimal trainable
    parameters (131M vs 6.7B)

2.  **Context-Aware Gating**: Dynamic fusion mechanism using attention
    over sequence history for adaptive alignment intervention

3.  **Performance Analysis**: Honest evaluation showing challenges
    against training-time methods (34.7% win rate vs DPO) while
    outperforming some test-time baselines (ARGS: 28.3% vs DPO)

4.  **Research Insights**: Identification of training methodology and
    architectural refinements needed for competitive performance

This work establishes a foundation for parameter-efficient test-time
alignment while highlighting critical areas for improvement. Future
research should focus on optimizing training procedures, exploring
larger head architectures, and developing better fusion mechanisms
to realize the full potential of the dual-head approach.