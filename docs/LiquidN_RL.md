# **Liquid-N Advantage Estimation**

## **Adaptive Temporal Credit Assignment via Leaky Horizon Dynamics**

**DevaNik**
2026

---

# **Abstract**

Temporal credit assignment remains one of the central challenges in reinforcement learning (RL). Existing advantage estimation techniques such as **n-step returns** and **Generalized Advantage Estimation (GAE)** approximate the true advantage function using fixed or geometrically weighted horizons. These estimators impose rigid **bias–variance trade-offs** governed by static hyperparameters.

This work proposes **Liquid-N Advantage Estimation (LNAE)**, a dynamically adaptive advantage estimator where the temporal horizon evolves during learning. Instead of a fixed decay parameter, a **time-varying horizon parameter** $\beta_t$ modulates the effective credit assignment length.

The resulting estimator transforms advantage computation into a **trajectory-dependent temporal filter** capable of adapting to reward dynamics, prediction error, and environmental volatility.

Liquid-N generalizes both **n-step returns** and **GAE** as special cases. We derive the mathematical formulation, analyze bias–variance properties, and outline an empirical evaluation protocol.
<!-- TODO: Provide code examples or empirical benchmarks demonstrating Liquid-N's performance compared to GAE. -->

---

# **1. Introduction**

Reinforcement learning algorithms rely on estimating the **advantage function**.

$$
A^{\pi}(s_t,a_t) = Q^{\pi}(s_t,a_t) - V^{\pi}(s_t)
$$

This quantity determines the direction of **policy gradient updates**.

Because the true action-value function $Q^{\pi}$ is unknown, advantage estimates must be constructed from **sampled trajectories**.

The quality of this estimator directly affects:

* policy gradient variance
* training stability
* convergence speed

Modern actor–critic algorithms rely primarily on two estimators:

• **n-step returns**
• **Generalized Advantage Estimation (GAE)**

These estimators define **temporal weighting kernels** that determine how future rewards influence gradient updates.

However, their horizon structure is **static**. Hyperparameters such as $n$ or $\lambda$ remain fixed during training regardless of environmental dynamics.

In practice, reward noise, value prediction error, and state transitions vary significantly across time. A fixed estimator cannot adapt its horizon accordingly.

This motivates a **dynamic horizon estimator**.

---

# **2. Background**

## **2.1 n-Step Returns**

The n-step return estimator is defined as

$$
R_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n V(s_{t+n})
$$

The corresponding advantage estimate is

$$
A_t^{(n)} = R_t^{(n)} - V(s_t)
$$

Properties:

* small $n$ → higher bias, lower variance
* large $n$ → lower bias, higher variance

---

## **2.2 Generalized Advantage Estimation (GAE)**

Define the **temporal difference residual**:

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

GAE computes the advantage estimate:

$$
A_t^{GAE(\lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
$$

Interpretation:

GAE constructs a **geometric mixture of n-step estimators**.

However, the parameter $\lambda$ remains **constant throughout training**.

---

# **3. Problem Statement**

Existing estimators assume **stationary temporal weighting kernels**.

For GAE, the weight applied to a TD residual at lag $l$ is

$$
w(l) = (\gamma \lambda)^l
$$

This structure cannot adapt to changes in trajectory statistics such as:

* reward volatility
* prediction error magnitude
* environment stochasticity

An ideal estimator would instead allow

$$
w_t(l) = f(\text{trajectory dynamics})
$$

where the weighting kernel evolves during learning.

---

# **4. Liquid-N Advantage Estimation**

We introduce a **time-dependent horizon modulation parameter**:

$$
\beta_t \in [0,1]
$$

The **Liquid-N advantage estimator** is defined as

$$
A_t^{Liquid} = \sum_{l=0}^{\infty} \left( \prod_{i=0}^{l-1} \gamma \beta_{t+i} \right) \delta_{t+l}
$$

This replaces the constant $\lambda$ used in GAE with a **time-varying parameter**.

---

## **4.1 Special Cases**

### Constant Horizon (GAE)

If

$$
\beta_t = \lambda
$$

then

$$
A_t^{Liquid} = \sum_{l=0}^{\infty}(\gamma \lambda)^l \delta_{t+l}
$$

which reduces exactly to **GAE**.

---

### Deterministic Horizon (n-step)

If

$$
\beta_t =
\begin{cases}
1 & l < n \
0 & l \ge n
\end{cases}
$$

then the estimator becomes the **n-step estimator**.

Thus **Liquid-N forms a strict generalization** of both methods.

---

# **5. Dynamic Horizon Update**

The horizon parameter evolves through a **leaky integration rule**:

$$
\beta_{t+1} = (1-\alpha)\beta_t + \alpha f(\xi_t)
$$

where

$$
\xi_t = (s_t, a_t, r_{t+1}, s_{t+1})
$$

and

$$
0 < \alpha < 1
$$

controls the **adaptation speed**.

---

# **6. Candidate Adaptation Functions**

## TD-error based adaptation

$$
f = \sigma(|\delta_t|)
$$

Interpretation:

Large prediction errors **shorten the effective horizon**.

---

## Uncertainty-based adaptation

$$
f = \frac{\sigma_V^2}{\sigma_V^2 + \delta_t^2}
$$

where $\sigma_V^2$ represents **value uncertainty**.

---

## Reward volatility

$$
f = \frac{|r_t - r_{t-1}|}{|r_t| + \epsilon}
$$

Interpretation:

High reward volatility leads to **shorter temporal credit assignment**.

---

# **7. Algorithm Integration**

Liquid-N can replace **GAE** in any actor–critic algorithm such as **PPO** or **A2C**.

### Pseudocode

```
for timestep t:

    δ_t = r_t + γ * V(s_{t+1}) - V(s_t)

    β_t = (1 - α) * β_{t-1} + α * f(ξ_t)

    running_weight = γ * β_t

    A_t = δ_t

    for future step l:

        A_t += running_weight * δ_{t+l}

        running_weight *= γ * β_{t+l}
```

---

# **8. Bias–Variance Interpretation**

Define the **expected effective horizon**:

$$
H_t = \sum_{l=0}^{\infty} \prod_{i=0}^{l-1} \beta_{t+i}
$$

High $\beta$ values extend the horizon while low values shorten it.

Thus the estimator **dynamically adapts the bias–variance trade-off**.

---

# **9. Experimental Protocol**

### Baselines

* n-step advantage estimation
* Generalized Advantage Estimation

### Algorithms

* A2C
* PPO

### Environments

* CartPole-v1
* LunarLander-v2
* HalfCheetah-v4

### Metrics

* policy loss
* value loss
* training stability
* sample efficiency

Primary comparison:

$$
\text{Policy Loss vs Training Epochs}
$$

across estimators.

---

# **10. Hypothesis**

Liquid-N Advantage Estimation may:

1. Reduce gradient variance
2. Improve long-horizon credit assignment
3. Reduce hyperparameter sensitivity

---

# **11. Future Work**

Potential extensions include:

* meta-learning horizon dynamics
* Bayesian horizon estimation
* spectral analysis of reward signals

---

# **12. Conclusion**

**Liquid-N Advantage Estimation** introduces a dynamically adaptive horizon for advantage computation.

By allowing temporal credit assignment to evolve with trajectory dynamics, the estimator provides a flexible mechanism for **adaptive bias–variance control**.

This framework opens new directions for **adaptive reinforcement learning algorithms and estimator design**.
<!-- Jules-Patrol: A very thorough mathematical derivation. Future iterations could add empirical benchmark graphs against standard GAE to visually support the claims of reduced gradient variance. -->
