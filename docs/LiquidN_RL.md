# Liquid-N Advantage Estimation

## Adaptive Temporal Credit Assignment via Leaky Horizon Dynamics

Nik Prince
2026

---

## Abstract

Temporal credit assignment remains one of the central challenges in reinforcement learning (RL). Existing advantage estimation techniques such as n-step returns and Generalized Advantage Estimation (GAE) approximate the true advantage function using fixed or geometrically weighted horizons. These estimators impose rigid bias–variance trade-offs governed by static hyperparameters.

This work proposes Liquid-N Advantage Estimation (LNAE), a dynamically adaptive advantage estimator where the temporal horizon evolves during learning. Instead of a fixed decay parameter, a time-varying horizon parameter β_t modulates the effective credit assignment length. The resulting estimator transforms the advantage computation into a trajectory-dependent temporal filter capable of adapting to reward dynamics, prediction error, and environmental volatility.

Liquid-N generalizes both n-step returns and GAE as special cases. We derive the mathematical formulation, analyze the bias–variance properties, and present an experimental protocol for empirical evaluation.

---

## 1. Introduction

Reinforcement learning algorithms rely on estimating the advantage function

A^π(s_t, a_t) = Q^π(s_t,a_t) − V^π(s_t)

in order to compute policy gradients.

Since the action-value function Q^π is generally unknown, advantage estimates are constructed from sampled trajectories. The quality of these estimators strongly influences training stability, variance of policy gradients, and convergence speed.

Two estimators dominate modern actor–critic algorithms:

• n-step returns
• Generalized Advantage Estimation (GAE)

Both define temporal weighting kernels that determine how future rewards influence gradient updates.

However, these kernels are fixed during training. The effective horizon length is determined by hyperparameters (n or λ) which remain constant regardless of environment dynamics.

In many environments this assumption is suboptimal because reward noise, prediction error, and state transitions vary across time. A static estimator cannot adapt its temporal horizon accordingly.

This paper introduces Liquid-N Advantage Estimation, where the horizon parameter evolves dynamically based on trajectory information.

---

## 2. Background

### 2.1 n-Step Returns

The n-step return estimator is defined as

R_t^(n) = Σ_{k=0}^{n−1} γ^k r_{t+k} + γ^n V(s_{t+n})

The advantage estimate becomes

A_t^(n) = R_t^(n) − V(s_t)

Small values of n introduce higher bias but lower variance. Large values of n reduce bias but increase variance.

### 2.2 Generalized Advantage Estimation

Define the temporal-difference residual

δ_t = r_t + γ V(s_{t+1}) − V(s_t)

GAE computes the advantage estimate

A_t^{GAE(λ)} = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}

GAE forms a geometric mixture of n-step estimators. The parameter λ controls the bias–variance trade-off.

However λ remains constant during training.

---

## 3. Problem Statement

Current advantage estimators assume stationary temporal weighting kernels.

Formally, the weight applied to a TD residual at lag l is

w(l) = (γλ)^l

for GAE.

This fixed structure prevents the estimator from adapting to changes in trajectory statistics such as reward volatility or prediction error magnitude.

An ideal estimator should instead use

w_t(l) = f(trajectory dynamics)

where the temporal weighting evolves during learning.

---

## 4. Liquid-N Advantage Estimation

We introduce a time-dependent horizon modulation parameter

β_t ∈ [0,1]

The Liquid-N advantage estimator is

A_t^Liquid = Σ_{l=0}^{∞} ( Π_{i=0}^{l−1} γ β_{t+i} ) δ_{t+l}

This replaces the constant λ used in GAE with a time-varying parameter.

### 4.1 Special Cases

If β_t = λ (constant)

A_t^Liquid = Σ (γλ)^l δ_{t+l}

which reduces exactly to GAE.

If β_t = 1 for l < n and 0 otherwise, the estimator becomes the n-step estimator.

Thus Liquid-N forms a strict generalization of both methods.

---

## 5. Dynamic Horizon Update

The horizon parameter evolves through a leaky update rule

β_{t+1} = (1 − α) β_t + α f(ξ_t)

where

ξ_t = (s_t, a_t, r_{t+1}, s_{t+1})

and α is the leak rate controlling adaptation speed.

---

## 6. Candidate Adaptation Functions

### TD-error adaptation

f = sigmoid(|δ_t|)

### Uncertainty-based adaptation

f = σ_V^2 / (σ_V^2 + δ_t^2)

### Reward volatility

f = |r_t − r_{t−1}| / (|r_t| + ε)

These functions allow the estimator to shorten or extend its horizon based on trajectory dynamics.

---

## 7. Algorithm Integration

Liquid-N can replace GAE in actor–critic algorithms such as PPO or A2C.

### Pseudocode

for each timestep t:

```
δ_t = r_t + γ V(s_{t+1}) − V(s_t)

β_t = (1 − α) β_{t−1} + α f(ξ_t)

running_weight = γ β_t

A_t = δ_t

for future step l:

    A_t += running_weight * δ_{t+l}

    running_weight *= γ β_{t+l}
```

---

## 8. Bias–Variance Interpretation

Define the expected effective horizon

H_t = Σ_{l=0}^{∞} Π_{i=0}^{l−1} β_{t+i}

High β values extend the horizon while low values shorten it.

The estimator therefore adapts its bias–variance trade-off dynamically.

---

## 9. Experimental Protocol

Baselines:

• n-step advantage estimation
• Generalized Advantage Estimation

Algorithms:

• A2C
• PPO

Environments:

• CartPole-v1
• LunarLander-v2
• HalfCheetah-v4

Metrics:

• policy loss
• value loss
• training stability
• sample efficiency

Primary comparison will analyze loss curves across training epochs.

---

## 10. Hypothesis

Liquid-N Advantage Estimation may:

1. Reduce gradient variance
2. Improve long-horizon credit assignment
3. Reduce hyperparameter sensitivity

---

## 11. Future Work

Possible extensions include

• meta-learning β dynamics
• Bayesian horizon estimation
• spectral reward analysis

---

## 12. Conclusion

Liquid-N Advantage Estimation introduces a dynamically adaptive horizon for advantage computation. By allowing temporal credit assignment to evolve with trajectory dynamics, the estimator provides a flexible mechanism for bias–variance control.

This framework opens new directions for adaptive reinforcement learning algorithms.
