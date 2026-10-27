# ProjectTh: Provably Safe Shielded RL-IPID-MPC for Automatic Train Operation (ATO/ATP)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Real-Time Pass](https://img.shields.io/badge/Inference-%3C55%20%CE%BCs%2Fstep-brightgreen.svg)]()

**ProjectTh** bridges formal railway safety supervision (**Automatic Train Protection / ATP**, such as ETCS and KVB) and multi-objective **Automatic Train Operation (ATO)** speed tracking. Instead of relying on crude kinetic-energy emergency overrides that degrade passenger comfort and energy efficiency, this framework unifies **Heuristic Intelligent-PID (IPID)**, **Shift-Adaptive Model Predictive Control (MPC)**, **Causal Neural Physics Emulation**, and **Dynamic Model Predictive Shielding (DMPS)** into a real-time ($\sim 48\,\mu\text{s}/\text{step}$) control architecture.

---

## Key Contributions & Architecture

1. **Idea A — Horizon-Aware IPID Switching:**
   Upgrades the 4-algorithm heuristic PID bank (**GA, PSO, DE, ICA** tuned via Cardano's cubic root formula) of Havaei & Sandidzadeh (2022) with a 20-dimensional path-agnostic observation space encoding a 60-step spatial kinematic acceleration lookahead horizon $\mathbf{A}_{\text{ref}}$ and past history $\mathbf{A}_{\text{past}}$.
2. **Idea B — Shift-Aware Adaptive MPC:**
   Upgrades constrained linear-quadratic Model Predictive Control (Liu et al., 2025) with dynamic shift-onset weight relaxation ($Q_2, Q_3 \downarrow, Q_1 \uparrow$), 1-step actuator-lag phase-lead compensation, counter-braking anti-overshoot damping, and a terminal station docking precision boost.
3. **Causal Neural Physics Surrogate Emulator (150-Step):**
   Replaces static 3rd-order transfer function approximations with a learned residual neural network emulator trained on true nonlinear point-mass train dynamics (Davis aerodynamic drag $F_B = Av^2 + Bv + C$, sinusoidal track grades, curve resistance, motor field-weakening, and actuator lag $\tau = 0.28$).
4. **Unified 5-Action Hybrid RL-IPID-MPC Policy:**
   Combines `[GA, PSO, DE, ICA, MPC]` over the probability simplex with bumpless internal state synchronization between the PID bank and the MPC solver.
5. **Atomic Runtime Safety Shield (`safety_shield.py`):**
   Propagates analytic $z\text{-}\sigma$ ($99\%$ confidence) uncertainty tubes over the neural emulator and supports three non-hardcoded runtime intervention schemes:
   - `mode="switch"`: Classic binary Model Predictive Shielding (MPS).
   - `mode="cbf_opt"`: 1-Step Discrete-Time Control Barrier Function (DCBF) minimal-intervention bisection.
   - `mode="mps_rollout"`: $N=4$ step Dynamic Model Predictive Shielding (`planRec`) optimizing discounted recovery return and terminal Q-value synergy.

---

## Repository Structure

```text
ProjectTh/
├── benchmark_central.py        # Main training & 30-track zero-shot evaluation suite + 6-panel plotting
├── havaei_generalizing_rl.py   # Cardano IPID bank, nonlinear train physics, neural emulator & RL trainers
├── pure_mpc_controller.py      # Standalone & Idea C Shift-Adaptive Model Predictive Controller (PureLiuMPC)
├── safety_shield.py            # Atomic Safety Filter (Switch MPS, 1-Step DCBF, Multi-Step DMPS Rollout)
├── fuzzy_pid_controller.py     # Mamdani 49-rule Fuzzy-PID & ILR-PSO-Fuzzy-PID baseline (He et al., 2024)
└── final_paper_benchmark.png   # Generated 6-panel benchmark figure across unseen trajectories
