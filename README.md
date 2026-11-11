# Safe and Smooth Automatic Train Speed Control (ATP + ATO)
### From State-Augmented LTV-MPC to Multi-Station Hierarchical Differential and Assume-Guarantee Contract Shielding

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](#)
[![Real-Time Latency](https://img.shields.io/badge/Step%20Latency-~23%20%C2%B5s%20(%3C500%20%C2%B5s)-brightgreen.svg)](#)
[![Tube Containment](https://img.shields.io/badge/99%25%20Scissor%20Tube-99.96%25%20Contained-success.svg)](#)

## Overview

Modern railway control systems are traditionally split between **Automatic Train Protection (ATP)**, which enforces hard speed ceilings via conservative worst-case braking envelopes, and **Automatic Train Operation (ATO)**, which optimizes passenger ride comfort, traction energy efficiency, and station docking accuracy without formal multi-property safety guarantees.

This repository implements a **three-stage real-time control and shielding architecture** that unifies ATP safety certification and ATO multi-objective optimization within a single **~23 $\mu\text{s}$** control loop:

1. **Stage 1 — State-Augmented LTV-MPC & Heteroscedastic Scissor Shielding:** Lifts the physical 1st-order drivetrain lag ($u_{\text{act}}(k) = 0.72\,u_{\text{cmd}}(k) + 0.28\,u_{\text{act}}(k-1)$) directly into a Linear Time-Varying MPC state vector $z_k = [v_k, u_{\text{act}}(k-1)]^\top$, eliminating speed-shift overshoot ($0.748\text{ m/s} \to 0.035\text{ m/s}$). Replaces offline polyhedral tube reachability sets with a two-head **Heteroscedastic Neural Physics Emulator** ($\hat{f}_\theta$) trained under Gaussian Negative Log-Likelihood (NLL) to propagate a contractive **99% Gaussian Scissor Tube** ($\hat{v}_{\text{emu}}(k) \pm 2.576\,\sigma_{\text{tube}}(k)$).
2. **Stage 2 — Short-Horizon Compression, Disturbance Observer (DOB) & Ultra-Fast Solver:** Exploits actuator lag decay ($0.28^4 \approx 0.6\%$) to compress the MPC horizon from $(N_p=10, N_c=5)$ to $(N_p=5, N_c=3)$, removes redundant PID switching, and pairs an online Disturbance Observer (DOB) with a gust-gated scissor shield to reject $1.03\text{ m/s}^2$ storm gusts ($0.0169 \to 0.0069\text{ m/s}$ tracking error).
3. **Stage 3 — Multi-Station PPO Trajectory Pilot, $C^2$ Cosine Shaper & Hierarchical Assume-Guarantee Contract Shield:** Generalizes online trajectory generation across **1-to-5 station corridors ($4\text{--}31\text{ km}$)** under unknown passenger boarding/alighting mass shifts ($75.6\text{--}94.2\text{ t}$), wet/snowy rail adhesion drops ($\mu \in [0.68, 1.0]$), ambient temperature air-density shifts, and sporadic wind fronts.

---

## System Architecture (Stage 3 Closed-Loop Hierarchy)

```text
+---------------------------------------------------------------------------------------------------+
| 1. MACRO-STEP PPO PILOT (Every 4.0s in Cruise, 0.5s in Landing | 15-D Observation Vector)         |
|    Outputs p_task = (v_cmd, a_start, b_land) -> Target Cruise Speed + Overshoot & Ramp Rates      |
+-------------------------------------------------+-------------------------------------------------+
                                                  | p_task
                                                  v
+---------------------------------------------------------------------------------------------------+
| 2. HIERARCHICAL DIFFERENTIAL & ASSUME-GUARANTEE CONTRACT SHIELD (Goodall + Cano + Adalat)         |
|    * Evaluates Derivative Pressures rho_d(t) <= 1.0 across nested domains W_3 subset W_2 subset W_1|
|    * Certified Contract Library L_cert = {C0, C1_Speed, C2_JerkDamp, C3_Deadline, C4_EcoRegen}    |
|    * Discounted-UCB Multi-Contract Selector + Lazy 2-Step Neural Rollout + Ramp Latching          |
+-------------------------------------------------+-------------------------------------------------+
                                                  | p_safe = (v_safe, a_safe, b_safe)
                                                  v
+---------------------------------------------------------------------------------------------------+
| 3. PHASE-TRACKED C^2 HALF-COSINE S-CURVE SHAPER (cosine_trajectory_shaper.py)                     |
|    * Generates jerk-bounded S-curves for all speed shifts and dynamic station landings            |
|    * Computes dynamic braking initiation x_brake(v, b_eff, d_hat) & 5-step horizon V_ref(k)       |
+-------------------------------------------------+-------------------------------------------------+
                                                  | V_ref(k) in R^5
                                                  v
+---------------------------------------------------------------------------------------------------+
| 4. ULTRA-FAST 5x3 STATE-AUGMENTED LTV-MPC (pure_mpc_controller.py | ~11 us Core Execution)        |
|    * Upper-triangular scalar recurrence + closed-form 3x3 cofactor/Cramer QP + Gershgorin bound   |
+-------------------------------------------------+-------------------------------------------------+
                                                  | u_cmd(k) in [-1, 1]
                                                  v
+---------------------------------------------------------------------------------------------------+
| 5. NONLINEAR TRAIN PLANT & ONLINE DECOUPLED ESTIMATORS (dynamic_railroad_mpc.py)                  |
|    * Steps k < 8 (Platform Zone): Uses station telemetry to lock Passenger Mass M_hat (0.0t err)  |
|    * Steps k >= 8 (Open Corridor): Locks M_hat and tracks unseen Weather/Wind Disturbances d_hat  |
+---------------------------------------------------------------------------------------------------+
```

---

## Key Technical Contributions

### 1. Decoupled 8-Step Station Mass Identification & Open-Line Weather Observer
Simultaneously estimating vehicle mass $M$ and aerodynamic/weather forces $d_{\text{wind}}$ on open track is algebraically ill-conditioned. We decouple them causally:
* **Station Departure Window ($k = 0 \dots 7$, $t \le 4.0\text{ s}$):** While departing the station platform, the train reads wayside station telemetry (anemometer $d_{\text{station}}$, ambient temperature $T_{\text{station}}$, and rail adhesion $\mu_{\text{station}}$) and solves for the true passenger mass from wheel-rim traction response:
  $$\hat{M}_{\text{sample}}(k) = \frac{F_{\text{motor}}(k) - F_d(v_{k-1})}{(1 + \gamma)\big(a_{\text{kin}}(k) - d_{\text{station}}(k)\big) + g\,\iota(x_k)}, \qquad \hat{M}_{\text{locked}} = \text{median}_{k=1\dots 8}\big\{\hat{M}_{\text{sample}}(k)\big\}$$
* **Open Corridor ($k \ge 8$):** The wayside link terminates, $\hat{M}_{\text{locked}}$ is injected into the LTV-MPC without resetting the active departure S-curve phase, and all subsequent acceleration residuals are attributed to unseen open-line weather and wind disturbances $\hat{d}_{\text{wind}}(k)$.

### 2. Phase-Tracked $C^2$ Half-Cosine S-Curves & Cruise Velocity Overshoot
All upward/downward speed shifts and station stops follow analytical half-cosine profiles:
$$v(\tau) = \frac{v_{\text{eff}}}{2}\left(1 + \cos\left(\frac{\pi \tau}{T_{\text{land}}}\right)\right), \qquad T_{\text{land}} = \frac{\pi v_{\text{eff}}}{2 b_{\text{eff}}}, \qquad \Delta x_{\text{brake}} = \frac{\pi v_{\text{eff}}^2}{4 b_{\text{eff}}}$$
Because the train averages $\frac{1}{2}v_{\text{eff}}$ along the braking flare, tracking a naive pace $v_{\text{naive}} = x_{\text{rem}}/t_{\text{rem}}$ arrives $\sim 19.8\text{ s}$ late per leg. The 15-feature PPO pilot learns the exact open-track cruise velocity overshoot $\Delta v_{\text{overshoot}}$ required to enter the braking flare on schedule.

### 3. Hierarchical Differential Shielding & Assume-Guarantee Contract Portfolio ($\mathcal{L}_{\text{cert}}$)
Unifying higher-order differential safety (Cano et al., 2026) with Assume-Guarantee contracts (Adalat et al., 2026), the shield monitors normalized derivative pressures $\rho_d(t) = \|\partial_t^d x\| / r_d \le 1.0$ across nested viability domains $\mathcal{W}_3 \subseteq \mathcal{W}_2 \subseteq \mathcal{W}_1$:
* **$\mathcal{C}_0$ (`C0_NOMINAL_BALANCED`):** Pass-through contract when all 6 operational margins $h_i \ge 0$.
* **$\mathcal{C}_1$ (`C1_SPEED_SCISSOR_CLAMP` — Order $d=1$):** Clamps target velocity to $v_{\text{ceil}}(x) - 2.576\,\sigma_{\text{tube}} - 0.02\text{ m/s}$.
* **$\mathcal{C}_2$ (`C2_COMFORT_JERK_DAMP` — Orders $d=2, 3$):** Evaluates candidate ramps at initiation ($k \in \{0, 8\}$ or $|\Delta v| > 1.65\text{ m/s}$) and latches gentle ramp caps ($a_{\text{cap}} = 0.74\text{ m/s}^2, b_{\text{cap}} = 0.66\text{ m/s}^2$) across the active S-curve with $+0.06\text{--}0.12\text{ m/s}$ cruise velocity compensation.
* **$\mathcal{C}_3$ (`C3_DEADLINE_RECOVERY` — Temporal Liveness):** Compensates cruise speed against strong wind gusts ($|\hat{d}_{\text{wind}}| > 0.30\text{ m/s}^2$) and compresses station dwell times ($25\text{ s} \to 19\text{ s}$) after stress-test delays.
* **$\mathcal{C}_4$ (`C4_ECO_REGEN_COAST` — Traction-Chain Eco-Driving):** Initiates predictive coasting ahead of slower speed zones ($280\text{ m}$ lookahead) and pre-landing windows ($x_{\text{rem}} \in [220, 285]\text{ m}$) to keep braking inside the $100\%$ electrodynamic regenerative window ($b \le 0.70\text{ m/s}^2$, $46\%$ grid recovery).

 candidate contracts are selected online via a **Discounted-UCB** multi-armed bandit ($\eta = 0.95, \beta_{\text{ucb}} = 0.35$) and verified with a lazy 2-step neural emulator rollout.

### 4. Sub-25 $\mu\text{s}$ Real-Time Numerical Engineering
To meet strict onboard hardware ceilings ($< 500\text{ }\mu\text{s}$ per step), the codebase eliminates dynamic memory and C++ wrapper bottlenecks:
* **Upper-Triangular Scalar Recurrence:** Builds $F, \Phi, \Psi$ without matrix multiplications.
* **Closed-Form $3 \times 3$ Cofactor QP + $O(1)$ Gershgorin Bound:** Replaces LAPACK `np.linalg.solve` and `eigvalsh` calls.
* **Zero-Copy In-Place NumPy Parameter Views:** Evaluates the PyTorch Actor MLP and SiLU Physics Emulator directly in pre-allocated NumPy buffers (`~11.3 us` core step, `~23.4 us` total shielded step).

---

## Repository Structure

| File | Description |
| :--- | :--- |
| `railroad_corridor.py` | **Module 1:** Generalized 1-to-5 station railway generator ($3.8\text{--}31.0\text{ km}$), 5-zone spatial speed limits, anticipatory $C^1$ cosine spatial ceiling, stochastic passenger boarding/alighting mass model, and localized weather/wind generator (`Dry`, `Rain-WetRail`, `Cold-Humid`, `Snow-Slush`). |
| `cosine_trajectory_shaper.py` | **Module 2:** Phase-tracked $C^2$ half-cosine S-curve generator for upward/downward speed shifts and dynamic station landing (`dynamic_cosine_braking_spot` & `invert_cosine_distance_phase`). |
| `goodall_trajectory_shield.py` | **Module 3:** Heteroscedastic GP/NLL 99% Scissor Tube propagation, hierarchical derivative-pressure evaluator ($\rho_1, \rho_2, \rho_3, \rho_E, \rho_T$), Assume-Guarantee contract library ($\mathcal{C}_0\text{--}\mathcal{C}_4$), and Switch / 1-Step DCBF / UCB-DMPS Rollout shielding modes. |
| `dynamic_railroad_mpc.py` | **Module 4 (Main Runner):** 15-feature Multi-Station RL environment (`MPCTransitionRailroadEnv`), 8-step station mass identifier, traction-chain copper/slew loss & regenerative braking model, PPO+GAE trainer, latency spike profiler, and 30-scenario benchmark suite. |
| `pure_mpc_controller.py` | State-augmented LTV-MPC ($z_k = [v_k, u_{\text{act}}(k-1)]^\top$) with horizon $N_p=5, N_c=3$, causal plateau lookahead guard, and fast box-constrained QP solver. |
| `bilbao_cbtc_ato.py` | Standalone implementation of the Bilbao-Moreno et al. (2025) two-layer CBTC-ATO controller (RMBTP Regula Falsi Mesoscopic Planner + Microscopic Cascade PI-Position/PI-Speed with braking switcher). |
| `havaei_generalizing_rl.py` | Nonlinear subway train plant (`NonlinearTrainPhysics`), Davis drag, grade/curve resistance, and `LearnedLinearizationModel` heteroscedastic neural network. |
| `safety_shield.py` | Standalone Stage 1 & Stage 2 atomic safety filter and analytical scissor band diagnostics. |

---

## Installation & Quick Start

### 1. Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy torch matplotlib
```

### 2. Instant Evaluation Using Pre-Trained Checkpoints (~3 Seconds)
To skip the 220-epoch PPO training loop and immediately run the 30-scenario generalized benchmark (Clean, Weather-Disturbed, and Combined tables + diagnostic plots) using the saved `.pt` weights (`best_ppo_cosine_pilot.pt` and `best_emulator.pt`):
```bash
python dynamic_railroad_mpc.py --eval-runs 30 --n-legs 0 --wind-mode mixed --load-checkpoint best_ppo_cosine_pilot.pt
```

### 3. Full Training & Evaluation from Scratch (~1.8 Minutes)
To train the 150-step Heteroscedastic Physics Emulator and the 15-feature Multi-Property PPO Pilot from scratch across 220 epochs:
```bash
python dynamic_railroad_mpc.py --epochs 220 --eval-runs 30 --n-legs 0 --wind-mode mixed
```

### CLI Options
* `--epochs`: Number of PPO training epochs (default: `220`).
* `--eval-runs`: Number of randomized multi-station test scenarios (default: `30`).
* `--n-legs`: Number of station-to-station legs per journey (`0` = randomly sample `1..5` stations per scenario; or set fixed `1..5`).
* `--wind-mode`: Weather disturbance mode (`mixed` = 15 Clean + 15 Weather-Disturbed; `disjoint` = sporadic wind/weather on all runs; `continuous` = evolving wind; `none` = clean only).
* `--load-checkpoint`: Path to pre-trained policy `.pt` file to bypass training.

---

## Experimental Results

### Stage 1 & Stage 2 Summary (Single-Corridor Speed Tracking)
* **Stage 1 (30 Unseen Trajectories, No Wind):** Lifting actuator delay into LTV-MPC reduces shift overshoot by **95.3%** ($0.7480\text{ m/s} \to 0.0350\text{ m/s}$) and stopping error from $57.1\text{ cm}$ to **$8.2\text{ cm}$** ($0.0031\text{ m/s}$ tracking RMSE).
* **Stage 2 (30 Unseen Trajectories, $1.03\text{ m/s}^2$ Storm Gusts):** Compressing to $N_p=5, N_c=3$ with DOB and gust-gated Scissor Shielding halves wind-induced tracking error ($0.0169\text{ m/s} \to \mathbf{0.0069\text{ m/s}}$) and improves stopping accuracy to **$10.6\text{ cm}$**.

### Stage 3 Generalized Multi-Station Benchmark ($N = 30$ Scenarios, $1\text{--}5$ Stations, $4\text{--}31\text{ km}$)
Evaluated under the loss-consistent multi-objective metric:
$$\text{LossScore} = 1.0\,|\text{TimeErr}| + 15.0\,|\text{StopErr}|_{\text{m}} + 2.0\,\text{SmoothLoss} + 0.025\,\text{SpecEnergy}$$

#### Part A: Clean / No-Weather Scenarios Only ($N = 15$)
| Trajectory Pilot + Contract Shield Method | $\|\text{TimeErr}\|$ (s) | $\|\text{StopErr}\|$ (m) | SmoothLoss | SpecEnergy ($\text{Wh/t}\cdot\text{km}$) | LossScore | Shield% | Core Step | Total (Mean / P99 / Max) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1. Max-Speed Greedy Cosine + MPC | 3.73 s | 0.004 m | 0.0729 | 39.35 | 4.9143 | 0.0% | 10.8 $\mu$s | 13.1 / 21.7 / 34.1 $\mu$s |
| 2. Naive Pace (No Overshoot) + MPC | 19.20 s | 0.003 m | 0.0550 | 34.56 | 20.2272 | 0.0% | 10.6 $\mu$s | 12.6 / 19.8 / 52.0 $\mu$s |
| 3. Static Plan (No Replan) + MPC | 9.50 s | 0.003 m | 0.0593 | 35.19 | 10.5484 | 0.0% | 10.7 $\mu$s | 13.2 / 22.5 / 63.8 $\mu$s |
| 4. Multi-Property PPO (Unshielded) | **1.10 s** | 0.005 m | 0.0615 | 36.80 | **2.2196** | 0.0% | 10.9 $\mu$s | 13.9 / 25.4 / 75.0 $\mu$s |
| 5. Contract-Shielded PPO (Switch MPS) | 2.86 s | 0.005 m | 0.0696 | 36.31 | 3.9707 | 23.8% | 11.2 $\mu$s | 22.2 / 38.8 / 121.1 $\mu$s |
| 6. Contract-Shielded PPO (1-Step DCBF) | 1.18 s | 0.005 m | 0.0621 | 36.74 | 2.2923 | 3.9% | 11.2 $\mu$s | 22.2 / 39.7 / 60.7 $\mu$s |
| 7. Contract-Shielded PPO (UCB-DMPS) | 1.50 s | 0.005 m | **0.0529** | **35.98** | 2.5753 | 8.1% | 11.3 $\mu$s | 23.3 / 44.8 / 172.0 $\mu$s |
| 8. Bilbao (2025) CBTC-ATO (RMBTP+PI) | 4.73 s | 0.014 m | 0.1092 | 37.14 | 6.0875 | 0.0% | 19.6 $\mu$s | 19.6 / 22.2 / 387.9 $\mu$s |

#### Part B: Weather-Disturbed (Wind / Rain / Snow / Cold) Scenarios Only ($N = 15$)
| Trajectory Pilot + Contract Shield Method | $\|\text{TimeErr}\|$ (s) | $\|\text{StopErr}\|$ (m) | SmoothLoss | SpecEnergy ($\text{Wh/t}\cdot\text{km}$) | LossScore | Shield% | Core Step | Total (Mean / P99 / Max) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1. Max-Speed Greedy Cosine + MPC | 3.13 s | 0.008 m | 0.0738 | 57.12 | 4.8163 | 0.0% | 11.0 $\mu$s | 13.2 / 22.3 / 38.9 $\mu$s |
| 2. Naive Pace (No Overshoot) + MPC | 20.43 s | 0.010 m | 0.0542 | 49.87 | 21.9367 | 0.0% | 10.8 $\mu$s | 12.7 / 20.7 / 74.8 $\mu$s |
| 3. Static Plan (No Replan) + MPC | 10.35 s | 0.013 m | 0.0596 | 49.31 | 11.8981 | 0.0% | 10.9 $\mu$s | 13.4 / 23.5 / 105.7 $\mu$s |
| 4. Multi-Property PPO (Unshielded) | **2.03 s** | 0.017 m | 0.0597 | 52.76 | **3.7189** | 0.0% | 11.0 $\mu$s | 14.1 / 25.5 / 141.7 $\mu$s |
| 5. Contract-Shielded PPO (Switch MPS) | 3.99 s | 0.017 m | 0.0681 | 51.99 | 5.6849 | 26.1% | 11.2 $\mu$s | 22.4 / 39.0 / 128.6 $\mu$s |
| 6. Contract-Shielded PPO (1-Step DCBF) | 2.26 s | 0.016 m | 0.0629 | 53.57 | 3.9644 | 9.8% | 11.3 $\mu$s | 22.6 / 39.6 / 220.8 $\mu$s |
| 7. Contract-Shielded PPO (UCB-DMPS) | 2.48 s | **0.015 m** | **0.0525** | **51.98** | 4.1088 | 9.7% | 11.3 $\mu$s | 23.5 / 39.7 / 132.7 $\mu$s |
| 8. Bilbao (2025) CBTC-ATO (RMBTP+PI) | 3.36 s | 0.047 m | 0.0981 | 49.93 | 5.5146 | 0.0% | 19.6 $\mu$s | 19.6 / 25.5 / 349.2 $\mu$s |

#### Part C: United Combined Benchmark Across All 30 Scenarios ($430.9\text{ km}$ Total, $69,745$ Micro-Steps)
| Trajectory Pilot + Contract Shield Method | $\|\text{TimeErr}\|$ (s) | $\|\text{StopErr}\|$ (m) | SmoothLoss | SpecEnergy ($\text{Wh/t}\cdot\text{km}$) | LossScore | Shield% | Core Step | Total (Mean / P99 / Max) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1. Max-Speed Greedy Cosine + MPC | 3.43 s | 0.006 m | 0.0733 | 48.24 | 4.8653 | 0.0% | 10.9 $\mu$s | 13.1 / 22.0 / 38.9 $\mu$s |
| 2. Naive Pace (No Overshoot) + MPC | 19.82 s | 0.007 m | 0.0546 | 42.22 | 21.0819 | 0.0% | 10.7 $\mu$s | 12.7 / 20.2 / 74.8 $\mu$s |
| 3. Static Plan (No Replan) + MPC | 9.92 s | 0.008 m | 0.0594 | 42.25 | 11.2232 | 0.0% | 10.8 $\mu$s | 13.3 / 23.0 / 105.7 $\mu$s |
| 4. Multi-Property PPO (Unshielded) | **1.57 s** | 0.011 m | 0.0606 | 44.78 | **2.9693** | 0.0% | 10.9 $\mu$s | 14.0 / 25.4 / 141.7 $\mu$s |
| 5. Contract-Shielded PPO (Switch MPS) | 3.43 s | 0.011 m | 0.0689 | 44.15 | 4.8278 | 24.9% | 11.2 $\mu$s | 22.3 / 38.9 / 128.6 $\mu$s |
| 6. Contract-Shielded PPO (1-Step DCBF) | 1.72 s | **0.010 m** | 0.0625 | 45.15 | 3.1284 | 6.8% | 11.3 $\mu$s | 22.4 / 39.7 / 220.8 $\mu$s |
| 7. Contract-Shielded PPO (UCB-DMPS) | 1.99 s | **0.010 m** | **0.0527** | **43.98** | 3.3420 | 8.9% | 11.3 $\mu$s | 23.4 / 42.3 / 172.0 $\mu$s |
| 8. Bilbao (2025) CBTC-ATO (RMBTP+PI) | 4.05 s | 0.031 m | 0.1037 | 43.53 | 5.8010 | 0.0% | 19.6 $\mu$s | 19.6 / 23.8 / 387.9 $\mu$s |

* **Physical Tube Containment:** **99.96%** across $69,745$ micro-steps (Overall Emulator-to-Real $\text{RMSE} = 0.0205\text{ m/s}$, Empirical 99th-Percentile Error $= 0.0559\text{ m/s}$).
* **Real-Time Hardware Compliance:** Worst-case single-step latency across all $69,745$ steps for UCB-DMPS is **$172.0\text{ }\mu\text{s}$** ($< 500\text{ }\mu\text{s}$ hard real-time ceiling, $> 1,200\times$ faster than space-domain RTI-SQP and $> 21,000\times$ faster than pseudospectral NLP).

---

## Generated Diagnostic Figures

Running `dynamic_railroad_mpc.py` automatically exports two high-resolution diagnostic figures:
1. **`wind_trajectory_shield_dual_axis.png`:** Dual-horizontal-axis 4-station ($21.35\text{ km}$) trajectory profile showing cumulative track position $x\text{ (m)}$ on the bottom axis and clock time $t\text{ (s)}$ on the top axis, the Goodall 99% Scissor Band (yellow) and Safe Envelope (green), active contract interventions (red dots), disjoint wind fronts, MPC throttle/brake effort $u_{\text{act}}$, and 8-step passenger mass convergence $\hat{M}(x) \to M(x)$.
2. **`rl_cosine_mpc.png`:** 6-panel benchmark overview comparing (1) multi-station speed profiles $v(x)$, (2) Order-2 acceleration $a(t)$ and Order-3 jerk $j(t)$, (3) online 8-step passenger mass identification and weather observer $\hat{d}(t)$, (4) mean deadline error across Clean, Weather-Disturbed, and Combined runs, (5) composite `LossScore`, and (6) normalized multi-property breakdown (`SmoothLoss`, `StopErr`, and `SpecEnergy`).

---

## References

1. P. Havaei and M. A. Sandidzadeh, *"Intelligent-PID controller design for speed track in automatic train operation system with heuristic algorithms,"* Journal of Rail Transport Planning & Management, vol. 22, p. 100321, 2022.
2. D. Liu, G. Yan, J. Zhou, and B. Chen, *"MPC-Based Speed Tracking Control for Subway Trains Under Actuator Constraints,"* Actuators, vol. 14, no. 3, p. 134, 2025.
3. A. Banerjee, K. Rahmani, J. Biswas, and I. Dillig, *"Dynamic Model Predictive Shielding for Provably Safe Reinforcement Learning,"* arXiv:2405.13863, 2024.
4. A. W. Goodall and F. Belardinelli, *"Safe Reinforcement Learning via Recovery-based Shielding with Gaussian Process Dynamics Models,"* in Proc. 25th Int. Conf. on Autonomous Agents and Multiagent Systems (AAMAS), 2026.
5. K.-C. Hsu, H. Hu, and J. F. Fisac, *"The Safety Filter: A Unified View of Safety-Critical Control in Autonomous Systems,"* Annual Review of Control, Robotics, and Autonomous Systems, vol. 7, pp. 47–72, 2024.
6. F. Cano, T. A. Henzinger, and K. Kueffner, *"Shielding for Higher-Order Safety,"* arXiv:2608.03662, 2026.
7. O. Adalat, E. Hamel-De le Court, and F. Belardinelli, *"Contract-Based Compositional Shielding for Safe Multi-Agent Reinforcement Learning,"* arXiv:2606.14130, 2026.
8. W. Zhong, S. Li, H. Xu, and W. Zhang, *"On-line train speed profile generation of high-speed railway with energy-saving: A model predictive control method,"* IEEE Trans. Intell. Transp. Syst., vol. 23, no. 5, pp. 4063–4074, 2022.
9. Z. Xiao et al., *"Eco-driving for metro trains: A computationally efficient approach using convex programming,"* IEEE Trans. Veh. Technol., vol. 72, no. 8, pp. 10063–10076, 2023.
10. A. Bilbao-Moreno, I. Elorza, E. Irigoyen, and I. Trojaola, *"ATO system for CBTC-based railways: Planning and disturbance estimation for enhanced automation,"* Results in Engineering, vol. 26, p. 105357, 2025.