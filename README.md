# Safe and Smooth Automatic Train Speed Control (ATP + ATO)
### From State-Augmented LTV-MPC to Multi-Station Hierarchical Differential and Assume-Guarantee Contract Shielding

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](#)
[![Real-Time Latency](https://img.shields.io/badge/Core%20%2F%20Total%20Latency-12.2%20%C2%B5s%20%2F%2030.1%20%C2%B5s%20(%3C500%20%C2%B5s)-brightgreen.svg)](#)
[![Tube Containment](https://img.shields.io/badge/99%25%20Scissor%20Tube-99.94%25%20Contained-success.svg)](#)

## Overview

Modern railway control systems are traditionally split between **Automatic Train Protection (ATP)**, which enforces hard speed ceilings via conservative worst-case braking envelopes, and **Automatic Train Operation (ATO)**, which optimizes passenger ride comfort, traction energy efficiency, and station docking accuracy without formal multi-property safety guarantees.

This repository implements a **three-stage real-time control and shielding architecture** that unifies ATP safety certification and ATO multi-objective optimization within a single **12.2–30.1 $\mu\text{s}$** control loop (worst-case step **$225.0\,\mu\text{s} < 500\,\mu\text{s}$**):

1. **Stage 1 — State-Augmented LTV-MPC & Heteroscedastic Scissor Shielding:** Lifts the physical 1st-order drivetrain lag ($u_{\text{act}}(k) = 0.72\,u_{\text{cmd}}(k) + 0.28\,u_{\text{act}}(k-1)$) directly into a Linear Time-Varying MPC state vector $z_k = [v_k, u_{\text{act}}(k-1)]^\top$, eliminating speed-shift overshoot ($0.748\text{ m/s} \to 0.035\text{ m/s}$). Replaces offline polyhedral tube reachability sets with a two-head **Heteroscedastic Neural Physics Emulator** ($\hat{f}_\theta$) trained under Gaussian Negative Log-Likelihood (NLL) to propagate a contractive **99% Gaussian Scissor Tube** ($\hat{v}_{\text{emu}}(k) \pm 2.576\,\sigma_{\text{tube}}(k)$).
2. **Stage 2 — Short-Horizon Compression, Disturbance Observer (DOB) & Ultra-Fast Solver:** Exploits actuator lag decay ($0.28^4 \approx 0.6\%$) to compress the MPC horizon from $(N_p=10, N_c=5)$ to $(N_p=5, N_c=3)$, removes redundant PID switching, and pairs an online Disturbance Observer (DOB) with a gust-gated scissor shield to reject $1.03\text{ m/s}^2$ storm gusts ($0.0169 \to 0.0069\text{ m/s}$ tracking error).
3. **Stage 3 — Multi-Station PPO Trajectory Pilot, $C^2$ Cosine & Algebraic Soft-Curve Shapers, Tree-Guided Contract Shield & Event-Triggered Reachable Tubes:** Generalizes online trajectory generation across **1-to-5 station corridors ($4.75\text{--}30.8\text{ km}$)** under unknown passenger boarding/alighting mass shifts ($75.6\text{--}94.2\text{ t}$) and **five disturbed weather regimes** (`Windy-Dry`, `Rain-WetRail`, `Cold-Humid`, `Snow-Slush`, and high-temperature `Heat-Thermal`). Integrates event-triggered forward reachable tubes with 4-criterion mid-horizon preemption (Sun et al., 2024; Wang et al., 2026) and a Step-Adaptive Safe Contract Searching Tree with Tchebycheff multi-objective decomposition (Zhao, 2024; Wang et al., 2026).

---

## System Architecture (Stage 3 Closed-Loop Hierarchy)

```text
+---------------------------------------------------------------------------------------------------+
| 1. MACRO-STEP PPO PILOT (Every M=8 steps / 4.0s in Cruise, 0.5s in Landing | 15-D Observation)    |
|    Outputs p_task = (v_cmd, a_start, b_land) -> Target Cruise Speed + Overshoot & Ramp Rates      |
|    * Event-Triggered Detector (Sun et al. 2024; Wang et al. 2026) preempts M=8 on gust arrival    |
+-------------------------------------------------+-------------------------------------------------+
                                                  | p_task
                                                  v
+---------------------------------------------------------------------------------------------------+
| 2. HIERARCHICAL DIFFERENTIAL & ASSUME-GUARANTEE CONTRACT SHIELD (Cano + Adalat + Zhao + Wang)     |
|    * Evaluates Derivative Pressures rho_d(t) <= 1.0 across nested domains W_3 subset W_2 subset W_1|
|    * Certified Contract Library L_cert = {C0, C1_Speed, C2_JerkDamp, C3_Deadline, C4_EcoRegen}    |
|    * Step-Adaptive Safe Contract Searching Tree + Level-2 Reachable Tube Pruning + Tchebycheff    |
|    * Discounted-UCB Multi-Contract Selector + 2-Step Neural Rollout + Leg Ramp Latching           |
+-------------------------------------------------+-------------------------------------------------+
                                                  | p_safe = (v_safe, a_safe, b_safe)
                                                  v
+---------------------------------------------------------------------------------------------------+
| 3. TRAJECTORY SHAPER: C^2 HALF-COSINE OR LEARNED NON-COSINE C^2 ALGEBRAIC SOFT-CURVE               |
|    * Generates jerk-bounded C^2 transitions & dynamic landing x_brake(v, b_eff, d_hat) -> V_ref   |
+-------------------------------------------------+-------------------------------------------------+
                                                  | V_ref(k) in R^5
                                                  v
+---------------------------------------------------------------------------------------------------+
| 4. ULTRA-FAST 5x3 STATE-AUGMENTED LTV-MPC (pure_mpc_controller.py | ~11–14 us Core Execution)     |
|    * Upper-triangular scalar recurrence + closed-form 3x3 cofactor/Cramer QP + Gershgorin bound   |
+-------------------------------------------------+-------------------------------------------------+
                                                  | u_cmd(k) in [-1, 1]
                                                  v
+---------------------------------------------------------------------------------------------------+
| 5. NONLINEAR TRAIN PLANT & ONLINE DECOUPLED ESTIMATORS (dynamic_railroad_mpc.py)                  |
|    * Steps k < 8 (Platform Zone): Uses station telemetry to lock Passenger Mass M_hat             |
|    * Steps k >= 8 (Open Corridor): Locks M_hat (preserving climb phase) & tracks d_hat via DOB    |
|    * Models rho_air(T), CWR thermal expansion resistance kappa_cwr(T), & stator Cu loss kappa_Cu(T) |
+---------------------------------------------------------------------------------------------------+
```

---

## Key Technical Contributions

### 1. Decoupled 8-Step Station Mass Identification & Adaptive Weather Observer
Simultaneously estimating vehicle mass $M$ and aerodynamic/weather forces $d_{\text{wind}}$ on open track is algebraically ill-conditioned. We decouple them causally:
* **Station Departure Window ($k = 0 \dots 7$, $t \le 4.0\text{ s}$):** While departing the station platform, the train reads wayside station telemetry (anemometer $d_{\text{station}}$, ambient temperature $T_{\text{amb}}$, and rail adhesion $\mu_{\text{adh}}$) and solves for true passenger mass from wheel-rim traction response:
  $$\hat{M}_{\text{sample}}(k) = \frac{F_{\text{motor}}(k) - F_d(v_{k-1}, T_{\text{amb}})}{(1 + \gamma)\big(a_{\text{kin}}(k) - d_{\text{station}}(k)\big) + g\,\iota(x_k, T_{\text{amb}})}, \qquad \hat{M}_{\text{locked}} = \text{median}_{k=1\dots 8}\big\{\hat{M}_{\text{sample}}(k)\big\}$$
* **Open Corridor ($k \ge 8$):** The wayside link terminates, $\hat{M}_{\text{locked}}$ is locked into the LTV-MPC *without* resetting the active departure climb S-curve phase (`reset_shaper=False`), and an online Lyapunov/fuzzy-adaptive disturbance observer (Wang et al., 2026, Eqs. 20–21) adjusts gain $\alpha_h(\delta_d, \dot{\delta}_d) \in [0.42, 0.82]$ to track unseen open-line weather and wind forces $\hat{d}_{\text{wind}}(k)$.

### 2. Weather-Disturbed Physical Regimes (Including High-Temperature `Heat-Thermal`)
Across disturbed multi-station legs, the simulator models five weather regimes with peer-reviewed physical force modifications ([railroad_corridor.py](deadline_station_trackGen/railroad_corridor.py), [dynamic_railroad_mpc.py](deadline_station_trackGen/dynamic_railroad_mpc.py), detailed in Table 1 of [Report2.tex](deadline_station_trackGen/Report2.tex)):

| Regime (Prob.) | $T_{\text{amb}}$ | Air Density $\rho_{\text{air}}$ | Adhesion $\mu_{\text{adh}}$ | Wind / CWR / Cu Multipliers | Physical Impact & Literature Sources |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Clear-Dry** (Clean) | $18.0^\circ\text{C}$ | $1.007$ | $1.00$ | $0.00$ / $1.000$ / $1.000$ | Nominal dry steel-on-steel contact; zero external gusts (Havaei & Sandidzadeh 2022; Liu et al. 2025). |
| **Windy-Dry** ($0.22$) | $16.0^\circ\text{C}$ | $1.014$ ($+1.4\%$) | $1.00$ | $1.00\times$ / $1.000$ / $1.000$ | Slightly denser cool air increases Davis drag $+1.4\%$; open-corridor gusts up to $0.72\text{ m/s}^2$ (Rochard & Schmid 2000; Baker 2010). |
| **Rain-WetRail** ($0.24$) | $8.0^\circ\text{C}$ | $1.043$ ($+4.3\%$) | $0.86$ ($-14\%$) | $1.15\times$ / $1.000$ / $1.000$ | Elastohydrodynamic water film cuts peak adhesion by $14\%$; raindrop impingement & wet skin friction raise gust drag $1.15\times$ (Polach 2005; Zhang et al. 2002; Baker 2010). |
| **Cold-Humid** ($0.18$) | $-6.0^\circ\text{C}$ | $1.097$ ($+9.7\%$) | $0.92$ ($-8\%$) | $1.12\times$ / $1.000$ / $1.000$ | Cold dense air raises aerodynamic Davis drag $+9.7\%$ and yaw gusts $1.12\times$; condensation/frost film lowers adhesion by $8\%$ (Rochard & Schmid 2000; Polach 2005; Zhang et al. 2002). |
| **Snow-Slush** ($0.18$) | $-3.0^\circ\text{C}$ | $1.085$ ($+8.5\%$) | $0.80$ ($-20\%$) | $1.20\times$ / $1.000$ / $1.000$ | Compacted snow/slush shear layer drops adhesion by $20\%$; dense air plus snow ingestion/bogie ploughing raise gust bursts $1.20\times$ (Polach 2005; Zhang et al. 2002; Rochard & Schmid 2000). |
| **Heat-Thermal** ($0.18$) | $38.0^\circ\text{C}$ ($T_{\text{rail}}\approx 56^\circ\text{C}$) | $0.942$ ($-5.8\%$) | $0.94$ ($-6\%$) | $1.05\times$ / $1.0585$ / $1.0707$ | (i) Warm air lowers aero drag $-5.8\%$; (ii) CWR compressive thermal stress above $T_0=25^\circ\text{C}$ increases lateral compliance & curve creepage resistance $+5.85\%$; (iii) high rail temperature softens third-body oxide layers ($\mu_{\text{adh}}=0.94$); (iv) stator copper resistance rises $+7.07\%$ ($\alpha_{\text{Cu}}=0.00393\,\text{K}^{-1}$), increasing $I^2R$ traction loss in $\eta_{\text{trac}}$ (Kish & Samavedam 2013; Dobney et al. 2009; Polach 2005; Xiao et al. 2023; Rochard & Schmid 2000). |

### 3. $C^2$ Half-Cosine S-Curves vs. Learned Non-Cosine $C^2$ Algebraic Soft-Curve Generator
* **Half-Cosine Shaper (Method 7):** Parameterizes speed transitions and station landings as phase-tracked half-cosine profiles ($v(\tau) = \frac{v_{\text{eff}}}{2}(1 + \cos(\pi\tau / T_{\text{land}}))$), guaranteeing zero jerk at ramp endpoints.
* **Non-Cosine $C^2$ Algebraic Soft-Curve Generator (`step_free_reference_horizon`, Methods 8 & 9):** Uses a trigonometric-free $C^2$-continuous Hermite cubic-acceleration flare (quartic velocity progression $\psi_{\text{soft}}(u; \alpha)$) whose acceleration and jerk vanish continuously at all interior and exterior junctions, paired with PPO-adapted + fuzzy-softened flare width $\alpha(k) = \min(0.46, \alpha_{\text{RL}}(a_{\text{start}}, b_{\text{land}}) + 0.04(1 - \mu_y(k)))$ and an exact 5th-degree Newton-Raphson distance-phase solver ($\kappa_{\text{land}} = 1.66$).

### 4. Inductive 8-Step Safety Guarantees & Step-Adaptive Safe Contract Searching Tree
* **Event-Triggered Inductive 8-Step Preemption (Sun et al., 2024 Eq. 22; Wang et al., 2026 Eqs. 22–25):** Preempts the $M=8$ macro-window at any interior micro-step $j \in \{1,\dots,7\}$ whenever an instantaneous gust shift ($|\Delta\hat{d}| > 0.145\text{ m/s}^2$), directional CUSUM drift ($|S_{\text{cusum}}| > 0.22\text{ m/s}^2$), low fuzzy satisfaction ($\mu_{\min} < 0.15$), or remaining-horizon forward reachable tube breach ($v_{\text{tube}}^{\text{rem}} > v_{\text{lim}} + 0.02$) occurs.
* **Step-Adaptive Safe Contract Searching Tree + Tchebycheff Scoring (Zhao, 2024 Alg. 1; Wang et al., 2026 Eq. 11):** Expands Level-1 blend nodes across certified contracts $\{\mathcal{C}_1\text{--}\mathcal{C}_4\}$ with shield pruning ($h_1 \ge 0, h_2 \ge 0$), projects Level-2 child states $(x_{\text{leaf}}, v_{\text{leaf}})$ and contractive reachable tube variances $\sigma_{\text{leaf}}^2$ across $M_{\text{rem}} = \max(2, M - (k \bmod M))$ with ceiling pruning, scores via Tchebycheff multi-objective decomposition against $Z^* = (0.0, 0.42, 0.22, 0.48)$, backpropagates Bellman tree returns + Discounted-UCB ($\eta = 0.95, \beta_{\text{ucb}} = 0.22$), and synchronizes damped ramp caps $(a_{\text{lat}}, b_{\text{lat}})$ with exact feasible deadline pace $v_{\text{feas}}(a_{\text{lat}}, b_{\text{lat}})$.

### 5. Sub-35 $\mu\text{s}$ Real-Time Numerical Engineering
* **Upper-Triangular Scalar Recurrence:** Builds $F, \Phi, \Psi$ without matrix multiplications.
* **Closed-Form $3 \times 3$ Cofactor QP + $O(1)$ Gershgorin Bound:** Replaces LAPACK `np.linalg.solve` and `eigvalsh` calls.
* **Zero-Copy In-Place NumPy Parameter Views:** Evaluates the PyTorch Actor MLP and SiLU Physics Emulator directly in pre-allocated NumPy buffers (`12.2 us` core step, `30.1 us` total shielded step).

---

## Repository Structure

| File | Description |
| :--- | :--- |
| [deadline_station_trackGen/railroad_corridor.py](deadline_station_trackGen/railroad_corridor.py) | **Module 1:** Generalized 1-to-5 station railway generator ($4.75\text{--}30.8\text{ km}$), 5-zone spatial speed limits, anticipatory $C^1$ cosine & $C^2$ algebraic spatial ceilings, stochastic passenger mass model, and 5 disturbed weather/thermal regimes (`Windy-Dry`, `Rain-WetRail`, `Cold-Humid`, `Snow-Slush`, `Heat-Thermal`). |
| [deadline_station_trackGen/cosine_trajectory_shaper.py](deadline_station_trackGen/cosine_trajectory_shaper.py) | **Module 2:** Phase-tracked $C^2$ half-cosine S-curve generator and trigonometric-free $C^2$ Hermite algebraic soft-curve generator (`step_free_reference_horizon`) with online fuzzy softening. |
| [deadline_station_trackGen/goodall_trajectory_shield.py](deadline_station_trackGen/goodall_trajectory_shield.py) | **Module 3:** Heteroscedastic 99% Scissor Tube propagation, hierarchical derivative pressures ($\rho_1, \rho_2, \rho_3, \rho_E, \rho_T$), Step-Adaptive Safe Contract Searching Tree, Tchebycheff multi-objective scoring, and Switch / 1-Step DCBF / UCB-DMPS modes. |
| [deadline_station_trackGen/dynamic_railroad_mpc.py](deadline_station_trackGen/dynamic_railroad_mpc.py) | **Module 4 (Main Runner):** 15-feature Multi-Station RL environment (`MPCTransitionRailroadEnv`), 8-step mass ID, adaptive DOB, 4-criterion mid-horizon preemption, thermal-resistance & EDS energy model, PPO+GAE trainer, latency spike profiler, and 10-method benchmark suite. |
| [deadline_station_trackGen/bilbao_cbtc_ato.py](deadline_station_trackGen/bilbao_cbtc_ato.py) | Standalone implementation of the Bilbao-Moreno et al. (2025) two-layer CBTC-ATO controller (RMBTP Regula Falsi Mesoscopic Planner + Microscopic Cascade PI-Position/PI-Speed). |
| [deadline_station_trackGen/safety_shield.py](deadline_station_trackGen/safety_shield.py) | Standalone Stage 1 & Stage 2 atomic safety filter and analytical scissor band diagnostics. |
| [pure_mpc_controller.py](pure_mpc_controller.py) | State-augmented LTV-MPC ($z_k = [v_k, u_{\text{act}}(k-1)]^\top$) with horizon $N_p=5, N_c=3$, causal plateau lookahead guard, and fast box-constrained QP solver. |
| [havaei_generalizing_rl.py](havaei_generalizing_rl.py) | Nonlinear subway train plant (`NonlinearTrainPhysics`), Davis drag, grade/curve resistance, and `LearnedLinearizationModel` heteroscedastic neural network. |
| [deadline_station_trackGen/Report2.tex](deadline_station_trackGen/Report2.tex) & [CHANGES.md](deadline_station_trackGen/CHANGES.md) | Full LaTeX technical report (Stages 1–3, weather physics table & sources, mathematical derivations, and verified tables) and detailed changelog. |

---

## Installation & Quick Start

### 1. Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy torch matplotlib
```

### 2. Instant Evaluation Using Pre-Trained Checkpoints
Run from inside `deadline_station_trackGen/` to evaluate all 10 methods across 30 randomized multi-station scenarios (using `best_emulator.pt`, `best_ppo_cosine_pilot.pt`, and `best_ppo_free_pilot.pt`):
```bash
cd deadline_station_trackGen
python dynamic_railroad_mpc.py --eval-runs 30 --n-legs 0 --wind-mode mixed --load-checkpoint best_ppo_cosine_pilot.pt
```

### 3. Full Training & Evaluation from Scratch
To train the 150-step Heteroscedastic Physics Emulator and both 15-feature PPO Pilots (Cosine and Soft-Curve) across 220 epochs:
```bash
cd deadline_station_trackGen
python dynamic_railroad_mpc.py --epochs 220 --eval-runs 30 --n-legs 0 --wind-mode mixed
```

### CLI Options
* `--epochs`: Number of PPO training epochs (default: `220`).
* `--eval-runs`: Number of randomized multi-station test scenarios (default: `30`).
* `--n-legs`: Number of station-to-station legs per journey (`0` = randomly sample `1..5` stations per scenario; or set fixed `1..5`).
* `--wind-mode`: Weather disturbance mode (`mixed` = 15 Clean + 15 Weather-Disturbed; `disjoint` = sporadic gusts + weather on all runs; `continuous` = evolving wind; `none` = clean only).
* `--load-checkpoint`: Path to pre-trained cosine PPO `.pt` file to bypass training (`best_ppo_free_pilot.pt` and `best_emulator.pt` are loaded automatically if present).

---

## Experimental Results

### Stage 1 & Stage 2 Summary (Single-Corridor Speed Tracking)
* **Stage 1 (30 Unseen Trajectories, No Wind):** Lifting actuator delay into LTV-MPC reduces shift overshoot by **95.3%** ($0.7480\text{ m/s} \to 0.0350\text{ m/s}$) and stopping error from $57.1\text{ cm}$ to **$8.2\text{ cm}$** ($0.0031\text{ m/s}$ tracking RMSE).
* **Stage 2 (30 Unseen Trajectories, $1.03\text{ m/s}^2$ Storm Gusts):** Compressing to $N_p=5, N_c=3$ with DOB and gust-gated Scissor Shielding halves wind-induced tracking error ($0.0169\text{ m/s} \to \mathbf{0.0069\text{ m/s}}$) and improves stopping accuracy to **$10.6\text{ cm}$**.

### Stage 3 Generalized Multi-Station Benchmark ($N = 30$ Scenarios, $1\text{--}5$ Stations, $4.75\text{--}30.8\text{ km}$)
Evaluated across 84 inter-station legs (`37 Clear-Dry`, `8 Windy-Dry`, `8 Rain-WetRail`, `10 Cold-Humid`, `12 Snow-Slush`, `9 Heat-Thermal`) under the loss-consistent multi-objective metric:
$$\text{LossScore} = 1.0\,|\text{TimeErr}| + 15.0\,|\text{StopErr}|_{\text{m}} + 2.0\,\text{SmoothLoss} + 0.025\,\text{SpecEnergy}$$

#### Part A: Clean / No-Weather Scenarios Only ($N = 15$)
| Trajectory Pilot + Contract Shield Method | $\|\text{TimeErr}\|$ (s) | $\|\text{StopErr}\|$ (m) | SmoothLoss | SpecEnergy ($\text{Wh/t}\cdot\text{km}$) | LossScore | Shield% | Core Step | Total (Mean / P99 / Max) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1. Max-Speed Greedy Cosine + MPC | 4.58 s | 0.004 m | 0.0632 | 38.86 | 5.7346 | 0.0% | 11.6 $\mu$s | 14.1 / 23.8 / 90.4 $\mu$s |
| 2. Naive Pace (No Overshoot) + MPC | 18.44 s | 0.003 m | 0.0502 | 34.24 | 19.4508 | 0.0% | 11.1 $\mu$s | 13.2 / 20.9 / 72.8 $\mu$s |
| 3. Static Plan (No Replan) + MPC | 6.02 s | 0.003 m | 0.0559 | 35.06 | 7.0558 | 0.0% | 11.1 $\mu$s | 13.8 / 23.7 / 113.0 $\mu$s |
| 4. Multi-Property PPO (Unshielded) | 0.87 s | 0.003 m | 0.0558 | 36.40 | 1.9405 | 0.0% | 11.4 $\mu$s | 14.7 / 26.8 / 76.9 $\mu$s |
| 5. Contract-Shielded PPO (Switch MPS) | 3.03 s | 0.004 m | 0.0698 | 35.81 | 4.1276 | 25.6% | 11.6 $\mu$s | 23.5 / 40.5 / 59.8 $\mu$s |
| 6. Contract-Shielded PPO (1-Step DCBF) | 0.98 s | 0.005 m | 0.0569 | 36.41 | 2.0819 | 4.8% | 11.7 $\mu$s | 23.7 / 42.4 / 176.2 $\mu$s |
| 7. Contract-Shielded PPO (UCB-DMPS Cosine) | 0.44 s | **0.003 m** | **0.0467** | **35.27** | 1.4564 | 5.9% | 12.1 $\mu$s | 29.7 / 98.7 / 225.0 $\mu$s |
| 8. Soft-Curve PPO (No Cosine, Unshielded) | 0.41 s | 0.005 m | 0.0582 | 35.95 | 1.5037 | 0.0% | 12.9 $\mu$s | 18.9 / 40.7 / 97.5 $\mu$s |
| 9. Soft-Curve PPO + Jerk-Horizon UCB-DMPS | **0.34 s** | 0.005 m | 0.0525 | 35.90 | **1.4244** | 5.3% | 13.9 $\mu$s | 36.2 / 72.0 / 261.8 $\mu$s |
| 10. Bilbao (2025) CBTC-ATO (RMBTP+PI) | 4.73 s | 0.014 m | 0.1092 | 37.14 | 6.0875 | 0.0% | 20.1 $\mu$s | 20.1 / 26.4 / 413.2 $\mu$s |

#### Part B: Weather-Disturbed (Wind / Rain / Snow / Cold / Heat-Thermal) Scenarios Only ($N = 15$)
| Trajectory Pilot + Contract Shield Method | $\|\text{TimeErr}\|$ (s) | $\|\text{StopErr}\|$ (m) | SmoothLoss | SpecEnergy ($\text{Wh/t}\cdot\text{km}$) | LossScore | Shield% | Core Step | Total (Mean / P99 / Max) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1. Max-Speed Greedy Cosine + MPC | 3.88 s | 0.006 m | 0.0659 | 56.87 | 5.5282 | 0.0% | 11.7 $\mu$s | 14.4 / 25.0 / 108.4 $\mu$s |
| 2. Naive Pace (No Overshoot) + MPC | 19.50 s | 0.007 m | 0.0493 | 49.20 | 20.9398 | 0.0% | 11.3 $\mu$s | 13.6 / 21.3 / 67.8 $\mu$s |
| 3. Static Plan (No Replan) + MPC | 6.83 s | 0.014 m | 0.0574 | 48.82 | 8.3723 | 0.0% | 11.7 $\mu$s | 15.8 / 27.1 / 115.9 $\mu$s |
| 4. Multi-Property PPO (Unshielded) | 1.57 s | 0.013 m | 0.0547 | 52.21 | 3.1833 | 0.0% | 11.8 $\mu$s | 15.4 / 27.6 / 75.8 $\mu$s |
| 5. Contract-Shielded PPO (Switch MPS) | 4.02 s | 0.014 m | 0.0681 | 50.89 | 5.6318 | 27.2% | 11.9 $\mu$s | 24.2 / 41.9 / 138.4 $\mu$s |
| 6. Contract-Shielded PPO (1-Step DCBF) | 1.94 s | 0.013 m | 0.0573 | 53.02 | 3.5776 | 11.5% | 12.1 $\mu$s | 24.6 / 44.2 / 163.6 $\mu$s |
| 7. Contract-Shielded PPO (UCB-DMPS Cosine) | **0.97 s** | **0.007 m** | **0.0457** | **49.45** | **2.3990** | 9.8% | 12.3 $\mu$s | 30.5 / 75.1 / 177.0 $\mu$s |
| 8. Soft-Curve PPO (No Cosine, Unshielded) | 1.60 s | 0.011 m | 0.0557 | 50.16 | 3.1375 | 0.0% | 13.5 $\mu$s | 20.2 / 45.1 / 109.4 $\mu$s |
| 9. Soft-Curve PPO + Jerk-Horizon UCB-DMPS | 1.44 s | 0.010 m | 0.0499 | 50.03 | 2.9457 | 9.1% | 13.9 $\mu$s | 36.5 / 80.4 / 350.4 $\mu$s |
| 10. Bilbao (2025) CBTC-ATO (RMBTP+PI) | 3.25 s | 0.035 m | 0.0972 | 49.89 | 5.2158 | 0.0% | 19.7 $\mu$s | 19.7 / 23.9 / 240.0 $\mu$s |

#### Part C: United Combined Benchmark Across All 30 Scenarios ($430.9\text{ km}$ Total, $66,975$ Active Driving Steps)
| Trajectory Pilot + Contract Shield Method | $\|\text{TimeErr}\|$ (s) | $\|\text{StopErr}\|$ (m) | SmoothLoss | SpecEnergy ($\text{Wh/t}\cdot\text{km}$) | LossScore | Shield% | Core Step | Total (Mean / P99 / Max) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1. Max-Speed Greedy Cosine + MPC | 4.23 s | 0.005 m | 0.0645 | 47.87 | 5.6314 | 0.0% | 11.7 $\mu$s | 14.2 / 24.4 / 108.4 $\mu$s |
| 2. Naive Pace (No Overshoot) + MPC | 18.97 s | 0.005 m | 0.0498 | 41.72 | 20.1953 | 0.0% | 11.2 $\mu$s | 13.4 / 21.1 / 72.8 $\mu$s |
| 3. Static Plan (No Replan) + MPC | 6.42 s | 0.009 m | 0.0566 | 41.94 | 7.7141 | 0.0% | 11.4 $\mu$s | 14.8 / 25.4 / 115.9 $\mu$s |
| 4. Multi-Property PPO (Unshielded) | 1.22 s | 0.008 m | 0.0553 | 44.31 | 2.5619 | 0.0% | 11.6 $\mu$s | 15.1 / 27.2 / 76.9 $\mu$s |
| 5. Contract-Shielded PPO (Switch MPS) | 3.52 s | 0.009 m | 0.0689 | 43.35 | 4.8797 | 26.4% | 11.7 $\mu$s | 23.9 / 41.2 / 138.4 $\mu$s |
| 6. Contract-Shielded PPO (1-Step DCBF) | 1.46 s | 0.009 m | 0.0571 | 44.72 | 2.8297 | 8.1% | 11.9 $\mu$s | 24.1 / 43.3 / 176.2 $\mu$s |
| **7. Contract-Shielded PPO (UCB-DMPS Cosine)** | **0.70 s** | **0.005 m** | **0.0462** | **42.36** | **1.9277** | 7.9% | 12.2 $\mu$s | 30.1 / 86.9 / 225.0 $\mu$s |
| 8. Soft-Curve PPO (No Cosine, Unshielded) | 1.01 s | 0.008 m | 0.0570 | 43.05 | 2.3206 | 0.0% | 13.2 $\mu$s | 19.6 / 42.9 / 109.4 $\mu$s |
| **9. Soft-Curve PPO + Jerk-Horizon UCB-DMPS** | **0.89 s** | 0.008 m | **0.0512** | **42.96** | **2.1850** | 7.2% | 13.9 $\mu$s | 36.3 / 76.2 / 350.4 $\mu$s |
| 10. Bilbao (2025) CBTC-ATO (RMBTP+PI) | 3.99 s | 0.024 m | 0.1032 | 43.52 | 5.6517 | 0.0% | 19.9 $\mu$s | 19.9 / 25.2 / 413.2 $\mu$s |

* **Physical Tube Containment:** **99.94%** across $66,975$ active driving micro-steps (excluding passive station dwell; Overall Emulator-to-Real $\text{RMSE} = 0.0196\text{ m/s}$, Empirical 99th-Percentile Error $= 0.0519\text{ m/s}$).
* **Real-Time Hardware Compliance:** Worst-case single-step latency across all $66,975$ steps for UCB-DMPS Cosine is **$225.0\text{ }\mu\text{s}$** ($< 500\text{ }\mu\text{s}$ hard real-time ceiling, $> 1,200\times$ faster than space-domain RTI-SQP and $> 20,000\times$ faster than pseudospectral NLP).

---

## Generated Diagnostic Figures

Running `dynamic_railroad_mpc.py` automatically exports two high-resolution diagnostic figures inside `deadline_station_trackGen/` (and copies `rl_cosine_mpc.png` to the project root):
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
9. Z. Xiao, N. Murgovski, M. Chen, X. Feng, Q. Wang, and P. Sun, *"Eco-driving for metro trains: A computationally efficient approach using convex programming,"* IEEE Trans. Veh. Technol., vol. 72, no. 8, pp. 10063–10076, 2023.
10. A. Bilbao-Moreno, I. Elorza, E. Irigoyen, and I. Trojaola, *"ATO system for CBTC-based railways: Planning and disturbance estimation for enhanced automation,"* Results in Engineering, vol. 26, p. 105357, 2025.
11. P. Sun, J. Zhang, K. Yan, B. Yao, and Q. Wang, *"Event-triggered tube model predictive control for high-speed trains trajectory tracking based on robust reachable set,"* Proc. SPIE 13064, p. 130640F, 2024.
12. Y. Wang, J. Wang, and J. Chen, *"Optimized Model Predictive Controller Using Multi-Objective Whale Optimization Algorithm for Urban Rail Train Tracking Control,"* Biomimetics, vol. 11, no. 1, p. 60, 2026.
13. Z. Zhao, *"How to ensure a safe control strategy? Towards a SRL for urban transit autonomous operation,"* arXiv:2311.14457v2, 2024.
14. B. P. Rochard and F. Schmid, *"A review of methods to measure and calculate train resistances,"* Proc. IMechE Part F: J. Rail and Rapid Transit, vol. 214, no. 4, pp. 185–199, 2000.
15. O. Polach, *"Creep forces in simulations of traction vehicles running on adhesion limit,"* Wear, vol. 258, no. 7–8, pp. 992–1000, 2005.
16. W. H. Zhang, J. Z. Chen, X. J. Wu, and X. S. Jin, *"Wheel/rail adhesion and analysis by using full scale roller rig,"* Wear, vol. 253, no. 1–2, pp. 82–88, 2002.
17. C. J. Baker, *"The flow around high speed trains,"* J. Wind Eng. Ind. Aerodyn., vol. 98, no. 6–7, pp. 277–298, 2010.
18. A. Kish and G. Samavedam, *Track Buckling Prevention: Theory, Safety Concepts, and Applications*, U.S. DOT Federal Railroad Administration Report DOT/FRA/ORD-13/16, 2013.
19. K. Dobney, C. J. Baker, A. D. Quinn, and L. Chapman, *"Quantifying the effects of high summer temperatures due to climate change on buckling and rail related delays in south-east United Kingdom,"* Meteorological Applications, vol. 16, no. 2, pp. 245–258, 2009.
