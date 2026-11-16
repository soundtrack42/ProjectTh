# Safe and Smooth Automatic Train Speed Control (ATP + ATO)
### From State-Augmented LTV-MPC to Multi-Station Hierarchical Differential and Assume-Guarantee Contract Shielding

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](#)
[![Real-Time Latency](https://img.shields.io/badge/Core%20%2F%20Total%20Latency-11.8%20%C2%B5s%20%2F%2027.1%20%C2%B5s%20(%3C500%20%C2%B5s)-brightgreen.svg)](#)
[![Tube Containment](https://img.shields.io/badge/99%25%20Scissor%20Tube-99.96%25%20Contained-success.svg)](#)

## Overview

Modern railway control systems are traditionally split between **Automatic Train Protection (ATP)**, which enforces hard speed ceilings via conservative worst-case braking envelopes, and **Automatic Train Operation (ATO)**, which optimizes passenger ride comfort, traction energy efficiency, and station docking accuracy without formal multi-property safety guarantees.

This repository implements a **three-stage real-time control and shielding architecture** that unifies ATP safety certification and ATO multi-objective optimization within a single **11.8–27.1 $\mu\text{s}$** control loop (worst-case step **$223.9\,\mu\text{s} < 500\,\mu\text{s}$**):

1. **Stage 1 — State-Augmented LTV-MPC & Heteroscedastic Scissor Shielding:** Lifts the physical 1st-order drivetrain lag ($u_{\text{act}}(k) = 0.72\,u_{\text{cmd}}(k) + 0.28\,u_{\text{act}}(k-1)$) directly into a Linear Time-Varying MPC state vector $z_k = [v_k, u_{\text{act}}(k-1)]^\top$, eliminating speed-shift overshoot ($0.748\text{ m/s} \to 0.035\text{ m/s}$). Uses an **8-feature Heteroscedastic Neural Physics Emulator** ($\hat{f}_\theta$, all 8 inputs active) trained under Gaussian NLL from actual prediction residuals (`1e-8` numerical floor) to propagate a contractive **99% Gaussian Scissor Tube** ($\hat{v}_{\text{emu}}(k) \pm 2.576\,\sigma_{\text{tube}}(k)$).
2. **Stage 2 — Short-Horizon Compression, Disturbance Observer (DOB) & Ultra-Fast Solver:** Exploits actuator lag decay ($0.28^4 \approx 0.6\%$) to compress the MPC horizon from $(N_p=10, N_c=5)$ to $(N_p=5, N_c=3)$, verifies unbiased Hybrid RL selection (`76.23%` MPC unshielded / `88.16%` DMPS-shielded without PID penalty bias), and pairs an online Disturbance Observer (DOB) with a gust-gated scissor shield to reject $1.03\text{ m/s}^2$ storm gusts.
3. **Stage 3 — Multi-Station 18-D Outcome-Trained PPO Pilot, Dual Analytical $C^2$ Shapers, Full-Contract Tree Shield & Real-World Validation:** Generalizes online trajectory generation across **1-to-5 station corridors ($4.75\text{--}27.80\text{ km}$)** under unknown passenger boarding/alighting mass shifts ($75.3\text{--}99.2\text{ t}$) and **five disturbed weather regimes** (`Windy-Dry`, `Rain-WetRail`, `Cold-Humid`, `Snow-Slush`, and high-temperature `Heat-Thermal`), plus live **Finnish Digitraffic + Open-Meteo** API runs. Integrates causal **telescoping 3-block second-difference IRLS MAP mass ID**, **18-feature outcome-trained PPO** (with forward slope lookahead and adhesion deficit, zero imitation bias), full 4-contract Discounted-UCB tree search updated on causal mechanical energy excess with **forward "we're on track" zone-wise remaining-time slack + downhill/curve coasting ($\mathcal{C}_4$)**, downward-only $\mathcal{C}_1$ speed clamping, immediate-release 2-step latch (`13.5%` / `11.4%` active eco-plus-safety intervention rates cutting energy below unshielded PPO), and strictly equal station dwells.

---

## System Architecture (Stage 3 Closed-Loop Hierarchy)

```text
+---------------------------------------------------------------------------------------------------+
| 1. MACRO-STEP PPO PILOT (Every M=8 steps / 4.0s Cruise, 0.5s Landing | 18-D Forward-Slope Obs)   |
|    Outputs p_task = (v_cmd, a_start, b_land) -> Trained on Pure Multi-Objective Physical Outcomes |
|    * Event-Triggered Detector (Sun et al. 2024; Wang et al. 2026) preempts M=8 on gust arrival    |
+-------------------------------------------------+-------------------------------------------------+
                                                  | p_task
                                                  v
+---------------------------------------------------------------------------------------------------+
| 2. HIERARCHICAL DIFFERENTIAL & ASSUME-GUARANTEE CONTRACT SHIELD (Cano + Adalat + Zhao + Wang)     |
|    * Evaluates Derivative Pressures rho_d(t) <= 1.0 + Forward 'On-Track' Zone/Stop Remaining-Time Slack |
|    * Certified Contract Library L_cert = {C0, C1_Speed, C2_JerkDamp, C3_Deadline, C4_EcoRegen}    |
|    * Full-Contract Searching Tree (all 4 contracts C1..C4 x 3 blends) + Full 2.576*sigma Leaf Tube|
|    * Causal Energy-Excess Discounted-UCB + 2-Step Rollout + Immediate-Clear Latch (13.5% Shield%)|
+-------------------------------------------------+-------------------------------------------------+
                                                  | p_safe = (v_safe, a_safe, b_safe)
                                                  v
+---------------------------------------------------------------------------------------------------+
| 3. ANALYTICAL TRAJECTORY SHAPER: C^2 HALF-COSINE OR PPO-GUIDED C^2 ALGEBRAIC SOFT-CURVE           |
|    * Generates jerk-bounded C^2 transitions & dynamic landing x_brake(v, b_eff, d_hat) -> V_ref   |
+-------------------------------------------------+-------------------------------------------------+
                                                  | V_ref(k) in R^5
                                                  v
+---------------------------------------------------------------------------------------------------+
| 4. ULTRA-FAST 5x3 STATE-AUGMENTED LTV-MPC (pure_mpc_controller.py | ~11–13 us Core Execution)     |
|    * Upper-triangular scalar recurrence + closed-form 3x3 cofactor/Cramer QP + Gershgorin bound   |
+-------------------------------------------------+-------------------------------------------------+
                                                  | u_cmd(k) in [-1, 1]
                                                  v
+---------------------------------------------------------------------------------------------------+
| 5. NONLINEAR TRAIN PLANT & ONLINE CAUSAL ESTIMATORS (dynamic_railroad_mpc.py)                     |
|    * Steps k in [8, 63]: Telescoping 3-block 2nd-diff IRLS MAP M_hat (lag f_k + wind drift cancel)|
|    * Steps k >= 8: Lyapunov-adaptive DOB tracks unseen open-line weather & thermal drag d_hat     |
|    * Models rho_air(T), CWR thermal expansion resistance kappa_cwr(T), & stator Cu loss kappa_Cu(T) |
+---------------------------------------------------------------------------------------------------+
```

---

## Key Technical Contributions

### 1. Decoupled Telescoping 3-Block 2nd-Diff MAP Mass ID & Adaptive Weather Observer
Simultaneously estimating vehicle mass $M$ and aerodynamic/weather forces $d_{\text{wind}}$ on open track is ill-conditioned if linear wind ramps or sensor noise corrupt finite-differenced acceleration. We decouple them causally:
* **Causal Departure Window ($k \in [8, 63]$):** Integrates per-step lagged force sums $f_k$ (preserving exact $0.72/0.28$ actuator lag) and velocity-telescoped acceleration blocks across triple windows $[s, s+W), [s+W, s+2W), [s+2W, s+3W)$ ($W \in \{1,2,4,8\}$ when empirically smooth, $W \in \{4,8,12\}$ when noisy/turbulent). Taking block second differences cancels both constant wind offsets and linear wind ramps identically while telescoping intermediate tachometer noise (`984.6 kg -> 1.4 kg` MAE across 22 clean/smooth-wind legs, `max = 10.7 kg`; `910.6 kg -> 774.0 kg` MAE, `med = 516.2 kg` under `±0.004 m/s` tach noise + `+7–18%` unmodeled drag + OU turbulence).
* **Open Corridor ($k \ge 8$):** An online Lyapunov/fuzzy-adaptive disturbance observer with sensor-noise deadband adjusts gain $\alpha_h(\delta_d, \dot{\delta}_d)$ to track unseen open-line weather and wind forces $\hat{d}_{\text{wind}}(k)$.

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

### 3. Analytical $C^2$ Half-Cosine S-Curves vs. PPO-Guided Non-Cosine $C^2$ Algebraic Soft-Curve Shaper
* **Half-Cosine Shaper (Method 7):** Analytical phase-tracked half-cosine profiles ($v(\tau) = \frac{v_{\text{eff}}}{2}(1 + \cos(\pi\tau / T_{\text{land}}))$), guaranteeing zero jerk at ramp endpoints.
* **Non-Cosine $C^2$ Algebraic Soft-Curve Shaper (`step_free_reference_horizon`, Methods 8 & 9):** Analytical trigonometric-free $C^2$-continuous Hermite cubic-acceleration flare (quartic velocity progression $\psi_{\text{soft}}(u; \alpha)$) guided online by PPO-learned comfort actions $\alpha_{\text{RL}}(a_{\text{start}}, b_{\text{land}})$ and fuzzy tracking softening $\alpha(k) = \min(0.46, \alpha_{\text{RL}} + 0.04(1 - \mu_y(k)))$.

### 4. Full-Contract Persistent UCB Tree, Causal Energy Feedback & Low-Chatter Latch
* **Event-Triggered Inductive 8-Step Preemption (Sun et al., 2024 Eq. 22; Wang et al., 2026 Eqs. 22–25):** Preempts the $M=8$ macro-window whenever an instantaneous gust shift, directional CUSUM drift, low fuzzy satisfaction, or forward reachable-tube breach occurs.
* **Full 4-Contract Searching Tree + Tchebycheff Scoring (Zhao, 2024 Alg. 1; Wang et al., 2026 Eq. 11):** Expands **all four active recovery contracts** $\{\mathcal{C}_1\text{--}\mathcal{C}_4\} \times \lambda \in \{0.55, 0.85, 1.0\}$, evaluates macro traction/regen energy at tree leaves (`_estimate_macro_energy_proxy`), updates Discounted-UCB online from realized causal mechanical energy excess ($\Delta E_{\text{net}} - \Delta E_{\text{kin}} - W_{\text{grade+curve}} - W_{\text{drag}} - W_{\text{wind}}$), clamps $\mathcal{C}_1$ downward only (`v_ppo_safe`), triggers forward physics "we're on track" remaining-time slack + downhill/curve eco-coasting (`estimate_remaining_time_at_speed` + `_compute_eco_speed_limit`, lowering $\mathcal{C}_4$ energy below unshielded PPO), and clears the short 2-step ramp latch immediately once $h_{i^*} \ge 0$ (`Shield% = 13.5%` Cosine / `11.4%` Soft-Curve).

### 5. Sub-30 $\mu\text{s}$ Real-Time Numerical Engineering
* **Upper-Triangular Scalar Recurrence:** Builds $F, \Phi, \Psi$ without matrix multiplications.
* **Closed-Form $3 \times 3$ Cofactor QP + $O(1)$ Gershgorin Bound:** Replaces LAPACK `np.linalg.solve` and `eigvalsh` calls.
* **Zero-Copy In-Place NumPy Parameter Views:** Evaluates the PyTorch Actor MLP and 8-feature SiLU Physics Emulator directly in pre-allocated NumPy buffers (`11.8 us` core step, `27.1 us` total shielded step).

---

## Repository Structure

| File | Description |
| :--- | :--- |
| [deadline_station_trackGen/railroad_corridor.py](deadline_station_trackGen/railroad_corridor.py) | **Module 1:** Generalized 1-to-5 station railway generator ($4.75\text{--}27.80\text{ km}$), 5-zone spatial speed limits, anticipatory $C^1$ cosine & $C^2$ algebraic spatial ceilings, stochastic passenger mass model, and 5 disturbed weather/thermal regimes (`Windy-Dry`, `Rain-WetRail`, `Cold-Humid`, `Snow-Slush`, `Heat-Thermal`). |
| [deadline_station_trackGen/cosine_trajectory_shaper.py](deadline_station_trackGen/cosine_trajectory_shaper.py) | **Module 2:** Analytical phase-tracked $C^2$ half-cosine S-curve and trigonometric-free $C^2$ Hermite algebraic soft-curve shapers (`step_free_reference_horizon`) with online fuzzy softening. |
| [deadline_station_trackGen/goodall_trajectory_shield.py](deadline_station_trackGen/goodall_trajectory_shield.py) | **Module 3:** Full 8-feature Heteroscedastic 99% Scissor Tube propagation, hierarchical derivative pressures ($\rho_1, \rho_2, \rho_3, \rho_E, \rho_T$) with downhill gravity coasting, full 4-contract Searching Tree, persistent Discounted-UCB, and honest latch accounting. |
| [deadline_station_trackGen/dynamic_railroad_mpc.py](deadline_station_trackGen/dynamic_railroad_mpc.py) | **Module 4 (Main Runner):** 18-feature forward-slope & adhesion-aware Multi-Station RL env (`MPCTransitionRailroadEnv`), multi-lag causal MAP mass ID, adaptive DOB, 4-criterion preemption, pure outcome-trained PPO+GAE trainer, real-world Digitraffic/Open-Meteo hook, and 10-method benchmark suite. |
| [deadline_station_trackGen/real_world_rail_weather.py](deadline_station_trackGen/real_world_rail_weather.py) | Live Finnish Digitraffic train timetable/GPS fetcher paired with Open-Meteo wind/temperature/precipitation profiles and optional harsh extreme-noise injection. |
| [deadline_station_trackGen/bilbao_cbtc_ato.py](deadline_station_trackGen/bilbao_cbtc_ato.py) | Standalone implementation of the Bilbao-Moreno et al. (2025) two-layer CBTC-ATO controller (RMBTP Regula Falsi Mesoscopic Planner + Microscopic Cascade PI-Position/PI-Speed). |
| [deadline_station_trackGen/safety_shield.py](deadline_station_trackGen/safety_shield.py) | Standalone Stage 1 & Stage 2 safety filter using all 8 active neural emulator channels and analytical scissor band diagnostics. |
| [pure_mpc_controller.py](pure_mpc_controller.py) | State-augmented LTV-MPC ($z_k = [v_k, u_{\text{act}}(k-1)]^\top$) with horizon $N_p=5, N_c=3$, causal plateau lookahead guard, and fast box-constrained QP solver. |
| [havaei_generalizing_rl.py](havaei_generalizing_rl.py) | Nonlinear subway train plant (`NonlinearTrainPhysics`), Davis drag, grade/curve resistance, active 8-feature `LearnedLinearizationModel`, and unbiased 5-action Hybrid RL selector. |
| [deadline_station_trackGen/Report2.tex](deadline_station_trackGen/Report2.tex) & [CHANGES.md](deadline_station_trackGen/CHANGES.md) | Full LaTeX technical report and detailed changelog aligned with current implementation and benchmark measurements. |

---

## Installation & Quick Start

### 1. Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy torch matplotlib
```

### 2. Instant Evaluation Using Pre-Trained Checkpoints
Run from inside `deadline_station_trackGen/` to evaluate all 10 methods across 30 randomized multi-station scenarios (using `best_emulator.pt`, `best_ppo_cosine_pilot.pt`, and `best_ppo_free_pilot.pt`), or pass `--real-data --eval-runs 10` for Finnish Digitraffic + Open-Meteo evaluation:
```bash
cd deadline_station_trackGen
python dynamic_railroad_mpc.py --eval-runs 30 --n-legs 0 --wind-mode mixed --load-checkpoint best_ppo_cosine_pilot.pt
python dynamic_railroad_mpc.py --load-checkpoint best_ppo_cosine_pilot.pt --real-data --eval-runs 10 --n-legs 0 --wind-mode mixed
```

### 3. Full Training & Evaluation from Scratch
To train the 8-feature 150-step Heteroscedastic Physics Emulator and both 18-feature PPO Pilots (Cosine and Soft-Curve) across 220 epochs:
```bash
cd deadline_station_trackGen
python dynamic_railroad_mpc.py --epochs 220 --eval-runs 30 --n-legs 0 --wind-mode mixed
```

### CLI Options
* `--epochs`: Number of PPO training epochs (default: `220`).
* `--eval-runs`: Number of randomized multi-station test scenarios (default: `30`).
* `--n-legs`: Number of station-to-station legs per journey (`0` = randomly sample `1..5` stations per scenario; or set fixed `1..5`).
* `--wind-mode`: Weather disturbance mode (`mixed`, `disjoint`, `continuous`, `none`).
* `--real-data`: Evaluate on live Finnish Digitraffic train timetables + Open-Meteo weather profiles (default `--harsh-weather mixed`).
* `--load-checkpoint`: Path to pre-trained cosine PPO `.pt` file to bypass training (`best_ppo_free_pilot.pt` and `best_emulator.pt` are loaded automatically if present).

---

## Experimental Results

### Stage 1 & Stage 2 Summary (Single-Corridor Speed Tracking)
* **Stage 1 (`havaei_generalizing_rl.py`):** Lifting actuator delay into LTV-MPC eliminates shift overshoot ($0.7480\text{ m/s} \to 0.0350\text{ m/s}$); an unbiased 5-action Hybrid RL policy across 15 unseen 1,000-step trajectories selects lifted MPC on `76.23%` of steps unshielded (`88.16%` DMPS-shielded), achieving `PaperScore = 0.0910` (`0.0585` shielded).

### Stage 3 Generalized Multi-Station Synthetic Benchmark ($N = 30$ Scenarios, $1\text{--}5$ Stations, $4.75\text{--}27.80\text{ km}$)
Evaluated across 84 inter-station legs (`37 Clear-Dry`, `4 Windy-Dry`, `12 Rain-WetRail`, `9 Cold-Humid`, `11 Snow-Slush`, `11 Heat-Thermal`) under the loss-consistent multi-objective metric:
$$\text{LossScore} = 1.0\,|\text{TimeErr}| + 15.0\,|\text{StopErr}|_{\text{m}} + 2.0\,\text{SmoothLoss} + 0.025\,\text{SpecEnergy}$$

#### Part A: Clean / No-Weather Scenarios Only ($N = 15$)
| Trajectory Pilot + Contract Shield Method | $\|\text{TimeErr}\|$ (s) | $\|\text{StopErr}\|$ (m) | SmoothLoss | SpecEnergy ($\text{Wh/t}\cdot\text{km}$) | LossScore | Shield% | Core Step | Total (Mean / P99 / Max) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1. Max-Speed Greedy Cosine + MPC | 4.58 s | 0.004 m | 0.0632 | 38.87 | 5.7325 | 0.0% | 11.2 $\mu$s | 13.8 / 22.3 / 64.9 $\mu$s |
| 2. Naive Pace (No Overshoot) + MPC | 18.44 s | **0.003 m** | 0.0502 | 34.24 | 19.4498 | 0.0% | 11.0 $\mu$s | 13.3 / 21.5 / 67.3 $\mu$s |
| 3. Static Plan (No Replan) + MPC | 6.01 s | 0.004 m | 0.0559 | 35.06 | 7.0442 | 0.0% | 11.0 $\mu$s | 13.2 / 21.2 / 132.6 $\mu$s |
| 4. Multi-Property PPO (Unshielded) | 0.58 s | 0.004 m | 0.0514 | 36.70 | 1.6580 | 0.0% | 11.1 $\mu$s | 14.3 / 25.3 / 106.7 $\mu$s |
| 5. Contract-Shielded PPO (Switch MPS) | **0.53 s** | 0.004 m | 0.0713 | 36.60 | 1.6449 | 19.1% | 11.6 $\mu$s | 23.5 / 40.5 / 138.0 $\mu$s |
| 6. Contract-Shielded PPO (1-Step DCBF) | 0.56 s | **0.003 m** | 0.0522 | 36.74 | 1.6367 | 6.0% | 11.6 $\mu$s | 23.7 / 42.1 / 77.5 $\mu$s |
| **7. Contract-Shielded PPO (UCB-DMPS Cosine)** | 0.54 s | 0.004 m | 0.0529 | **36.34** | **1.6096** | 14.9% | 11.7 $\mu$s | 27.1 / 53.6 / 222.5 $\mu$s |
| 8. Soft-Curve PPO (No Cosine, Unshielded) | 0.78 s | 0.004 m | 0.0600 | 37.00 | 1.8890 | 0.0% | 12.4 $\mu$s | 16.0 / 28.2 / 93.3 $\mu$s |
| 9. Soft-Curve PPO + Jerk-Horizon UCB-DMPS | 0.54 s | 0.006 m | 0.0619 | 36.71 | 1.6697 | 12.8% | 13.0 $\mu$s | 28.3 / 50.5 / 231.9 $\mu$s |
| 10. Bilbao (2025) CBTC-ATO (RMBTP+PI) | 4.73 s | 0.014 m | 0.1092 | 37.14 | 6.0875 | 0.0% | 19.5 $\mu$s | 19.5 / 26.5 / 468.0 $\mu$s |

#### Part B: Weather-Disturbed (Wind / Rain / Snow / Cold / Heat-Thermal) Scenarios Only ($N = 15$)
| Trajectory Pilot + Contract Shield Method | $\|\text{TimeErr}\|$ (s) | $\|\text{StopErr}\|$ (m) | SmoothLoss | SpecEnergy ($\text{Wh/t}\cdot\text{km}$) | LossScore | Shield% | Core Step | Total (Mean / P99 / Max) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1. Max-Speed Greedy Cosine + MPC | 4.41 s | **0.007 m** | 0.0629 | 56.07 | 6.0327 | 0.0% | 11.4 $\mu$s | 14.1 / 24.1 / 134.3 $\mu$s |
| 2. Naive Pace (No Overshoot) + MPC | 18.47 s | 0.008 m | **0.0502** | 48.91 | 19.9162 | 0.0% | 11.1 $\mu$s | 13.5 / 21.2 / 112.1 $\mu$s |
| 3. Static Plan (No Replan) + MPC | 5.90 s | 0.009 m | 0.0571 | 48.21 | 7.3625 | 0.0% | 11.3 $\mu$s | 14.8 / 23.0 / 98.8 $\mu$s |
| 4. Multi-Property PPO (Unshielded) | 1.07 s | 0.011 m | 0.0509 | 51.27 | 2.6157 | 0.0% | 11.4 $\mu$s | 15.0 / 26.8 / 90.7 $\mu$s |
| 5. Contract-Shielded PPO (Switch MPS) | 0.86 s | 0.009 m | 0.0618 | 51.18 | 2.4010 | 10.6% | 11.8 $\mu$s | 24.2 / 40.7 / 166.8 $\mu$s |
| 6. Contract-Shielded PPO (1-Step DCBF) | 0.99 s | 0.010 m | 0.0514 | 51.24 | 2.5189 | 3.7% | 11.8 $\mu$s | 24.4 / 44.0 / 200.8 $\mu$s |
| **7. Contract-Shielded PPO (UCB-DMPS Cosine)** | **0.78 s** | 0.009 m | 0.0539 | **51.17** | **2.3117** | 12.0% | 11.9 $\mu$s | 27.2 / 52.9 / 223.9 $\mu$s |
| 8. Soft-Curve PPO (No Cosine, Unshielded) | 1.30 s | 0.009 m | 0.0519 | 51.88 | 2.8333 | 0.0% | 12.5 $\mu$s | 16.4 / 28.5 / 120.4 $\mu$s |
| 9. Soft-Curve PPO + Jerk-Horizon UCB-DMPS | 1.05 s | 0.009 m | 0.0555 | 51.85 | 2.5977 | 9.9% | 13.1 $\mu$s | 28.6 / 53.4 / 232.6 $\mu$s |
| 10. Bilbao (2025) CBTC-ATO (RMBTP+PI) | 3.70 s | 0.033 m | 0.0958 | 49.30 | 5.6293 | 0.0% | 19.3 $\mu$s | 19.3 / 21.9 / 279.2 $\mu$s |

#### Part C: United Combined Benchmark Across All 30 Scenarios ($430.9\text{ km}$ Total, $66,920$ Active Driving Steps)
| Trajectory Pilot + Contract Shield Method | $\|\text{TimeErr}\|$ (s) | $\|\text{StopErr}\|$ (m) | SmoothLoss | SpecEnergy ($\text{Wh/t}\cdot\text{km}$) | LossScore | Shield% | Core Step | Total (Mean / P99 / Max) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1. Max-Speed Greedy Cosine + MPC | 4.49 s | **0.005 m** | 0.0630 | 47.47 | 5.8826 | 0.0% | 11.3 $\mu$s | 13.9 / 23.2 / 134.3 $\mu$s |
| 2. Naive Pace (No Overshoot) + MPC | 18.46 s | 0.006 m | **0.0502** | 41.57 | 19.6831 | 0.0% | 11.1 $\mu$s | 13.4 / 21.3 / 112.1 $\mu$s |
| 3. Static Plan (No Replan) + MPC | 5.95 s | 0.007 m | 0.0565 | 41.63 | 7.2034 | 0.0% | 11.1 $\mu$s | 14.0 / 22.1 / 132.6 $\mu$s |
| 4. Multi-Property PPO (Unshielded) | 0.82 s | 0.008 m | 0.0512 | 43.98 | 2.1369 | 0.0% | 11.3 $\mu$s | 14.7 / 26.0 / 106.7 $\mu$s |
| 5. Contract-Shielded PPO (Switch MPS) | 0.70 s | 0.006 m | 0.0666 | 43.89 | 2.0230 | 14.9% | 11.7 $\mu$s | 23.8 / 40.6 / 166.8 $\mu$s |
| 6. Contract-Shielded PPO (1-Step DCBF) | 0.77 s | 0.007 m | 0.0518 | 43.99 | 2.0778 | 4.9% | 11.7 $\mu$s | 24.1 / 43.0 / 200.8 $\mu$s |
| **7. Contract-Shielded PPO (UCB-DMPS Cosine)** | **0.66 s** | 0.007 m | 0.0534 | **43.76** | **1.9606** | 13.5% | 11.8 $\mu$s | 27.1 / 53.3 / 223.9 $\mu$s |
| 8. Soft-Curve PPO (No Cosine, Unshielded) | 1.04 s | 0.006 m | 0.0559 | 44.44 | 2.3611 | 0.0% | 12.5 $\mu$s | 16.2 / 28.4 / 120.4 $\mu$s |
| **9. Soft-Curve PPO + Jerk-Horizon UCB-DMPS** | 0.80 s | 0.008 m | 0.0587 | **44.28** | 2.1337 | 11.4% | 13.1 $\mu$s | 28.5 / 51.9 / 232.6 $\mu$s |
| 10. Bilbao (2025) CBTC-ATO (RMBTP+PI) | 4.22 s | 0.024 m | 0.1025 | 43.22 | 5.8584 | 0.0% | 19.4 $\mu$s | 19.4 / 24.2 / 468.0 $\mu$s |

* **Physical Tube Containment:** **99.96%** across $66,920$ active driving micro-steps (Emulator RMSE $= 0.0241\text{ m/s}$, P99 $= 0.0719\text{ m/s}$); UCB Contract Mix: `C1=0 | C2=43 | C3=334 | C4=1631`.
* **Real-World Digitraffic + Open-Meteo Validation (10 Mixed Scenarios):** Raw API (5 Scen., `66.0 km`, `99.99%` containment): PPO Unshielded `LossScore = 3.2745` ($41.75\text{ Wh/tk}$) $\to$ UCB-DMPS Cosine `2.1671` ($0.92\text{ s}$ error, $41.52\text{ Wh/tk}$, $8.7\%$ shield rate); Extreme Noise (5 Scen., `44.6 km`, `99.72%` containment): Unshielded `6.8367` $\to$ UCB-DMPS Cosine `5.0866` (`8.2%` shield); Combined Real-World (10 Scen.): UCB-DMPS Cosine `3.6269` vs. `5.0556` Unshielded (`8.4%` shield rate vs. `16.9%` Switch MPS) (note: API ingestion currently normalizes segment lengths to `[3.8, 6.4] km`, tare masses to `[70.5, 76.0] t`, and uses synthetic speed zones and periodic geometry).
* **Real-Time Hardware Compliance:** Worst-case single-step latency across all $66,920$ synthetic steps for UCB-DMPS Cosine is **$223.9\text{ }\mu\text{s}$** ($< 500\text{ }\mu\text{s}$ hard real-time ceiling).

---

## Generated Diagnostic Figures

Running `dynamic_railroad_mpc.py` automatically exports two high-resolution diagnostic figures inside `deadline_station_trackGen/` (and copies `rl_cosine_mpc.png` to the project root):
1. **`wind_trajectory_shield_dual_axis.png`:** Dual-horizontal-axis 4-station ($21.35\text{ km}$) trajectory profile showing cumulative track position $x\text{ (m)}$ on the bottom axis and clock time $t\text{ (s)}$ on the top axis, the Goodall 99% Scissor Band (yellow) and Safe Envelope (green), active contract interventions (red dots), disjoint wind fronts, MPC throttle/brake effort $u_{\text{act}}$, and causal telescoping 3-block 2nd-diff MAP passenger mass convergence $\hat{M}(x) \to M(x)$.
2. **`rl_cosine_mpc.png`:** 6-panel benchmark overview comparing (1) multi-station speed profiles $v(x)$, (2) Order-2 acceleration $a(t)$ and Order-3 jerk $j(t)$, (3) online causal 3-block 2nd-diff MAP passenger mass identification and weather observer $\hat{d}(t)$, (4) mean deadline error across Clean, Weather-Disturbed, and Combined runs, (5) composite `LossScore`, and (6) normalized multi-property breakdown (`SmoothLoss`, `StopErr`, and `SpecEnergy`).

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
