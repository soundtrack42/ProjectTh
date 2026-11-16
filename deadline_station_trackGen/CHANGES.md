# Change Log: Codebase Verification, High-Temperature Weather Physics, Sun et al. (2024), Wang et al. (2026), Zhao (2024), and Learned Non-Cosine $C^2$ Soft-Curve Trajectory Generation

This document tracks all mathematical, physical, algorithmic, and verification updates across the Stage 3 Multi-Station Automatic Train Operation (ATO) / LTV-MPC / Hierarchical Contract Shielding codebase and [Report2.tex](file:///home/giorge/Desktop/ProjectTh/deadline_station_trackGen/Report2.tex).

---

## 1. Summary of Integrated Literature, Physical Weather Modeling & Core Innovations

### 1.1 Weather-Disturbed Operational Regimes & High-Temperature (`Heat-Thermal`) Force Modeling
* **Source Papers**:
  - B. P. Rochard and F. Schmid, *"A review of methods to measure and calculate train resistances"*, Proc. IMechE Part F: J. Rail and Rapid Transit, 214(4):185–199, 2000 (`\cite{ref23}`).
  - O. Polach, *"Creep forces in simulations of traction vehicles running on adhesion limit"*, Wear, 258(7–8):992–1000, 2005 (`\cite{ref24}`).
  - W. H. Zhang, J. Z. Chen, X. J. Wu, and X. S. Jin, *"Wheel/rail adhesion and analysis by using full scale roller rig"*, Wear, 253(1–2):82–88, 2002 (`\cite{ref25}`).
  - C. J. Baker, *"The flow around high speed trains"*, J. Wind Eng. Ind. Aerodyn., 98(6–7):277–298, 2010 (`\cite{ref26}`).
  - A. Kish and G. Samavedam, *"Track Buckling Prevention: Theory, Safety Concepts, and Applications"*, US DOT Federal Railroad Administration Report DOT/FRA/ORD-13/16, 2013 (`\cite{ref27}`).
  - K. Dobney, C. J. Baker, A. D. Quinn, and L. Chapman, *"Quantifying the effects of high summer temperatures due to climate change on buckling and rail related delays in south-east United Kingdom"*, Meteorological Applications, 16(2):245–258, 2009 (`\cite{ref28}`).
  - Z. Xiao et al., *"Eco-driving for metro trains: A computationally efficient approach using convex programming"*, IEEE Trans. Veh. Technol., 72(8):10063–10076, 2023 (`\cite{ref16}`).
* **Implemented Weather Regimes & Force Modifications** ([railroad_corridor.py](file:///home/giorge/Desktop/ProjectTh/deadline_station_trackGen/railroad_corridor.py#L403-L466), [dynamic_railroad_mpc.py](file:///home/giorge/Desktop/ProjectTh/deadline_station_trackGen/dynamic_railroad_mpc.py#L382-L468)):
  1. **`Windy-Dry` ($p = 0.22$, $T_{\text{amb}} = 16^\circ\text{C}$, $\mu_{\text{adh}} = 1.00$, $\rho_{\text{air}} = 1.014$, $\kappa_{\text{wind}} = 1.00\times$)**:
     Dry steel-on-steel adhesion; slightly denser air ($+1.4\%$ Davis drag); open-corridor headwind/tailwind gusts up to $0.72\text{ m/s}^2$ (Rochard & Schmid 2000; Baker 2010).
  2. **`Rain-WetRail` ($p = 0.24$, $T_{\text{amb}} = 8^\circ\text{C}$, $\mu_{\text{adh}} = 0.86$, $\rho_{\text{air}} = 1.043$, $\kappa_{\text{wind}} = 1.15\times$)**:
     Water boundary lubrication at wheel-rail micro-asperities reduces maximum traction/braking adhesion by $14\%$ (Polach 2005; Zhang et al. 2002); raindrop momentum impingement and wet skin friction amplify aerodynamic gust drag by $1.15\times$ (Baker 2010).
  3. **`Cold-Humid` ($p = 0.18$, $T_{\text{amb}} = -6^\circ\text{C}$, $\mu_{\text{adh}} = 0.92$, $\rho_{\text{air}} = 1.097$, $\kappa_{\text{wind}} = 1.12\times$)**:
     Sub-zero air increases isobaric density $\rho_{\text{air}}(T_{\text{amb}}) = 293.15 / (273.15 + T_{\text{amb}})$ by $+9.7\%$ (Rochard & Schmid 2000); railhead condensation/frost film lowers adhesion by $8\%$ (Polach 2005).
  4. **`Snow-Slush` ($p = 0.18$, $T_{\text{amb}} = -3^\circ\text{C}$, $\mu_{\text{adh}} = 0.80$, $\rho_{\text{air}} = 1.085$, $\kappa_{\text{wind}} = 1.20\times$)**:
     Compacted snow/slush shear layer drops wheel-rail adhesion by $20\%$ (Polach 2005; Zhang et al. 2002); dense cold air ($+8.5\%$) plus airborne snow ingestion and bogie slush ploughing amplify external drag bursts by $1.20\times$ (Rochard & Schmid 2000).
  5. **`Heat-Thermal` ($p = 0.18$, $T_{\text{amb}} = 38^\circ\text{C}$, $T_{\text{rail}} \approx 56^\circ\text{C}$, $\mu_{\text{adh}} = 0.94$, $\rho_{\text{air}} = 0.942$, $\kappa_{\text{wind}} = 1.05\times$, $\kappa_{\text{cwr}} = 1.0585$, $\kappa_{\text{Cu}} = 1.0707$)**:
     - **Isobaric Air Expansion**: Reduces aerodynamic Davis drag by $-5.8\%$ ($\rho_{\text{air}} = 0.942$, Rochard & Schmid 2000).
     - **CWR Rail Thermal Expansion & Lateral Compliance Resistance**: Above the rail neutral temperature $T_0 = 25^\circ\text{C}$, compressive axial thermal stress $P_{\text{th}} = E A_{\text{rail}}\alpha_{\text{steel}}(T_{\text{rail}} - T_0)$ reduces lateral track buckling stiffness and increases wheel-rail gauge-corner/flange creepage curve resistance by $\kappa_{\text{cwr}}(T_{\text{amb}}) = 1 + 0.0045\max(0, T_{\text{amb}} - 25^\circ\text{C}) = 1.0585$ ($+5.85\%$, Kish & Samavedam 2013; Dobney et al. 2009).
     - **Thermal Oxide Contact Layer**: Elevated wheel-rail flash/bulk contact temperature softens third-body iron-oxide ($\text{Fe}_2\text{O}_3/\text{Fe}_3\text{O}_4$) layers, reducing dry adhesion capacity to $\mu_{\text{adh}} = 0.94$ (Zhang et al. 2002; Polach 2005).
     - **Stator/Inverter Copper Winding Resistance**: Copper temperature coefficient $\alpha_{\text{Cu}} = 0.00393\text{ K}^{-1}$ above $20^\circ\text{C}$ increases $I^2R(T_{\text{amb}})$ Joule losses by $\kappa_{\text{Cu}}(T_{\text{amb}}) = 1 + 0.00393\max(0, T_{\text{amb}} - 20^\circ\text{C}) = 1.0707$ ($+7.07\%$) in $\eta_{\text{trac}}$ (Xiao et al. 2023).

---

### 1.2 Event-Triggered Forward Reachable Tube & Flexible Multi-Scale Disturbance Arrival Detection
* **Source Papers**: P. Sun et al., Proc. SPIE 130640F, 2024 (`130640F (2).pdf`), combined with Y. Wang, J. Wang, and J. Chen, *Biomimetics* 11(1), 60, 2026 (`biomimetics-11-00060-v2.pdf`).
* **Implemented Solution** ([dynamic_railroad_mpc.py](file:///home/giorge/Desktop/ProjectTh/deadline_station_trackGen/dynamic_railroad_mpc.py#L430-L590)):
  1. **Online Fuzzy-Adaptive Disturbance Observer (Wang et al. 2026, Eqs. 20–21, Table 2)**:
     $$\alpha_h(\delta_d, \dot{\delta}_d) = \text{clip}_{[0.42, 0.82]}\!\left(0.45 + \frac{0.85\,|\delta_d|}{0.14 + |\delta_d|} + \Delta\alpha_{\text{sign}}(\delta_d\dot{\delta}_d)\right)$$
  2. **Four-Criterion Inductive Mid-Horizon Preemption ($k_{t+1} = \min\{\bar{k}_{t+1}, k_t + M\}$, Sun et al. 2024 Eq. 22)**:
     Preempts the 8-step macro-window at any interior micro-step $j \in \{1, \dots, 7\}$ whenever:
     - $|\hat{d}(k_t+j) - \hat{d}(k_t)| > 0.145\text{ m/s}^2$ (Instantaneous Gust Shift),
     - $|S_{\text{cusum}}(k_t+j)| > 0.22\text{ m/s}^2$ (Directional CUSUM Drift),
     - $\mu_{\min}(k_t+j) = \min(\mu_y, \mu_d) < 0.15$ (Online Fuzzy Satisfaction Index, Wang et al. 2026 Eqs. 22–25), or
     - $v_{\text{tube}}^{\text{rem}}(k_t+j) = v_{k_t+j} + (M - 1 - j)\Delta t\max(0, a_{\text{real}} + 0.5\hat{d}) + 2.576\,\sigma_{\text{tube}} > v_{\text{lim}}(x_{k_t+j}) + 0.02$ (Inductive Remaining-Horizon Reachable Tube Projection, Sun et al. 2024 Eq. 18).

---

### 1.3 Step-Adaptive Safe Contract Searching Tree & Tchebycheff Multi-Objective Arbitration
* **Source Papers**: Z. Zhao, arXiv:2311.14457v2, 2024 (`2311.14457v2.pdf`, Section III.B, Algorithm 1, Eqs. 9–10) and Wang et al., *Biomimetics* 2026 (`biomimetics-11-00060-v2.pdf`, Eq. 11).
* **Implemented Solution** ([goodall_trajectory_shield.py](file:///home/giorge/Desktop/ProjectTh/deadline_station_trackGen/goodall_trajectory_shield.py#L400-L680)):
  1. **Tchebycheff Multi-Objective Score (`_tchebycheff_multi_obj_score`, Wang et al. 2026 Eq. 11)** relative to utopia point $Z^* = (0.0, 0.42, 0.22, 0.48)$, including candidate-dependent slew energy increment `cand_slew_e` in $\rho_E$.
  2. **Step-Adaptive Safe Contract Searching Tree (`filter_trajectory_action`, Zhao 2024 Alg. 1)**:
     - Expands Level-1 nodes across $\mathcal{A}'_{\text{sa}} = \{\mathcal{C}_{\text{primary}}, \mathcal{C}_{\text{alt}}\}$ and $\lambda \in \{0.55, 0.85\}$ with Level-1 shield pruning ($h_1 \ge 0, h_2 \ge 0$).
     - Expands Level-2 child nodes $(x_{\text{leaf}}, v_{\text{leaf}})$ over $M_{\text{rem}} = \max(2, M - (k \bmod M))$, propagating contractive tube variance $\sigma_{\text{leaf}}^2 = \gamma_\sigma^{M_{\text{rem}}}\sigma_{\text{tube}}^2 + \frac{1-\gamma_\sigma^{M_{\text{rem}}}}{1-\gamma_\sigma}\hat{\sigma}_{\text{step}}^2$, pruning branches that breach $v_{\text{ceil}}(x_{\text{leaf}}) + 0.02$, evaluating leaf Tchebycheff return $r^{\text{si}}_{\text{leaf}}$, and backpropagating Bellman return + Discounted-UCB bonus ($r^{\text{ex}} = r^{\text{si}}_{\text{L1}} + 0.90(0.35\,r^{\text{si}}_{\text{leaf}}) + 0.40\,\text{UCB}(\mathcal{C})$).

---

### 1.4 Verification Audit Fixes Across All Modules
1. **Departure Climb Phase Continuity at $k=8$ ([dynamic_railroad_mpc.py](file:///home/giorge/Desktop/ProjectTh/deadline_station_trackGen/dynamic_railroad_mpc.py#L70-L90))**:
   Added `reset_shaper: bool = False` to `_sync_physics_and_mpc` at `k_elapsed == 7` so locking $\hat{M}_{\text{locked}}$ updates `self.shaper.corridor` without re-instantiating `CosineTrajectoryShaper` and wiping `ramp_u_phase` mid-climb.
2. **Trained Non-Zero Checkpoint for Non-Cosine PPO Pilot (`best_ppo_free_pilot.pt`, [dynamic_railroad_mpc.py](file:///home/giorge/Desktop/ProjectTh/deadline_station_trackGen/dynamic_railroad_mpc.py#L888-L995))**:
   Moved validation evaluation in `train_pure_ppo_cosine_pilot` to execute *after* `optimizer.step()`, trained `best_ppo_free_pilot.pt` (60 epochs, non-zero actor weights `max |w| = 0.0490`), and loaded `best_ppo_free_pilot.pt` into `rl_policy_free` in `main()`.
3. **Instance-Scoped UCB Statistics & Active-Step Diagnostics ([goodall_trajectory_shield.py](file:///home/giorge/Desktop/ProjectTh/deadline_station_trackGen/goodall_trajectory_shield.py#L96-L115))**:
   Scoped `contract_counts` and `contract_scores` per shield instance, excluded passive station dwell steps (`is_dwell=True`) from active driving step counts and RMSE/P99 diagnostics, and computed exact macro-step latency amortization when mid-horizon preemption occurs.
4. **Scenario #1 Sample Table & Plot Title Consistency ([dynamic_railroad_mpc.py](file:///home/giorge/Desktop/ProjectTh/deadline_station_trackGen/dynamic_railroad_mpc.py#L1480-L1545))**:
   Reused `scenarios_list[ep_i]` in the sample printout table so Scenario #1's 4 stops and passenger mass evolution match `r_d['n_legs']` and the dual-axis plot title.

---

## 2. Verified 30-Scenario Benchmark Results (`dynamic_railroad_mpc.py`)

| Method | Metric | Clean (15 Scen.) | Weather-Disturbed (15 Scen.) | Combined (30 Scen.) |
| :--- | :--- | :---: | :---: | :---: |
| **4. Multi-Property PPO (Unshielded)** | `\|TimeErr\|` | `0.58 s` | `1.07 s` | `0.82 s` |
| | `\|StopErr\|` | `0.004 m` | `0.011 m` | `0.008 m` |
| | `SmoothLoss` | `0.0514` | `0.0509` | `0.0512` |
| | `SpecEnergy` | `36.70 Wh/t-km` | `51.27 Wh/t-km` | `43.98 Wh/t-km` |
| | `LossScore` | `1.6580` | `2.6157` | `2.1369` |
| **7. Contract-Shielded PPO (UCB-DMPS Cosine)** | `\|TimeErr\|` | **`0.54 s`** | **`0.78 s`** | **`0.66 s`** (**#1 Combined**) |
| | `\|StopErr\|` | `0.004 m` | `0.009 m` | `0.007 m` |
| | `SmoothLoss` | `0.0529` | `0.0539` | `0.0534` |
| | `SpecEnergy` | **`36.34 Wh/t-km`** | `51.17 Wh/t-km` | **`43.76 Wh/t-km`** (**Beats M4**) |
| | `LossScore` | **`1.6096`** | **`2.3117`** | **`1.9606`** (**#1 Overall**) |
| **8. Soft-Curve PPO (No Cosine, Unshielded)** | `\|TimeErr\|` | `0.78 s` | `1.30 s` | `1.04 s` |
| | `\|StopErr\|` | `0.004 m` | `0.008 m` | **`0.006 m`** |
| | `SmoothLoss` | `0.0600` | `0.0519` | `0.0559` |
| | `SpecEnergy` | `37.00 Wh/t-km` | `51.88 Wh/t-km` | `44.44 Wh/t-km` |
| | `LossScore` | `1.8890` | `2.8333` | `2.3611` |
| **9. Soft-Curve PPO + Jerk-Horizon UCB-DMPS** | `\|TimeErr\|` | `0.54 s` | `1.05 s` | `0.80 s` (**-23.1% vs M8**) |
| | `\|StopErr\|` | `0.007 m` | `0.009 m` | `0.008 m` |
| | `SmoothLoss` | `0.0616` | `0.0558` | `0.0587` |
| | `SpecEnergy` | `36.71 Wh/t-km` | `51.85 Wh/t-km` | `44.28 Wh/t-km` (**Beats M8**) |
| | `LossScore` | `1.6697` | `2.5977` | **`2.1337`** (**-9.6% vs M8**) |

Also verified on 10 live Finnish Digitraffic + Open-Meteo scenarios (`110.6 km`, `16,939` steps): combined M4 `3.42 s | 52.62 Wh/t-km | Loss = 5.0556` vs. M7 `2.02 s | 52.72 Wh/t-km | Loss = 3.6269 | Sh = 8.4%` (`99.99%` clean API / `99.72%` extreme noise tube containment).

---

## 3. Forward "On-Track" Eco-Coasting Shield & Causal 3-Block 2nd-Diff Mass ID (`goodall_trajectory_shield.py`, `dynamic_railroad_mpc.py`)

* **Forward Zone-by-Zone "On-Track" Eco-Coasting (`_build_remaining_time_estimator`, `estimate_remaining_time_at_speed`, `compute_on_track_slack_s`, `_compute_eco_speed_limit`):** Computes exact remaining corridor travel time `t_rem(min(v_cap_z, v_meas))` summed across remaining speed zones plus stop trajectory `t_land`, activating `C4_ECO_REGEN_COAST` against positive throttle commands whenever `ot_slack > 3.0 s`, `v_obs > max(12.4, v_pace_feasible + 0.25)`, `u_act > 0.02`, and `v_cmd > v_obs + 0.08` (alongside downhill/curve coasting, `C1` downward clamp, and immediate-clear latch release). Across 30 synthetic scenarios (`99.96%` containment), UCB arbitration actively selects `C4` (`C1=0, C2=43, C3=334, C4=1631`), cutting energy below Unshielded PPO on both Cosine (`43.76` vs. `43.98`, `Sh=13.5%`, `LossScore=1.9606`) and Soft-Curve (`44.28` vs. `44.44`, `Sh=11.4%`, `LossScore=2.1337`).
* **Causal Telescoping 3-Block 2nd-Diff IRLS MAP Mass ID (`_estimate_causal_mass_kg`):** Preserves `0.72/0.28` actuator lag inside per-step force projections `f_k` and `(1, -0.14)` velocity momentum over triple blocks `[s, s+W), [s+W, s+2W), [s+2W, s+3W)` (`k < 64`), taking the second difference to cancel smooth linear wind ramps while telescoping intermediate tachometer noise and self-calibrating between smooth `(1,2,4,8)` and noisy `(4,8,12)` departures via `W=1` MAD (`1.4 kg` MAE clean, `max = 10.7 kg`; `774.0 kg` MAE extreme noise).
* **Architecture Diagram Sync ([generate_architecture_diagram.py](file:///home/giorge/Desktop/ProjectTh/deadline_station_trackGen/generate_architecture_diagram.py)):** Regenerated `contracts_hierarchical_shielding_arch.png` reflecting the forward on-track slack estimator, 3-block 2nd-diff MAP mass ID, `86.5%` pass-through / `13.5%` shield-active split, and all-contract UCB feedback loop.
