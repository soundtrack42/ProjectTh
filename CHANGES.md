# Change Log: Unbiased Learning Audit, Multi-Lag MAP Mass ID, 18-D Slope-Aware PPO, Full-Contract Persistent UCB Tree & Report Alignment

See the full detailed change log in [deadline_station_trackGen/CHANGES.md](file:///home/giorge/Desktop/ProjectTh/deadline_station_trackGen/CHANGES.md).

## Summary of Latest Alignment & Anti-Hardcoding Updates

1. **Fair & Unbiased Stage 1–2 Hybrid RL and Active 8-Feature Neural Emulator** ([havaei_generalizing_rl.py](file:///home/giorge/Desktop/ProjectTh/havaei_generalizing_rl.py), [safety_shield.py](file:///home/giorge/Desktop/ProjectTh/deadline_station_trackGen/safety_shield.py)):
   - Removed the hardcoded PID penalty bias (`+0.28 + 1.8*err^2`) and forced MPC rollout lookahead from `_compute_surrogate_horizon_q_costs`.
   - Activated all 8 neural emulator inputs (`_build_features`, `_predict_step_with_uncertainty`, `predict_emulator_step_with_uncertainty`) without zero-padding or column slicing (`[:, :5]`).

2. **Fair Stage 3 PPO (18-D Slope- & Adhesion-Aware) and Shaper Integrity** ([dynamic_railroad_mpc.py](file:///home/giorge/Desktop/ProjectTh/deadline_station_trackGen/dynamic_railroad_mpc.py), [cosine_trajectory_shaper.py](file:///home/giorge/Desktop/ProjectTh/deadline_station_trackGen/cosine_trajectory_shaper.py)):
   - Expanded PPO observations from `15` to `18` features adding forward grade lookahead (`slope_ahead_150`, `slope_ahead_350`) and adhesion deficit.
   - Removed `pace_err`, `comfort_param_pen`, `smooth_reg_loss`, `0.68 * v_feas` override, and asymmetric `(0.88, 0.74)` mix in `decode_rl_action`, training purely from outcome metrics with Ng potential-based progress shaping across 1..5 stations.
   - Removed the unfair `if shield_mode is not None` dwell reduction (`eff_dwell_s = nominal_dwell_s`).

3. **Full-Contract Persistent UCB Tree, Forward "On-Track" Eco-Coasting & Causal 3-Block 2nd-Diff Mass ID** ([goodall_trajectory_shield.py](file:///home/giorge/Desktop/ProjectTh/deadline_station_trackGen/goodall_trajectory_shield.py), [dynamic_railroad_mpc.py](file:///home/giorge/Desktop/ProjectTh/deadline_station_trackGen/dynamic_railroad_mpc.py), [railroad_corridor.py](file:///home/giorge/Desktop/ProjectTh/deadline_station_trackGen/railroad_corridor.py)):
   - Expanded UCB searching tree across all 4 recovery contracts (`C1..C4`), persisted UCB stats across journey legs, fixed `C1_SPEED_SCISSOR_CLAMP` to clamp downward only (`v_ppo_safe`), prevented `C3` from interrupting station-landing braking, and immediately cleared the 2-step ramp latch whenever `min_margin >= 0.0`.
   - Replaced coarse slack heuristics with a shared forward zone-by-zone remaining-time estimator (`_build_remaining_time_estimator`, `estimate_remaining_time_at_speed`, `compute_on_track_slack_s`) summing `min(v_cap_z, v_meas)` across remaining speed zones plus the stop curve `t_land`. Wired `_compute_eco_speed_limit` into hierarchical pressures, scalar backups, and latching (`ot_slack > 3.0 s`, `u_act > 0.02`, `v_cmd > v_obs + 0.08`) together with downhill/curve coasting and causal mechanical energy-excess Discounted-UCB updates, cutting `SpecEnergy` below unshielded PPO on both pilots (`43.76` vs. `43.98 Wh/t-km` Cosine at `13.5%` Shield / `#1 LossScore = 1.9606`; `44.28` vs. `44.44 Wh/t-km` Soft-Curve at `11.4%` Shield / `2.1337`).
   - Replaced lag-(2,4,8) 1st-diff MAP with causal telescoping 3-block second-difference IRLS Bayesian MAP mass identification (`_estimate_causal_mass_kg` across `k < 64`, preserving `0.72/0.28` actuator lag and `0.14` velocity momentum, empirically self-calibrating between smooth `(1,2,4,8)` and noisy `(4,8,12)` blocks via `W=1` MAD), cutting departure MAE from `183.2 kg` to **`1.4 kg`** clean (`max = 10.7 kg`) and **`774.0 kg`** under extreme sensor/turbulence noise, and fixed `mu_adhesion` scenario sampling order in `railroad_corridor.py`.

4. **Documentation & Architecture Alignment** ([Report2.tex](file:///home/giorge/Desktop/ProjectTh/deadline_station_trackGen/Report2.tex), [README.md](file:///home/giorge/Desktop/ProjectTh/README.md), [generate_architecture_diagram.py](file:///home/giorge/Desktop/ProjectTh/deadline_station_trackGen/generate_architecture_diagram.py)):
   - Aligned all architecture diagrams, text, equations, and synthetic (`N=30`) + live Digitraffic/Open-Meteo (`N=10`) tables with observed code and benchmarks.
