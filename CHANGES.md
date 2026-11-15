# Change Log: Codebase Verification, High-Temperature Weather Physics, Sun et al. (2024), Wang et al. (2026), Zhao (2024), and Learned Non-Cosine $C^2$ Soft-Curve Trajectory Generation

See the full detailed change log in [deadline_station_trackGen/CHANGES.md](file:///home/giorge/Desktop/ProjectTh/deadline_station_trackGen/CHANGES.md).

## Quick Summary of Changes

1. **Weather-Disturbed Operational Regimes & High-Temperature (`Heat-Thermal`) Force Modeling** ([railroad_corridor.py](file:///home/giorge/Desktop/ProjectTh/deadline_station_trackGen/railroad_corridor.py#L403-L466), [dynamic_railroad_mpc.py](file:///home/giorge/Desktop/ProjectTh/deadline_station_trackGen/dynamic_railroad_mpc.py#L382-L468), [Report2.tex](file:///home/giorge/Desktop/ProjectTh/deadline_station_trackGen/Report2.tex#L65-L96)):
   - Added the 5th weather-disturbed regime **`Heat-Thermal`** ($T_{\text{amb}} = 38^\circ\text{C}$, $T_{\text{rail}} \approx 56^\circ\text{C}$, $\mu_{\text{adh}} = 0.94$, $\rho_{\text{air}} = 0.942$, $\kappa_{\text{cwr}} = 1.0585$, $\kappa_{\text{Cu}} = 1.0707$) modeling isobaric air-density reduction ($-5.8\%$ aerodynamic drag), continuous welded rail (CWR) compressive thermal stress increasing lateral track compliance and curve/creepage resistance ($+5.85\%$), softened high-temperature third-body oxide adhesion ($\mu_{\text{adh}} = 0.94$), and stator copper winding thermal resistance ($+7.07\%$ $I^2R$ loss).
   - Added Table 1 and Subsection 1.3 in [Report2.tex](file:///home/giorge/Desktop/ProjectTh/deadline_station_trackGen/Report2.tex#L65-L96) with peer-reviewed railway citations for every weather parameter and force modification (`Rochard & Schmid 2000`, `Polach 2005`, `Zhang et al. 2002`, `Baker 2010`, `Kish & Samavedam 2013`, `Dobney et al. 2009`, `Xiao et al. 2023`).

2. **Codebase Verification & Audit Fixes** ([dynamic_railroad_mpc.py](file:///home/giorge/Desktop/ProjectTh/deadline_station_trackGen/dynamic_railroad_mpc.py), [cosine_trajectory_shaper.py](file:///home/giorge/Desktop/ProjectTh/deadline_station_trackGen/cosine_trajectory_shaper.py), [goodall_trajectory_shield.py](file:///home/giorge/Desktop/ProjectTh/deadline_station_trackGen/goodall_trajectory_shield.py)):
   - Fixed `_sync_physics_and_mpc(use_estimated_mass=True, reset_shaper=False)` at $k=8$ so locking $\hat{M}_{\text{locked}}$ preserves the active departure climb S-curve phase.
   - Trained and saved `best_ppo_free_pilot.pt` with non-zero actor weights and loaded it for Methods 8 & 9.
   - Implemented full Level-2 child state/tube variance projection and leaf Tchebycheff scoring in `GoodallTrajectoryShield`, candidate-dependent $\rho_E$, instance-scoped UCB statistics, active-step vs. dwell-step separation in diagnostics, and consistent Scenario #1 table/plot reporting.
