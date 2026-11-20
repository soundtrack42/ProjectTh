# Safe, Punctual, and Weather-Aware Learned Train Control

Sub-millisecond learned train speed control with contract shielding and station-based mass identification.

Railway automation is usually split in two:
- **Automatic Train Protection (ATP)** enforces hard braking envelopes.
- **Automatic Train Operation (ATO)** pursues punctuality, comfort and energy efficiency, without formal safety guarantees.

This repository unifies both in one onboard pipeline. It addresses the open challenges named by railway ATO surveys: an unknown passenger load, weather disturbances, conflicting objectives, safety of learned control, and real-time computation.

## How it works

At every station stop and on the line between stations, the pipeline runs these steps:

1. **Mass identification.** At each departure, the passenger mass is identified in closed form, together with the station wind, from wayside weather telemetry (one station anemometer, optionally a second one at the platform end).
2. **PPO pilot.** A learned 18-feature pilot chooses, every macro step (8 × 0.5 s), a target speed, a departure acceleration and a landing deceleration.
3. **Eco governor.** When the train is ahead of schedule, it caps the speed command so the train does not accelerate only to brake later.
4. **Contract shield (UCB-DMPS).** It checks the pilot's proposal for speed, comfort (acceleration and jerk) and deadline margins against a learned 99% uncertainty band. If a margin is violated, it searches a tree of backup contracts. It intervenes on at most 4.1% of macro steps.
5. **Reference shaper.** A $C^2$ half-cosine or Hermite shaper turns the command into a smooth, jerk-bounded speed reference.
6. **LTV-MPC tracker.** A closed-form state-augmented MPC tracks the reference. It models the drivetrain lag and feeds forward the estimated wind.

Every proposed method runs at ≤ 278 µs worst-case step latency.

### Weather model

- **Wind** acts on the air drag and is estimated online. Its force follows from the plant's own drag model, with gusts capped at 30 m/s (Beaufort ≤ 10).
- **Temperature, adhesion and precipitation** change slowly. They are measured at each station and then updated along the route from a weather service (one request per minute, outside the control loop; a failed request keeps the last value). Design study S9 (`weather_update_study.py`) shows the update is neutral on real legs and helps when a weather front passes, if the service data are minute-resolved.

## Results

The benchmark covers 15 methods: our 7 configurations plus 8 literature baselines. It runs on three test suites:

| Test suite | Journeys | Source |
|---|---|---|
| Real-world, measured weather | 15 | Finnish Digitraffic timetables + Open-Meteo weather |
| Real-world, extreme weather | 15 | the same kind of journeys with injected storms and sensor noise |
| Synthetic | 30 | fully generated (15 normal + 15 extreme) |

The score is
**LossScore = |TimeErr| + 15·|StopErr| + 2·SmoothLoss + 0.025·SpecEnergy**. Lower is better.

| Method | Real (measured) | Real (extreme) | Synthetic | Latency mean / max |
|---|---|---|---|---|
| 1. Greedy cosine + MPC | 5.928 | 6.029 | 7.060 | 33 / 89 µs |
| **2. PPO pilot, unshielded** | **1.228** | **1.503** | 1.595 | 33 / 89 µs |
| 3. PPO + switch shield | 1.298 | 1.538 | **1.527** | 52 / 165 µs |
| 4. PPO + 1-step DCBF | 1.305 | 1.558 | 1.575 | 52 / 129 µs |
| **5. PPO + UCB-DMPS (main method)** | 1.371 | 1.710 | 1.531 | 57 / 231 µs |
| 6. Non-cosine PPO, unshielded | 1.346 | 1.755 | 1.654 | 37 / 98 µs |
| 7. Non-cosine PPO + UCB-DMPS | 1.432 | 1.715 | 1.750 | 61 / 226 µs |
| 8. Bilbao (2025), CBTC-ATO | 6.005 | 4.685 | 5.442 | 8 / 322 µs |
| 9. Xiao (2023), convex eco-driving | 10.629 | 8.097 | 8.030 | 0.38 / 1.61 ms |
| 10. Zhong (2022), pseudospectral MPC | 2.660 | 3.810 | 2.965 | 0.35 / 1.49 ms |
| 11. CDQ (2025) | 6.890 | 6.800 | 5.958 | 35 / 132 µs |
| 12. Shield SARSA (2023) | 11.807 | 11.664 | 11.688 | 34 / 133 µs |
| 13. EITOP (2025) | 1.384 | 1.541 | 1.670 | 34 / 112 µs |
| 14. SRL-IETTO (2023) | 1.404 | 1.891 | 1.785 | 40 / 138 µs |
| 15. SSA-DRL (2025) | 1.539 | 1.713 | 1.772 | 83 / 276 µs |

- **Literature baselines.** None of them has a runtime safety layer that covers speed, comfort and deadline together: six have none, and two shield overspeed only.
- **Mass identification.** Under extreme sensor noise, the second anemometer at 10 m cuts the mass error from about 680–950 kg to about 110–130 kg.

The full tables, design studies and appendices are in the paper (`deadline_station_trackGen/docs/Report.tex`).

### Pilot learner study (MASA-Safe-RL)

PPO was kept as the pilot learner after comparing it with the five other learners of [MASA-Safe-RL](https://github.com/sacktock/MASA-Safe-RL) that fit our interface: A2C, TRPO, PPO-Lagrangian, TRPO-Lagrangian and CPO.

- **Training protocol:** same journeys and seeds for every learner.
- **Selection:** on separate tuning journeys.
- **TRPO** won on tuning, but tied with PPO on the held-out set and needed the shield 2–3× more often.
- **The constrained learners** learned to drive slowly (40–48 s late).
- **Training with the shield in the loop** cuts shield activation tenfold for PPO and TRPO, at a cost of a few percent of LossScore.

## Repository structure

| Path | Contents |
|---|---|
| `core/havaei_generalizing_rl.py` | Nonlinear train plant (Davis drag, grade/curve forces, adhesion, actuator lag, optional train length), heteroscedastic neural emulator |
| `core/pure_mpc_controller.py` | State-augmented LTV-MPC with closed-form QP |
| `core/fuzzy_pid_controller.py` | Fuzzy-PID baselines (stage 1–2 single-corridor study) |
| `deadline_station_trackGen/dynamic_railroad_mpc.py` | **Main runner**: RL environment, mass ID, disturbance observer, PPO training, 15-method benchmark |
| `deadline_station_trackGen/railroad_corridor.py` | Multi-station journey generator, speed zones, passenger mass, physical wind and weather regimes |
| `deadline_station_trackGen/real_world_rail_weather.py` | Digitraffic timetables + Open-Meteo weather, extreme-weather injection |
| `deadline_station_trackGen/goodall_trajectory_shield.py` | Contract shield: 99% tube, pressures/margins, contract searching tree, UCB-DMPS |
| `deadline_station_trackGen/cosine_trajectory_shaper.py` | $C^2$ half-cosine and Hermite reference shapers |
| `deadline_station_trackGen/literature_baselines.py` | Literature baselines 9–15 |
| `deadline_station_trackGen/bilbao_cbtc_ato.py` | Literature baseline 8 (Bilbao-Moreno et al., 2025) |
| `deadline_station_trackGen/benchmark_mass_id.py` | Departure mass-identification benchmark |
| `deadline_station_trackGen/weather_update_study.py` | Slow-weather update study (station vs. weather service along the route, with and without a passing front) |
| `deadline_station_trackGen/generate_paper_figures.py`, `generate_architecture_diagram.py` | Paper figures |
| `deadline_station_trackGen/masa_rail/` | MASA-Safe-RL pilot study (`pilot_masa.py` training, `eval_pilots.py` scoring; `rail_env.py` / `run_masa.py` are an earlier tabular experiment, unused) |
| `experiments/` | Stage 1–2 single-corridor and emulator ablation scripts |

Not tracked by git (see `.gitignore`): checkpoints (`*.pt`, `*.npz`), figures (`*.png`), data (`*.csv`, `*.json`), the paper sources and PDF (`*.tex`, `*.pdf`), `CHANGES.md`, and `external/`. Without the checkpoints, the runner trains the emulator and pilots from scratch.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy torch matplotlib
```

## Usage

Run from `deadline_station_trackGen/`.

```bash
cd deadline_station_trackGen

# Synthetic benchmark with the pre-trained pilots (30 journeys, 1-5 stations, normal + extreme weather)
python dynamic_railroad_mpc.py --eval-runs 30 --n-legs 0 --wind-mode mixed --harsh-weather mixed \
    --load-checkpoint best_ppo_cosine_pilot.pt

# Real-world benchmark (Digitraffic + Open-Meteo; needs network access)
python dynamic_railroad_mpc.py --real-data --train-date 2026-11-16 --eval-runs 30 --n-legs 0 --wind-mode mixed \
    --load-checkpoint best_ppo_cosine_pilot.pt

# Train the emulator and both pilots from scratch (400 epochs = 1,200 training journeys)
python dynamic_railroad_mpc.py --epochs 400 --eval-runs 30 --n-legs 0 --wind-mode mixed
```

Main options:

| Option | Meaning |
|---|---|
| `--eval-runs N` | number of test journeys |
| `--n-legs K` | stations per journey (`0` = random 1–5) |
| `--wind-mode` | `mixed`, `disjoint`, `continuous` or `none` |
| `--harsh-weather` | extreme-weather injection: `mixed` (half the journeys), `always`, `raw`/`none` (default `mixed` with `--real-data`) |
| `--real-data`, `--train-date`, `--train-number` | real-world journeys from Digitraffic |
| `--anemometer2-x-m X` | position of the second platform-end anemometer (default 10 m; `0` = station anemometer only) |
| `--no-literature` | skip literature baselines 9–15 |
| `--load-checkpoint`, `--free-checkpoint` | pre-trained cosine / non-cosine PPO pilots (looked up in `checkpoints/`) |
| `--pilot-npz`, `--free-pilot-npz` | use a pilot exported by the MASA study instead |
| `--train-cars N` | distributed-length train with N cars (length-averaged grade/curve forces, size-dependent drag, tail clearance); `0` with `--real-data` uses the real composition length |
| `--a-start-center` | departure acceleration centre (ablation) |
| `--weather-update` | slow-weather knowledge: `station`, `online` (default: station + weather service along the route) or `oracle`; only matters when the plant weather varies within a leg |
| `--coast` | enable pre-landing coasting (off by default) |
| `--pin-core C`, `--no-pin-cpu` | CPU core for latency measurements (default: highest core), or no pinning |
| `--plot`, `--dual-plot`, `--data-dir` | output figure and data paths |

### MASA-Safe-RL pilot study

This part needs a separate environment with [MASA-Safe-RL](https://github.com/sacktock/MASA-Safe-RL) (JAX/Flax) installed under `external/`. Run from `deadline_station_trackGen/masa_rail/`:

```bash
python pilot_masa.py --algo TRPO --seed 42 --out OUT_DIR                     # same protocol as our PPO
python pilot_masa.py --algo PPO --seed 2024 --shield-train --interv-penalty 0.02 --out OUT_DIR
python eval_pilots.py OUT_DIR/TRPO_pilot.npz result.json                     # score on the tuning journeys
```

`pilot_masa.py` turns off MASA's global JAX 64-bit mode after importing the library. MASA's probabilistic-shield modules switch it on, and that breaks TRPO, TRPO-Lagrangian and CPO on continuous actions.

## Main references

- P. Havaei and M. A. Sandidzadeh, "Intelligent-PID controller design for speed track in automatic train operation system with heuristic algorithms," *J. Rail Transp. Plan. Manag.*, 2022.
- A. Banerjee et al., "Dynamic model predictive shielding for provably safe reinforcement learning," arXiv:2405.13863, 2024.
- A. W. Goodall and F. Belardinelli, "Safe reinforcement learning via recovery-based shielding with Gaussian process dynamics models," AAMAS, 2026.
- A. Bilbao-Moreno et al., "ATO system for CBTC-based railways: Planning and disturbance estimation for enhanced automation," *Results in Engineering*, 2025.
- Z. Xiao et al., "Eco-driving for metro trains: A computationally efficient approach using convex programming," *IEEE Trans. Veh. Technol.*, 2023.
- W. Zhong et al., "On-line train speed profile generation of high-speed railway with energy-saving: A model predictive control method," *IEEE Trans. Intell. Transp. Syst.*, 2022.
- A. W. Goodall, O. Adalat, E. Hamel-De le Court, and F. Belardinelli, "MASA-Safe-RL: Multi and single agent safe reinforcement learning," GitHub, 2025.

The full reference list (60 entries) is in the paper.
