"""Retrain the PPO pilot with the eco governor (and optionally the UCB-DMPS shield) active during training.
Usage: train_eco.py PILOT(cos|free) VARIANT(base|eco|ecoshield) SEED EPOCHS
Same settings as the installed pilots (comfort_bound=True, wind_mode=mixed, 400 epochs); checkpoints go to
checkpoints/best_<variant>_<pilot>_s<seed>.pt. Validation (checkpoint choice) runs in the same configuration."""
import os, sys, json
SRC = "/home/giorge/Desktop/ProjectTh/deadline_station_trackGen"
os.chdir(SRC); sys.path[:0] = [SRC, os.path.dirname(SRC), os.path.join(os.path.dirname(SRC), "core")]
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1"); os.environ.setdefault("OMP_NUM_THREADS", "1")
import torch
torch.set_num_threads(1)
import dynamic_railroad_mpc as D
from railroad_corridor import RailroadCorridor

pilot, variant, seed, epochs = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
cos = pilot == "cos"
emu = D.LearnedLinearizationModel(hidden_dim=64)
emu.load_state_dict(torch.load("checkpoints/best_emulator.pt", map_location="cpu")); emu.eval()

if variant in ("eco", "ecoshield", "shield"):
    orig_init, orig_reset = D.MPCTransitionRailroadEnv.__init__, D.MPCTransitionRailroadEnv.reset
    def init(self, *a, **k):
        orig_init(self, *a, **k)
        self.eco_governor = variant != "shield"
        if not self.eco_governor:
            return
        if self.use_cosine_shaper:  # same eco configuration as method 5 / method 7 at test time
            self.eco_a_cap, self.eco_slack_reserve_s = 0.70, 5.0
        else:
            self.eco_a_cap, self.eco_slack_reserve_s = None, 4.0
    def reset(self, *a, **k):
        if variant in ("ecoshield", "shield"):  # fresh UCB-DMPS shield per leg, UCB statistics kept across legs (as in eval)
            cor = k.get("corridor_override") or self.corridor
            if not hasattr(self, "_ucb_state"):
                self._ucb_state = {}
            self.traj_shield = D.GoodallTrajectoryShield(
                cor, self.emulator, D.TrajectoryShieldConfig(mode="mps_rollout"), ucb_state=self._ucb_state)
        return orig_reset(self, *a, **k)
    D.MPCTransitionRailroadEnv.__init__ = init
    D.MPCTransitionRailroadEnv.reset = reset

prefix = f"{variant}_{pilot}_s{seed}"
pol, hist = D.train_pure_ppo_cosine_pilot(
    RailroadCorridor(), emulator=emu, epochs=epochs, wind_mode="mixed", seed=seed,
    checkpoint_prefix=prefix, use_cosine_shaper=cos, comfort_bound=True)
json.dump(hist, open(f"/tmp/claude-1000/-home-giorge-Desktop-ProjectTh/b16d013a-8abd-404a-82c9-8d2177fa6873/scratchpad/eco/hist_{prefix}.json", "w"))
print("done", prefix)
