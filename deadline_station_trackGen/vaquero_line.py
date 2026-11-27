"""
Real-track suite from the public high-speed line dataset of Vaquero-Serrano & Felez (2026),
"An Intelligent Learning-Based Model Predictive Control Framework for High-Speed Train Control under
Moving Block Signaling", Appl. Sci. 16(12):5822.  Dataset: Zenodo 10.5281/zenodo.20507243 (CC-BY 4.0),
file MPC_Line_HSPEED_2026.xlsx.

The dataset gives, per metre of a ~126 km line ("GO" direction, PK -1005 .. 124800 m):
  - grade (per mille, positive = uphill; checked against the force of their DP profile, corr 0.82),
  - curve radius (m; |R| >= 1e9 = straight),
  - speed limit for a stopping and a non-stopping service (km/h),
  - their dynamic-programming speed/force reference, and the train data (457.4 t, 8.8 MW, 350 km/h).

How the suite uses it (our trained models are not retrained):
  - Journeys keep everything the synthetic suite draws for the same seed (legs and lengths, passengers and
    mass, weather regime, wind, sensor noise, extreme-weather episodes), so the two suites are paired.
  - The track is replaced: each journey is placed on a contiguous stretch of the real line (legs follow
    each other along increasing PK) and gets its real grade and curve profile.  Leg ends are virtual stops.
  - Speed limits are the real ones capped at our train's line speed (default 17.5 m/s = 63 km/h, the
    top trunk limit of the calibrated corridor); only real limits below the cap stay visible.
  - Journeys avoid the two real station stop zones (PK -90..-30 and 94050..94110, 1.8 km/h).
  - Each deadline is shifted by the change in minimum running time (same point-mass model on the synthetic
    and on the real track), so the schedule slack of every leg is the synthetic one.

Install (download, checksum, convert to data/vaquero/vaquero_line_go.npz; no third-party Excel reader):
    python vaquero_line.py --install
Summary of the line and of the 30 benchmark journeys (no controller is run):
    python vaquero_line.py --summary
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import os
import re
import sys
import urllib.request
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "core"))

from havaei_generalizing_rl import NonlinearTrainPhysics          # noqa: E402
from railroad_corridor import SpeedLimitZone                      # noqa: E402

ZENODO_URL = "https://zenodo.org/records/20507243/files/MPC_Line_HSPEED_2026.xlsx?download=1"
ZENODO_DOI = "10.5281/zenodo.20507243"
XLSX_MD5 = "a039dd09f5e7e21a1680ffe0f1536674"
DATA_DIR = SCRIPT_DIR / "data" / "vaquero"
NPZ_PATH = DATA_DIR / "vaquero_line_go.npz"
TRAIN_JSON = DATA_DIR / "vaquero_train.json"

STRAIGHT_R_M = 1e9
# Real station stop zones (1.8 km/h) on the GO line; journeys stay out of them (with margin).
STOP_ZONES_PK = ((-90.0, -30.0), (94050.0, 94110.0))
# Usable stretches for our virtual-stop journeys: after the first station, and either side of the second.
USABLE_PK = ((0.0, 94000.0), (94200.0, 124800.0))
DEFAULT_V_CAP_MPS = 17.5

# Train data: the parameter names are images in the sheet, so they are inferred from units and magnitude.
TRAIN_FIELDS = (
    ("mass_kg", "(kg)"), ("length_m", "(m)"), ("davis_A_N", "(N)"), ("davis_B_N_per_mps", "(N/(m/s))"),
    ("davis_C_N_per_mps2", "(N/(m/s)2)"), ("traction_time_constant_s", "(s)"), ("max_force_N", "(N)"),
    ("max_power_W", "(W)"), ("rotating_mass_factor", ""), ("g_mps2", "(m/s2)"),
)
TRAIN_FIELDS_RIGHT = (
    ("mpc_horizon_steps", "(steps)"), ("mpc_control_steps", "(steps)"), ("dt_s", "(s)"), ("v_max_mps", "(m/s)"),
    ("speed_tolerance_mps", "(m/s)"), ("distance_param_1_m", "(m)"), ("distance_param_2_m", "(m)"),
    ("comfort_accel_mps2", "(m/s2)"), ("comfort_jerk_mps3", "(m/s3)"), ("coaches_per_trainset", ""),
)


# =====================================================================================================
# Install: download + checksum + stdlib xlsx parsing
# =====================================================================================================
_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
_RID = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"


def _xlsx_col(ref: str) -> int:
    n = 0
    for ch in re.match(r"[A-Z]+", ref).group():
        n = n * 26 + ord(ch) - 64
    return n - 1


def _read_xlsx_sheets(xlsx_bytes: bytes) -> dict[str, list[list[str]]]:
    """Values of every sheet as rows of strings (stdlib only)."""
    z = zipfile.ZipFile(io.BytesIO(xlsx_bytes))
    ss = [''.join(t.itertext()) for t in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", _NS)]
    rels = {r.get("Id"): r.get("Target") for r in ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))}
    out: dict[str, list[list[str]]] = {}
    for sh in ET.fromstring(z.read("xl/workbook.xml")).find("m:sheets", _NS):
        path = "xl/" + rels[sh.get(_RID)].lstrip("/").replace("xl/", "")
        rows: list[list[str]] = []
        with z.open(path) as f:
            for _ev, el in ET.iterparse(f):
                if not el.tag.endswith("}row"):
                    continue
                vals: dict[int, str] = {}
                for c in el.findall("m:c", _NS):
                    v = c.find("m:v", _NS)
                    if v is None:
                        t = c.find("m:is", _NS)
                        val = ''.join(t.itertext()) if t is not None else ""
                    else:
                        val = ss[int(v.text)] if c.get("t") == "s" else v.text
                    vals[_xlsx_col(c.get("r"))] = val
                if vals:
                    rows.append([vals.get(i, "") for i in range(max(vals) + 1)])
                el.clear()
        out[sh.get("name")] = rows
    return out


def _num(s: str) -> float:
    return float(str(s).replace(" ", "").replace("E+", "e+"))


def install(force: bool = False, xlsx_path: str | None = None) -> Path:
    """Downloads the Zenodo file (or uses `xlsx_path`), checks its MD5 and writes the compact NPZ + train JSON."""
    if NPZ_PATH.exists() and TRAIN_JSON.exists() and not force:
        print(f"[vaquero] already installed: {NPZ_PATH}")
        return NPZ_PATH
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if xlsx_path:
        raw = Path(xlsx_path).read_bytes()
        src = ZENODO_URL + " (local copy, checksum-verified)"
    else:
        print(f"[vaquero] downloading {ZENODO_URL}")
        req = urllib.request.Request(ZENODO_URL, headers={"User-Agent": "rail-benchmark-dataset-fetch/1.0"})
        with urllib.request.urlopen(req, timeout=120) as r:
            raw = r.read()
        src = ZENODO_URL
    md5 = hashlib.md5(raw).hexdigest()
    if md5 != XLSX_MD5:
        raise RuntimeError(f"MD5 mismatch for the Vaquero dataset: got {md5}, expected {XLSX_MD5}")
    sheets = _read_xlsx_sheets(raw)

    def table(name: str, ncol: int) -> np.ndarray:
        rows = sheets[name][1:]
        return np.array([[_num(r[i]) for i in range(ncol)] for r in rows if r and r[0] != ""], dtype=np.float64)

    line = table("Line GO", 3)                       # PK, grade (per mille), R (m)
    v_stop = table("Velocity GO STOP", 2)            # PK, Vmax (m/s)
    v_non = table("Velocity GO NONSTOP", 2)
    dp_stop = table("DP_Velocity GO STOP", 3)        # PK, V_DP (m/s), force (N)
    dp_non = table("DP_Velocity GO NONSTOP", 3)
    pk = line[:, 0]
    for t in (v_stop, v_non, dp_stop, dp_non):
        if t.shape[0] != pk.shape[0] or not np.array_equal(t[:, 0], pk):
            raise RuntimeError("Vaquero sheets are not on the same PK grid")
    if not np.all(np.diff(pk) == 1.0):
        raise RuntimeError("Vaquero line is not on a 1 m grid")
    np.savez_compressed(
        NPZ_PATH, pk_m=pk, grade_permille=line[:, 1], radius_m=line[:, 2],
        vmax_stop_mps=v_stop[:, 1], vmax_nonstop_mps=v_non[:, 1],
        dp_stop_v_mps=dp_stop[:, 1], dp_stop_force_N=dp_stop[:, 2],
        dp_nonstop_v_mps=dp_non[:, 1], dp_nonstop_force_N=dp_non[:, 2],
        source=np.array(f"{src} | DOI {ZENODO_DOI} | CC-BY 4.0 | md5 {md5}"),
    )
    td = sheets["TrainData"][1:]
    train: dict[str, object] = {"_note": "Parameter labels are images in the sheet; names inferred from units.",
                                "_source": f"DOI {ZENODO_DOI}"}
    for (key, _u), row in zip(TRAIN_FIELDS, td):
        if len(row) > 1 and row[1] != "":
            train[key] = _num(row[1])
    for (key, _u), row in zip(TRAIN_FIELDS_RIGHT, td):
        if len(row) > 3 and row[3] != "":
            train[key] = _num(row[3])
    TRAIN_JSON.write_text(json.dumps(train, indent=2))
    print(f"[vaquero] wrote {NPZ_PATH} ({pk.size} m of line) and {TRAIN_JSON} (md5 {md5} verified)")
    return NPZ_PATH


# =====================================================================================================
# Line model
# =====================================================================================================
class VaqueroLine:
    """Per-metre grade / curve / speed-limit profile of the GO line, with exact window means."""

    def __init__(self, npz_path: Path = NPZ_PATH, service: str = "nonstop"):
        if not Path(npz_path).exists():
            raise FileNotFoundError(f"{npz_path} missing: run `python vaquero_line.py --install` first")
        d = np.load(npz_path)
        self.pk0 = float(d["pk_m"][0])
        self.n = int(d["pk_m"].size)
        self.pk_end = self.pk0 + self.n - 1
        g = 9.81
        grade_acc = g * d["grade_permille"] / 1000.0                     # m/s^2 per unit mass, + = uphill
        r = np.abs(d["radius_m"])
        curve_acc = np.where(r >= STRAIGHT_R_M, 0.0, g / np.maximum(r, 1.0))  # same g/R law as the plant
        self.grade_permille = d["grade_permille"].astype(np.float64)
        self.radius_m = d["radius_m"].astype(np.float64)
        self.vmax_mps = (d["vmax_nonstop_mps"] if service == "nonstop" else d["vmax_stop_mps"]).astype(np.float64)
        self.source = str(d["source"])
        self._ga = grade_acc.tolist()
        self._ca = curve_acc.tolist()
        self._gcum = np.concatenate([[0.0], np.cumsum(grade_acc)]).tolist()
        self._ccum = np.concatenate([[0.0], np.cumsum(curve_acc)]).tolist()

    def _idx(self, pk: float) -> int:
        i = int(math.floor(pk - self.pk0))
        return 0 if i < 0 else (self.n - 1 if i >= self.n else i)

    def grade_acc(self, pk: float) -> float:
        return self._ga[self._idx(pk)]

    def curve_acc(self, pk: float) -> float:
        return self._ca[self._idx(pk)]

    def _mean(self, cum: list, pk_lo: float, pk_hi: float) -> float:
        a, b = self._idx(pk_lo), self._idx(pk_hi)
        return (cum[b + 1] - cum[a]) / (b - a + 1)

    def grade_acc_mean(self, pk_lo: float, pk_hi: float) -> float:
        return self._mean(self._gcum, pk_lo, pk_hi)

    def curve_acc_mean(self, pk_lo: float, pk_hi: float) -> float:
        return self._mean(self._ccum, pk_lo, pk_hi)

    def capped_zones(self, pk_a: float, length_m: float, v_cap: float, min_zone_m: float = 20.0) -> list[SpeedLimitZone]:
        """Leg-local speed zones from min(real limit, v_cap) on [pk_a, pk_a + length_m]; 1 m ramp steps are
        merged into the lower neighbouring limit (conservative), and the last zone runs 500 m past the stop."""
        i0, i1 = self._idx(pk_a), self._idx(pk_a + length_m)
        v = np.minimum(self.vmax_mps[i0:i1 + 1], v_cap)
        runs: list[list[float]] = []                                   # [x_start, x_end, v]
        start = 0
        for k in range(1, v.size + 1):
            if k == v.size or v[k] != v[start]:
                runs.append([float(start), float(k), float(v[start])])
                start = k
        merged: list[list[float]] = []
        for run in runs:
            if merged and (run[1] - run[0] < min_zone_m or merged[-1][1] - merged[-1][0] < min_zone_m):
                merged[-1][1] = run[1]
                merged[-1][2] = min(merged[-1][2], run[2])
            elif merged and abs(run[2] - merged[-1][2]) < 1e-9:
                merged[-1][1] = run[1]
            else:
                merged.append(run)
        zones = [SpeedLimitZone(a, b, round(vv, 3), f"Real PK {pk_a + a:.0f}-{pk_a + b:.0f}") for a, b, vv in merged]
        zones[-1] = SpeedLimitZone(zones[-1].x_start, length_m + 500.0, zones[-1].v_limit_mps, zones[-1].label)
        return zones


_LINE_CACHE: dict[str, VaqueroLine] = {}


def get_line(service: str = "nonstop") -> VaqueroLine:
    if service not in _LINE_CACHE:
        _LINE_CACHE[service] = VaqueroLine(service=service)
    return _LINE_CACHE[service]


# =====================================================================================================
# Plant hook: real grade / curve forces for the true and the controller model
# =====================================================================================================
class RealTrackPhysics(NonlinearTrainPhysics):
    """NonlinearTrainPhysics whose grade/curve profile is the real line from PK `track_pk0` on (leg x = 0)."""

    track: VaqueroLine
    track_pk0: float

    def grade_resistance(self, x: float) -> float:
        pk = self.track_pk0 + x
        if self.length_m > 0.0:
            return self.M * self.track.grade_acc_mean(pk - self.length_m, pk)
        return self.M * self.track.grade_acc(pk)

    def curve_resistance(self, x: float) -> float:
        pk = self.track_pk0 + x
        if self.length_m > 0.0:
            return self.M * self.track.curve_acc_mean(pk - self.length_m, pk)
        return self.M * self.track.curve_acc(pk)


def attach_real_track(physics: NonlinearTrainPhysics, corridor) -> None:
    """Turns `physics` into a RealTrackPhysics when the leg corridor carries a real-track reference."""
    pk0 = getattr(corridor, "real_track_pk0", None)
    if pk0 is None:
        return
    physics.__class__ = RealTrackPhysics
    physics.track = get_line(getattr(corridor, "real_track_service", "nonstop"))
    physics.track_pk0 = float(pk0)


def install_env_hook(env_cls) -> None:
    """Wraps env._sync_physics_and_mpc so the true plant and the controller model both use the real track.
    Every controller (MPC, shields, Bilbao, planners) reads grade/curve through env.physics."""
    if getattr(env_cls, "_vaquero_hooked", False):
        return
    orig = env_cls._sync_physics_and_mpc

    def _sync_with_real_track(self, *a, **kw):
        orig(self, *a, **kw)
        attach_real_track(self.true_physics, self.corridor)
        attach_real_track(self.physics, self.corridor)

    env_cls._sync_physics_and_mpc = _sync_with_real_track
    env_cls._vaquero_hooked = True


# =====================================================================================================
# Journeys
# =====================================================================================================
def min_running_time_s(corridor, physics: NonlinearTrainPhysics, a_acc: float = 0.96, b_dec: float = 0.78,
                       dx: float = 2.0) -> float:
    """Point-mass minimum running time over the leg: accelerate at min(a_acc, traction) against Davis,
    grade and curve forces, respect the zone limits, brake at b_dec into lower limits and the stop."""
    L = float(corridor.station_x_m)
    n = int(math.ceil(L / dx)) + 1
    xs = np.linspace(0.0, L, n)
    h = xs[1] - xs[0]
    ceil = np.array([corridor.get_speed_limit(x) for x in xs])
    ceil[-1] = 0.0
    for i in range(n - 2, -1, -1):                                       # braking envelope
        ceil[i] = min(ceil[i], math.sqrt(ceil[i + 1] ** 2 + 2.0 * b_dec * h))
    v = np.zeros(n)
    m_eff = physics.M_eff
    for i in range(n - 1):                                               # traction envelope
        vi = v[i]
        f_res = physics.basic_resistance(vi) + physics.grade_resistance(xs[i]) + physics.curve_resistance(xs[i])
        a = min(a_acc, physics.max_propulsion_force(vi, 1.0) / m_eff) - f_res / m_eff
        v[i + 1] = min(ceil[i + 1], math.sqrt(max(0.0, vi * vi + 2.0 * a * h)))
    v_mid = np.maximum(0.5 * (v[1:] + v[:-1]), 0.3)
    return float(np.sum(h / v_mid))


def _leg_physics(corridor) -> NonlinearTrainPhysics:
    ph = NonlinearTrainPhysics(mass_kg=corridor.mass_kg, car_scale=corridor.car_scale,
                               length_m=float(getattr(corridor, "train_length_m", 0.0)))
    ph.max_traction_accel = corridor.max_traction_accel
    ph.max_braking_decel = corridor.max_service_braking
    return ph


def _place_journey(total_m: float, rng: np.random.Generator) -> float:
    """Start PK of a journey of `total_m` metres inside one usable stretch (chosen by its spare room)."""
    room = [max(0.0, (b - a) - total_m) for a, b in USABLE_PK]
    if sum(room) <= 0.0:
        raise ValueError(f"journey of {total_m:.0f} m does not fit the usable stretches")
    k = int(rng.choice(len(USABLE_PK), p=np.array(room) / sum(room)))
    return float(np.round((USABLE_PK[k][0] + rng.uniform(0.0, room[k])) / 10.0) * 10.0)


def to_real_track(scen: dict, seed: int, v_cap: float = DEFAULT_V_CAP_MPS, service: str = "nonstop") -> dict:
    """Moves a synthetic journey onto the real line (see module docstring)."""
    line = get_line(service)
    legs = scen["legs"]
    total = float(sum(l["leg_length_m"] for l in legs))
    pk = _place_journey(total, np.random.default_rng(seed * 7919 + 17))   # own stream: synthetic draws unchanged
    shift = 0.0                                                          # cumulative deadline shift (s)
    for leg in legs:
        cor = leg["corridor"]
        L = float(leg["leg_length_m"])
        t_syn = min_running_time_s(cor, _leg_physics(cor))
        zones = line.capped_zones(pk, L, v_cap)
        new = copy.copy(cor)                       # keeps any attribute set after construction
        new.speed_zones = zones
        new.route_name = f"Vaquero HSL GO PK {pk / 1000:.2f}-{(pk + L) / 1000:.2f} km (virtual stops)"
        new.real_track_pk0 = pk
        new.real_track_service = service
        ph_real = _leg_physics(new)
        attach_real_track(ph_real, new)
        d_t = float(np.round((min_running_time_s(new, ph_real) - t_syn) * 2.0) / 2.0)
        new.start_time_s = cor.start_time_s + shift
        shift += d_t
        new.deadline_s = cor.deadline_s + shift
        # Wind and anemometer series are indexed per 0.5 s step: extend them (edge value) for longer legs.
        n_need = int(math.ceil((new.deadline_s - new.start_time_s + 150.0) / 0.5)) + 40
        if len(leg["wind_profile"]) < n_need:
            leg["wind_profile"] = np.pad(np.asarray(leg["wind_profile"]), (0, n_need - len(leg["wind_profile"])), mode="edge")
        sig = getattr(new, "anemometer_sigma_mps2", None)
        if sig is not None and len(sig) < n_need:
            new.anemometer_sigma_mps2 = np.pad(np.asarray(sig), (0, n_need - len(sig)), mode="edge")
        i0, i1 = line._idx(pk), line._idx(pk + L)
        r = np.abs(line.radius_m[i0:i1 + 1])
        leg.update({
            "corridor": new,
            "t_start": new.start_time_s,
            "deadline_s": new.deadline_s,
            "budget_s": leg["budget_s"] + d_t,
            "deadline_shift_s": d_t,
            "pk_start_m": pk,
            "pk_end_m": pk + L,
            "grade_min_permille": float(line.grade_permille[i0:i1 + 1].min()),
            "grade_max_permille": float(line.grade_permille[i0:i1 + 1].max()),
            "min_radius_m": float(r[r < STRAIGHT_R_M].min()) if np.any(r < STRAIGHT_R_M) else float("inf"),
            "v_limit_min_mps": float(min(z.v_limit_mps for z in zones)),
        })
        pk += L
    scen["track_source"] = f"Vaquero-Serrano & Felez (2026), DOI {ZENODO_DOI}, CC-BY 4.0"
    scen["track_v_cap_mps"] = v_cap
    return scen


def make_scenario_generator(base_generator, v_cap: float = DEFAULT_V_CAP_MPS, service: str = "nonstop"):
    """Wraps generate_multi_station_scenario so every journey runs on the real line."""
    def gen(seed, *a, **kw):
        return to_real_track(base_generator(seed, *a, **kw), int(seed), v_cap=v_cap, service=service)
    gen.__wrapped__ = base_generator
    return gen


# =====================================================================================================
# CLI
# =====================================================================================================
def _summary(eval_runs: int, n_legs: int, v_cap: float) -> None:
    from railroad_corridor import generate_multi_station_scenario
    line = get_line()
    print(f"Line: PK {line.pk0:.0f}..{line.pk_end:.0f} m ({(line.pk_end - line.pk0) / 1000:.1f} km), "
          f"grade {line.grade_permille.min():+.1f}..{line.grade_permille.max():+.1f} per mille, "
          f"min |R| {np.abs(line.radius_m).min():.0f} m | {line.source}")
    gen = make_scenario_generator(generate_multi_station_scenario, v_cap=v_cap)
    rows = []
    for ep in range(eval_runs):                         # same seeds / extreme split as the synthetic suite
        scen = gen(9000 + ep * 29, n_legs=n_legs, wind_mode="mixed", allow_infeasible_breaks=(ep > 0),
                   extreme_noise=(ep % 2 == 1))
        for leg in scen["legs"]:
            rows.append((ep, leg["leg_idx"], leg["pk_start_m"], leg["leg_length_m"], leg["grade_min_permille"],
                         leg["grade_max_permille"], leg["min_radius_m"], leg["v_limit_min_mps"],
                         leg["deadline_shift_s"], leg["budget_s"]))
    print(f"{'ep':>3} {'leg':>3} {'PK km':>8} {'len m':>6} {'grade min/max':>14} {'min R':>7} {'v_min':>6} {'dT s':>6} {'budget s':>8}")
    for r in rows:
        print(f"{r[0]:>3} {r[1]:>3} {r[2] / 1000:>8.2f} {r[3]:>6.0f} {r[4]:>+6.1f}/{r[5]:>+6.1f} "
              f"{r[6]:>7.0f} {r[7]:>6.1f} {r[8]:>+6.1f} {r[9]:>8.1f}")
    shifts = np.array([r[8] for r in rows])
    print(f"{len(rows)} legs; deadline shift mean {shifts.mean():+.1f} s, range {shifts.min():+.1f}..{shifts.max():+.1f} s; "
          f"{sum(r[7] < v_cap for r in rows)} legs with a real limit below the {v_cap} m/s cap")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--install", action="store_true", help="Download, verify and convert the dataset")
    ap.add_argument("--force", action="store_true", help="Re-install even if the NPZ exists")
    ap.add_argument("--xlsx", default="", help="Use a local copy of MPC_Line_HSPEED_2026.xlsx instead of downloading")
    ap.add_argument("--summary", action="store_true", help="Print the line and the benchmark journeys (no simulation)")
    ap.add_argument("--eval-runs", type=int, default=30)
    ap.add_argument("--n-legs", type=int, default=0)
    ap.add_argument("--v-cap", type=float, default=DEFAULT_V_CAP_MPS)
    args = ap.parse_args()
    if args.install:
        install(force=args.force, xlsx_path=args.xlsx or None)
    if args.summary:
        _summary(args.eval_runs, args.n_legs, args.v_cap)
    if not (args.install or args.summary):
        ap.print_help()


if __name__ == "__main__":
    main()
