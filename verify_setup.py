#!/usr/bin/env python3
"""Environment check. Must reproduce tau1 = 29.57 yr, chi2 = 0.045."""
import sys, os, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ok = True
def chk(label, cond, detail=""):
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}{'  ' + detail if detail else ''}")
    ok = ok and cond
    return cond

print("DLPMI environment check\n")
import torch, numpy, pandas, scipy
print(f"  torch {torch.__version__} | numpy {numpy.__version__}\n")

import dlpmi
from dlpmi import Inverter, Sample

# 1. token-based free-parameter matching
from dlpmi.params import BMMModel
m = BMMModel("DM", "DM", free_params="Fraction, 2nd Mean Age",
             params1_init={"tau": 20., "PD": .01},
             params2_init={"tau": 1000., "PD": .01})
chk("params.py uses token matching (tau1 fixed in Mode C)",
    not m._tau1_free, "'Fraction, 2nd Mean Age' must not free tau1")

# 2. Jurgens input curves present
try:
    from dlpmi.tracers import _load_curves
    df = _load_curves()
    chk("tracer_input_curves.csv loaded", df.shape[0] > 1000,
        f"{df.shape[0]} rows, {df.shape[1]} cols")
except Exception as e:
    chk("tracer_input_curves.csv loaded", False, str(e)[:60])

# 3. metrics module
try:
    from dlpmi.metrics import young_fraction, mixture_mean_age
    chk("dlpmi/metrics.py importable", True)
except Exception as e:
    chk("dlpmi/metrics.py importable", False, str(e)[:60])

# 4. reference fit
cfg = os.path.join("examples", "edwards_aquifer", "edwards_input_config.json")
rec = [x for x in json.load(open(cfg))["samples"]
       if x["sample_id"] == "EDTRPAS1-36"][0]
tr = [{"name": t["name"], "obs": t["obs"], "obs_err": t["obs_err"],
       "scale": t["scale"], "active": t["scale"] != 0.} for t in rec["tracers"]]
b, ini = rec["pinn_bounds"], rec["lpm"]["init"]
torch.manual_seed(0)
s = Sample(sample_id=rec["sample_id"], sample_date=rec["sample_date"],
           tracers=tr, model="DM",
           bounds={"tau0": ini["tau1_yr"], "tau_lo": b["tau1_lo"],
                   "tau_hi": b["tau1_hi"], "pd0": ini["pd1"],
                   "pd_lo": b["pd1_lo"], "pd_hi": b["pd1_hi"]},
           he4_rate=rec["he4_params"]["solution_rate_ccpgpyr"],
           dgmeta=rec["dgmeta_params"], age_category=rec["age_category"])
res = Inverter(n_adam=800, n_starts=1, verbose=False).fit(s)
tau, c2 = res["params"]["tau"], res["chi2"]
chk("EDTRPAS1-36 reference fit", abs(tau - 29.57) < 0.5 and c2 < 0.10,
    f"tau1={tau:.3f} yr (expect 29.57), chi2={c2:.4f} (expect 0.045)")

print("\n" + ("All checks passed." if ok else
      "FAILURES above — resolve before running sweeps."))
sys.exit(0 if ok else 1)
