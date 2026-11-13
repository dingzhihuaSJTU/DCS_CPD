"""Null law under AR(1) noise: moments and realised tail level (Table S.8)."""
import sys, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dcs_cpd as D

def ar1(rng, n, phi, var):
    e = rng.normal(0, np.sqrt(var*(1-phi**2)), n)
    x = np.empty(n); x[0] = rng.normal(0, np.sqrt(var))
    for i in range(1, n):
        x[i] = phi*x[i-1] + e[i]
    return x

def moments(snr, tau):
    mu = snr/(snr+1+1.0/tau)
    spread = np.sqrt(1.0/(tau*snr**2) + 2.0/((tau-1)*(snr+1+1.0/tau)**2))
    return mu, mu*spread, spread

rows=[]
L=200000
for snr in (2.0, 5.0):
    for phi in (0.0, 0.3, 0.6, 0.9):
        for tau in (32, 64):
            rng = np.random.default_rng(2024)
            t = np.arange(L)
            b=1.0
            var_xi=(0.5+b*b)/snr
            x = b + np.sin(2*np.pi*t/tau) + ar1(rng, L, phi, var_xi)
            y = D.cosine_series(x, tau)
            mu_p, sd_p, spread = moments(snr, tau)
            tau_eff = tau*(1-phi)/(1+phi)
            mu_e, sd_e, _ = moments(snr, tau_eff)
            yt_iid = D.threshold(0.005, snr, tau)
            yt_eff = D.threshold(0.005, snr, tau_eff)
            rows.append(dict(snr=snr, phi=phi, tau=tau, tau_eff=tau_eff,
                mu_pred=mu_p, mu_pred_eff=mu_e, mu_emp=float(y.mean()),
                sd_pred=sd_p, sd_pred_eff=sd_e, sd_emp=float(y.std()),
                alpha_iid=float((y<=yt_iid).mean()), alpha_eff=float((y<=yt_eff).mean()),
                q005=float(np.quantile(y,0.005))))

import csv
out = HERE / "results" / "ar.csv"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    w=csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
for r in rows:
    print(f"snr={r['snr']} phi={r['phi']:.1f} tau={r['tau']:2d} | mu {r['mu_pred']:.3f}/{r['mu_pred_eff']:.3f}/{r['mu_emp']:.3f} | sd {r['sd_pred']:.3f}/{r['sd_pred_eff']:.3f}/{r['sd_emp']:.3f} | a_iid {r['alpha_iid']:.4f} a_eff {r['alpha_eff']:.4f}")
