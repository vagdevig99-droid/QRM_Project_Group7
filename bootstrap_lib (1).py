"""
Block-bootstrap uncertainty quantification, built on top of the existing
project pipeline (marginal.py, models.py, rainfall_wgan.py) with no changes
to those files.

Where this code goes, relative to the existing pipeline:
    rainfall_wgan.load_london_csv   -> unchanged, used as-is to load the series
    rainfall_wgan.moving_block_bootstrap -> unchanged, used as-is to resample
    models.py (M2-M7)               -> unchanged, .fit()/.simulate()/.return_level()
    m1_potgpd.POT_GPD (new, additive) -> same interface, fills the missing M1
    THIS FILE (new, additive)       -> risk_measures() + bootstrap loops that
                                        call the above with no modification.
"""

import os
import time
import numpy as np
import pandas as pd

import rainfall_wgan as rw
from models import (RawGAN, SpliceGAN, NormalScoreGAN, MarkovWeatherGen,
                     CopulaWindow, GEVAnnualMax, _windows)
from m1_potgpd import POT_GPD

L = 7
ALPHAS = (0.99, 0.999)
T_LIST = (10, 30, 100)
NSIM_RISK = 200_000   # simulated windows used to read off VaR/CVaR per fit


# ---------------------------------------------------------------------------
# Uniform risk-measure extraction across all seven models
# ---------------------------------------------------------------------------

def risk_measures(model, n=NSIM_RISK):
    """VaR/CVaR at ALPHAS and return levels at T_LIST, from whichever
    interface the model exposes. Models with .simulate() (M2-M6) are read
    off a fresh Monte Carlo sample of window totals; M1 (POT_GPD) and M7
    (GEVAnnualMax) are analytic and use their own closed-form methods."""
    out = {}

    if hasattr(model, "simulate"):
        try:
            totals = model.simulate(n).sum(1)
            for a in ALPHAS:
                v = float(np.quantile(totals, a))
                tail = totals[totals >= v]
                out[f"VaR_{a}"] = v
                out[f"CVaR_{a}"] = float(tail.mean()) if len(tail) else np.nan
        except NotImplementedError:
            pass
    elif hasattr(model, "var"):
        for a in ALPHAS:
            out[f"VaR_{a}"] = model.var(a)
            out[f"CVaR_{a}"] = model.cvar(a)

    for T in T_LIST:
        try:
            out[f"RL_{T}"] = model.return_level(T)
        except Exception:
            out[f"RL_{T}"] = np.nan

    return out


# ---------------------------------------------------------------------------
# Bootstrap resampling
# ---------------------------------------------------------------------------

def iid_bootstrap(series, rng):
    """Ordinary IID bootstrap: resample individual days with replacement.
    Destroys serial dependence - kept only as the sanity-check comparator
    that justifies using the block bootstrap instead."""
    n = len(series)
    idx = rng.integers(0, n, size=n)
    return series[idx]


def block_bootstrap(series, block_len, rng):
    """Thin wrapper around rainfall_wgan.moving_block_bootstrap so every
    caller in this file goes through one place. No logic duplicated."""
    return rw.moving_block_bootstrap(series, block_len=block_len, rng=rng)


# ---------------------------------------------------------------------------
# Dependence-preservation sanity check (block vs IID), on the ORIGINAL series
# ---------------------------------------------------------------------------

def dependence_diagnostics(series, block_len=30, n_reps=100, seed=12345):
    """For n_reps resamples of each kind, compute wet fraction, mean, sd,
    lag-1..5 autocorrelation of the wet/dry indicator, and q0.99 of 7-day
    totals. Returns a tidy DataFrame: kind x replicate x statistic."""
    rng_master = np.random.default_rng(seed)
    rows = []
    orig_stats = _series_stats(series, block_len)
    orig_stats["kind"] = "original"
    orig_stats["rep"] = -1
    rows.append(orig_stats)

    for kind, fn in (("block", block_bootstrap), ("iid", iid_bootstrap)):
        for r in range(n_reps):
            rng = np.random.default_rng(rng_master.integers(0, 2**31 - 1))
            if kind == "block":
                bs = fn(series, block_len, rng)
            else:
                bs = fn(series, rng)
            st = _series_stats(bs, block_len)
            st["kind"] = kind
            st["rep"] = r
            rows.append(st)
    return pd.DataFrame(rows)


def _series_stats(series, block_len):
    wet = series > 0
    d = {
        "wet_fraction": wet.mean(),
        "mean_mm": series.mean(),
        "sd_mm": series.std(),
    }
    for lag in (1, 2, 3, 4, 5):
        a = wet[:-lag].astype(float)
        b = wet[lag:].astype(float)
        d[f"acf_wetdry_lag{lag}"] = np.corrcoef(a, b)[0, 1]
    S = _windows(series, L, L).sum(1)
    d["total_q99"] = np.quantile(S, 0.99)
    d["total_sd"] = S.std()
    return d


# ---------------------------------------------------------------------------
# Analytic-model bootstrap: M1, M5, M6, M7 - exact refit, cheap, no torch
# ---------------------------------------------------------------------------

ANALYTIC_MODEL_SPECS = {
    "M1 POT/GPD":         lambda: POT_GPD(L=L),
    "M5 Markov weathergen": lambda: MarkovWeatherGen(L=L),
    "M6 copula":          lambda: CopulaWindow(L=L),
    "M7 GEV annual max":  lambda: GEVAnnualMax(L=L, start_year=1979),
}


def run_analytic_bootstrap(series, B, block_len=30, seed0=20000, verbose=True,
                            b_offset=0, out_csv=None, resume=True):
    """Cheap enough (~1.5s/rep for all 4 models) that resume support mostly
    matters for consistency with run_gan_bootstrap, but it's here too."""
    done_reps = set()
    if out_csv is not None and resume and os.path.exists(out_csv):
        existing = pd.read_csv(out_csv)
        done_reps = set(existing["rep"].unique().tolist())
        if verbose:
            print(f"  resuming: {len(done_reps)} reps already in {out_csv}", flush=True)

    records = []
    t_start = time.time()
    for b in range(b_offset, b_offset + B):
        if b in done_reps:
            continue
        rng = np.random.default_rng(seed0 + b)
        bs = block_bootstrap(series, block_len, rng)
        rep_rows = []
        for name, ctor in ANALYTIC_MODEL_SPECS.items():
            m = ctor()
            try:
                m.fit(bs)
                rm = risk_measures(m)
            except Exception as e:
                rm = {"error": str(e)}
            rep_rows.append({"model": name, "rep": b, **rm})
        records.extend(rep_rows)
        if out_csv is not None:
            header = not os.path.exists(out_csv)
            pd.DataFrame(rep_rows).to_csv(out_csv, mode="a", header=header, index=False)
        if verbose and (b + 1) % max(1, B // 10) == 0:
            el = time.time() - t_start
            print(f"  analytic bootstrap {b+1}/{b_offset+B}  ({el:5.1f}s elapsed)")
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# GAN-model bootstrap: M2, M3, M4 - requires retraining a WGAN-GP per rep.
# Full retraining (steps=3000, as used for the original fit) at B~200-500 is
# not computationally feasible on this machine (1 CPU core, no GPU: ~175s per
# 3000-step fit -> >>10 hours for B=200 across three models). This runner
# exposes `steps` and `B` as explicit arguments precisely so the choice is
# never silent - see run_gan_bootstrap.py for the documented reduced-scale
# run actually executed in this session, and the README for the full-scale
# command to run on a GPU machine.
# ---------------------------------------------------------------------------

def run_gan_bootstrap(series, B, steps, block_len=30, zdim=32, ch=64,
                       seed0=30000, out_csv=None, verbose=True,
                       b_offset=0, resume=True):
    """Retrains M2 and M4 (M3 splices onto M2) from scratch on each bootstrap
    resample. This is the expensive one - see README for wall-clock estimates.

    Colab-safe: if out_csv already exists and resume=True, reps already
    present in it are skipped (by rep number) and new reps are appended, not
    overwritten. This means a session disconnect only costs you the reps that
    hadn't finished writing yet, not everything already done. Every single
    replicate's 3 rows (M2, M3, M4) are flushed to out_csv immediately after
    that replicate finishes, not batched at the end.
    """
    done_reps = set()
    if out_csv is not None and resume and os.path.exists(out_csv):
        existing = pd.read_csv(out_csv)
        done_reps = set(existing["rep"].unique().tolist())
        if verbose:
            print(f"  resuming: {len(done_reps)} reps already in {out_csv}, "
                  f"skipping those", flush=True)

    records = []
    t_start = time.time()
    for b in range(b_offset, b_offset + B):
        if b in done_reps:
            continue
        rng = np.random.default_rng(seed0 + b)
        bs = block_bootstrap(series, block_len, rng)
        seed = int(seed0 + b)

        m2 = RawGAN(L=L, steps=steps, zdim=zdim, ch=ch, seed=seed).fit(bs)
        rm2 = risk_measures(m2)
        rep_rows = [{"model": "M2 raw GAN", "rep": b, **rm2}]

        m3 = SpliceGAN(m2, splice_T=10.0).fit(bs)
        rm3 = risk_measures(m3)
        rep_rows.append({"model": "M3 GAN+GPD splice", "rep": b, **rm3})

        m4 = NormalScoreGAN(L=L, steps=steps, zdim=zdim, ch=ch, seed=seed).fit(bs)
        rm4 = risk_measures(m4)
        rep_rows.append({"model": "M4 normal-score GAN", "rep": b, **rm4})

        records.extend(rep_rows)

        if out_csv is not None:
            header = not os.path.exists(out_csv)
            pd.DataFrame(rep_rows).to_csv(out_csv, mode="a", header=header, index=False)

        if verbose:
            el = time.time() - t_start
            done_now = b - b_offset + 1 - len([x for x in done_reps if x >= b_offset])
            print(f"  GAN bootstrap rep {b} done  ({el:6.1f}s elapsed)", flush=True)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Summary table: original estimate, bootstrap mean, 95% percentile interval
# ---------------------------------------------------------------------------

def summarise(boot_df, original_estimates, measures=None):
    """original_estimates: dict[model_name] -> dict[measure] -> value
    boot_df: long DataFrame with columns model, rep, <measures...>"""
    if measures is None:
        measures = [c for c in boot_df.columns if c not in ("model", "rep", "error")]
    rows = []
    for name, g in boot_df.groupby("model"):
        orig = original_estimates.get(name, {})
        for meas in measures:
            if meas not in g.columns:
                continue
            vals = g[meas].dropna().values
            if len(vals) == 0:
                continue
            rows.append({
                "model": name,
                "measure": meas,
                "original_estimate": orig.get(meas, np.nan),
                "bootstrap_mean": float(np.mean(vals)),
                "bootstrap_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
                "ci_2.5%": float(np.quantile(vals, 0.025)),
                "ci_97.5%": float(np.quantile(vals, 0.975)),
                "n_reps": len(vals),
            })
    return pd.DataFrame(rows)
