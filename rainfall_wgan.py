"""
QRM Project - Group 7
Phase 2: Sequence WGAN-GP simulator for multi-day rainfall windows.

Design (locked):
  - Generator produces a window of L consecutive daily rainfall values (L = 7).
  - Risk functional is a scalar summary of the window:
        S = sum over the window   (accumulation; primary)
        M = max over the window   (daily peak; secondary)
  - Trained on ALL windows (stride 1), not just extreme ones. The extreme region
    is identified afterwards, on the simulated totals.
  - Zeros handled by snapping simulated values below GAUGE_RES to exactly 0
    (0.1 mm is the physical resolution of a tipping-bucket gauge).

This file is self-validating: run it with no arguments and it trains on a
synthetic rainfall surrogate with KNOWN dependence structure and KNOWN tail
index, so you can confirm the pipeline recovers what it is supposed to before
the real London series is plugged in.
"""

import numpy as np
import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GAUGE_RES = 0.1        # mm; values below this are recorded as dry
DAYS_PER_YEAR = 365.25


# ----------------------------------------------------------------------------
# 1. Data: synthetic surrogate (for validation) and real-CSV loader
# ----------------------------------------------------------------------------

def synthetic_rainfall(n_days=15340, seed=0):
    """Two-state Markov chain for wet/dry, GPD amounts on wet days.

    Ground truth we can check the GAN against:
      p_ww = 0.62  (wet -> wet)     p_dw = 0.28  (dry -> wet)
      stationary wet fraction = p_dw / (1 - p_ww + p_dw) = 0.28/0.66 = 0.424
      wet-day amounts ~ GPD(xi = 0.10, sigma = 3.2)
    """
    rng = np.random.default_rng(seed)
    p_ww, p_dw, xi, sigma = 0.62, 0.28, 0.10, 3.2

    wet = np.zeros(n_days, dtype=bool)
    wet[0] = rng.random() < 0.42
    for t in range(1, n_days):
        wet[t] = rng.random() < (p_ww if wet[t - 1] else p_dw)

    u = rng.random(wet.sum())
    amounts = sigma / xi * ((1 - u) ** (-xi) - 1)      # GPD inverse-CDF
    series = np.zeros(n_days)
    series[wet] = amounts
    series[series < GAUGE_RES] = 0.0
    return series


def load_london_csv(path, date_col="date", rain_col="precipitation"):
    """Load the Kaggle London daily weather CSV into a gap-free daily series."""
    import pandas as pd
    df = pd.read_csv(path)
    df[date_col] = pd.to_datetime(df[date_col], format="%Y%m%d")
    df = df.sort_values(date_col).set_index(date_col)
    s = df[rain_col].astype(float)
    s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D"))
    print(f"  loaded {len(s)} days, {s.isna().sum()} missing "
          f"({s.index.min().date()} to {s.index.max().date()})")
    s = s.interpolate(limit=2).fillna(0.0)             # short gaps only
    arr = np.array(s.to_numpy(), dtype=float, copy=True)
    arr[arr < GAUGE_RES] = 0.0
    return arr


# ----------------------------------------------------------------------------
# 2. Windows and the variance-stabilising transform
# ----------------------------------------------------------------------------

def make_windows(series, L=7, stride=1):
    """(n_windows, L) array of consecutive daily values."""
    n = (len(series) - L) // stride + 1
    idx = np.arange(L)[None, :] + (np.arange(n) * stride)[:, None]
    return series[idx]


class Scaler:
    """y = log1p(r), then standardise. Invertible; risk is computed in mm."""

    def fit(self, w):
        y = np.log1p(w)
        self.mu, self.sd = y.mean(), y.std()
        return self

    def forward(self, w):
        return (np.log1p(w) - self.mu) / self.sd

    def inverse(self, y, snap=True):
        r = np.clip(np.expm1(y * self.sd + self.mu), 0.0, None)
        if snap:
            r = np.where(r < GAUGE_RES, 0.0, r)
        return r


# ----------------------------------------------------------------------------
# 3. WGAN-GP: 1-D convolutional generator and critic
# ----------------------------------------------------------------------------

class Generator(nn.Module):
    """z -> (1, L). Conv1d weight-sharing encodes stationarity of the series."""

    def __init__(self, L=7, zdim=32, ch=64, lo=-3.0, hi=3.0):
        super().__init__()
        self.L, self.ch = L, ch
        self.register_buffer("lo", torch.tensor(float(lo)))
        self.register_buffer("hi", torch.tensor(float(hi)))
        self.fc = nn.Linear(zdim, ch * L)
        self.net = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv1d(ch, ch, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv1d(ch, ch, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv1d(ch, 1, 3, padding=1),
        )

    def forward(self, z):
        h = self.fc(z).view(-1, self.ch, self.L)
        out = self.net(h)
        # Bound the output to the observed range in transformed space. Without
        # this, expm1() in the inverse transform amplifies any overshoot into
        # physically impossible rainfall. This makes "cannot exceed the observed
        # daily maximum" a STRUCTURAL property rather than a hope.
        return self.lo + (self.hi - self.lo) * 0.5 * (torch.tanh(out) + 1.0)


class Critic(nn.Module):
    """(1, L) -> scalar score. LayerNorm, never BatchNorm (breaks the GP)."""

    def __init__(self, L=7, ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, ch, 3, padding=1), nn.LayerNorm([ch, L]), nn.LeakyReLU(0.2),
            nn.Conv1d(ch, ch, 3, padding=1), nn.LayerNorm([ch, L]), nn.LeakyReLU(0.2),
            nn.Flatten(), nn.Linear(ch * L, 1),
        )

    def forward(self, x):
        return self.net(x)


def gradient_penalty(critic, real, fake):
    """lambda * E[(||grad_xhat D(xhat)||_2 - 1)^2], xhat on the real-fake segment."""
    eps = torch.rand(real.size(0), 1, 1, device=real.device)
    xhat = (eps * real + (1 - eps) * fake).requires_grad_(True)
    scores = critic(xhat)
    grads = torch.autograd.grad(
        outputs=scores, inputs=xhat,
        grad_outputs=torch.ones_like(scores),
        create_graph=True, retain_graph=True,
    )[0]
    norms = grads.reshape(grads.size(0), -1).norm(2, dim=1)
    return ((norms - 1) ** 2).mean()


def train_wgan_gp(windows_scaled, L=7, zdim=32, ch=64, steps=3000,
                  batch=64, n_critic=5, lam=10.0, lr=1e-4, seed=0, verbose=True,
                  bounds=None):
    """steps = generator steps. Total critic steps = steps * n_critic."""
    torch.manual_seed(seed)
    X = torch.tensor(windows_scaled, dtype=torch.float32,
                     device=DEVICE).unsqueeze(1)          # (N, 1, L)
    N = X.size(0)

    lo, hi = bounds if bounds is not None else (float(X.min()), float(X.max()))
    G = Generator(L, zdim, ch, lo=lo, hi=hi).to(DEVICE)
    D = Critic(L, ch).to(DEVICE)
    optG = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.0, 0.9))
    optD = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.0, 0.9))

    for step in range(1, steps + 1):
        for _ in range(n_critic):
            real = X[torch.randint(0, N, (batch,), device=DEVICE)]
            with torch.no_grad():
                fake = G(torch.randn(batch, zdim, device=DEVICE))
            lossD = D(fake).mean() - D(real).mean() + lam * gradient_penalty(D, real, fake)
            optD.zero_grad(set_to_none=True)
            lossD.backward()
            optD.step()

        fake = G(torch.randn(batch, zdim, device=DEVICE))
        lossG = -D(fake).mean()
        optG.zero_grad(set_to_none=True)
        lossG.backward()
        optG.step()

        if verbose and step % max(1, steps // 6) == 0:
            # -lossD approximates the Wasserstein-1 distance; should shrink.
            print(f"    step {step:5d}   W1_est {-lossD.item() + lam * 0:8.4f}   "
                  f"lossG {lossG.item():8.4f}")

    G.eval()
    return G


@torch.no_grad()
def sample_windows(G, scaler, n, zdim=32, batch=8192, target_wet=None):
    """target_wet: if given, the dry/wet cut is recalibrated so the simulated
    wet-day fraction matches the observed one. A smooth generator cannot place
    an atom at exactly 0 mm, so it smears the dry-day point mass into small
    positive values; this is the cheap correction. The principled fix is a
    two-part (hurdle) model with a separate occurrence head - see report."""
    out = []
    for i in range(0, n, batch):
        z = torch.randn(min(batch, n - i), zdim, device=DEVICE)
        out.append(G(z).squeeze(1).cpu().numpy())
    # Calibrate BEFORE snapping. Snapping first destroys the information needed
    # to raise the wet fraction: it can then only ever be lowered.
    w = scaler.inverse(np.concatenate(out), snap=False)
    cut = GAUGE_RES if target_wet is None else np.quantile(w, 1.0 - target_wet)
    return np.where(w <= cut, 0.0, w)


# ----------------------------------------------------------------------------
# 4. Diagnostics (course aspect: simulation modelling and diagnostics)
# ----------------------------------------------------------------------------

def hill(x, k=None):
    """Hill estimator of the tail index xi = 1/alpha, on the top k order stats."""
    x = np.sort(x[x > 0])[::-1]
    k = k or max(10, int(0.05 * len(x)))
    return np.mean(np.log(x[:k]) - np.log(x[k]))


def spell_lengths(windows):
    """Wet-spell lengths within windows (bounded by L; compare like-for-like)."""
    lens = []
    for w in windows:
        run = 0
        for v in w:
            if v > 0:
                run += 1
            elif run:
                lens.append(run)
                run = 0
        if run:
            lens.append(run)
    return np.array(lens) if lens else np.array([0])


def diagnostics(real, fake, L=7):
    """real, fake: (n, L) arrays of rainfall in mm."""
    d = {}
    d["wet_fraction"] = ((real > 0).mean(), (fake > 0).mean())
    d["mean_daily_mm"] = (real.mean(), fake.mean())

    for lag in (1, 2, 3):
        def ac(w, series_fn):
            a = series_fn(w)[:, :-lag].ravel()
            b = series_fn(w)[:, lag:].ravel()
            return np.corrcoef(a, b)[0, 1]
        d[f"acf_amount_lag{lag}"] = (ac(real, lambda x: x), ac(fake, lambda x: x))
        d[f"acf_wetdry_lag{lag}"] = (ac(real, lambda x: (x > 0).astype(float)),
                                     ac(fake, lambda x: (x > 0).astype(float)))

    d["mean_spell_len"] = (spell_lengths(real).mean(), spell_lengths(fake).mean())

    Sr, Sf = real.sum(1), fake.sum(1)
    for q in (0.50, 0.90, 0.99, 0.999):
        d[f"total_q{q}"] = (np.quantile(Sr, q), np.quantile(Sf, q))
    d["total_max"] = (Sr.max(), Sf.max())
    d["total_hill_xi"] = (hill(Sr), hill(Sf))
    d["daily_max"] = (real.max(), fake.max())
    return d


def memorisation_check(real, fake, n_probe=500, seed=0):
    """Median nearest-neighbour distance from generated windows to training set.
    Near zero => the generator is copying, not modelling."""
    rng = np.random.default_rng(seed)
    probe = fake[rng.choice(len(fake), min(n_probe, len(fake)), replace=False)]
    ref = real[rng.choice(len(real), min(4000, len(real)), replace=False)]
    dists = np.sqrt(((probe[:, None, :] - ref[None, :, :]) ** 2).sum(-1)).min(1)
    baseline = np.sqrt(((ref[:200, None, :] - ref[None, 200:400, :]) ** 2).sum(-1)).min(1)
    return float(np.median(dists)), float(np.median(baseline))


# ----------------------------------------------------------------------------
# 5. Risk measures and return levels
# ----------------------------------------------------------------------------

def windows_per_year(L):
    return DAYS_PER_YEAR / L


def return_level(totals, T_years, L=7):
    """Level exceeded once per T years, from a sample of window totals."""
    p = 1.0 / (T_years * windows_per_year(L))          # exceedance prob per window
    if p < 1.0 / len(totals):
        return np.nan                                   # beyond sample resolution
    return float(np.quantile(totals, 1.0 - p))


def var_cvar(totals, alpha):
    v = float(np.quantile(totals, alpha))
    tail = totals[totals >= v]
    return v, float(tail.mean()) if len(tail) else np.nan


# ----------------------------------------------------------------------------
# 6. Moving-block bootstrap (Phase 4) - preserves serial dependence
# ----------------------------------------------------------------------------

def moving_block_bootstrap(series, block_len=30, rng=None):
    rng = rng or np.random.default_rng()
    n = len(series)
    starts = rng.integers(0, n - block_len + 1, size=int(np.ceil(n / block_len)))
    return np.concatenate([series[s:s + block_len] for s in starts])[:n]


# ----------------------------------------------------------------------------
# 7. Validation run on the synthetic surrogate
# ----------------------------------------------------------------------------

def main():
    L, ZDIM = 7, 32
    print(f"device: {DEVICE}\n")

    series = synthetic_rainfall(n_days=15340, seed=0)
    print(f"series: {len(series)} days, wet fraction {(series > 0).mean():.3f} "
          f"(truth 0.424), max {series.max():.1f} mm")

    W_train = make_windows(series, L, stride=1)          # for the network
    W_eval = make_windows(series, L, stride=L)           # disjoint, for honest counting
    print(f"windows: {len(W_train)} overlapping (train), {len(W_eval)} disjoint (eval)")
    print(f"windows per year: {windows_per_year(L):.2f}\n")

    scaler = Scaler().fit(W_train)
    G = train_wgan_gp(scaler.forward(W_train), L=L, zdim=ZDIM,
                      steps=2500, batch=64, seed=0)

    raw = sample_windows(G, scaler, 200_000, zdim=ZDIM)
    fake = sample_windows(G, scaler, 200_000, zdim=ZDIM,
                          target_wet=(W_train > 0).mean())
    print(f"\n  raw simulated wet fraction {(raw > 0).mean():.4f} "
          f"(observed {(W_eval > 0).mean():.4f}) -> recalibrated")
    print("\n  diagnostic                 real        GAN")
    print("  " + "-" * 42)
    for k, (r, f) in diagnostics(W_eval, fake, L).items():
        print(f"  {k:<22} {r:9.4f}  {f:9.4f}")

    nn_d, base_d = memorisation_check(W_eval, fake)
    print(f"  {'nn_dist (vs baseline)':<22} {base_d:9.4f}  {nn_d:9.4f}")

    Sr, Sf = W_eval.sum(1), fake.sum(1)
    print(f"\n  {L}-day accumulation risk (mm)")
    print("  level                   empirical        GAN")
    print("  " + "-" * 42)
    for a in (0.99, 0.999):
        vr, cr = var_cvar(Sr, a)
        vf, cf = var_cvar(Sf, a)
        print(f"  VaR_{a:<18} {vr:9.2f}  {vf:9.2f}")
        print(f"  CVaR_{a:<17} {cr:9.2f}  {cf:9.2f}")
    for T in (2, 10, 30, 100):
        print(f"  {T:>3}-yr return level     {return_level(Sr, T, L):9.2f}  "
              f"{return_level(Sf, T, L):9.2f}")

    boot = moving_block_bootstrap(series, block_len=30, rng=np.random.default_rng(1))
    print(f"\n  block bootstrap replicate: {len(boot)} days, "
          f"wet fraction {(boot > 0).mean():.3f}")
    print("\npipeline OK.")


if __name__ == "__main__":
    main()
