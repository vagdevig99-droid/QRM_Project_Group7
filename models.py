"""
Competing models for 7-day rainfall accumulation risk.

Common interface:
    .fit(daily_series)           -> self
    .simulate(n)                 -> (n, L) array of daily rainfall in mm
    .return_level(T)             -> T-year return level of the L-day total

Model set:
    M2  RawGAN              log1p-scaled WGAN-GP (the current model)
    M3  SpliceGAN           M2 body + GPD tail on the window totals
    M4  NormalScoreGAN      GAN learns the copula only; EVT owns the marginal
    M5  MarkovWeatherGen    order-k Markov occurrence + semi-parametric amounts
    M6  CopulaWindow        Gaussian or t copula with the same EVT marginal
    M7  GEVAnnualMax        GEV on 42 annual maxima (classical reference)

M5 and M6 reproduce the autocorrelation BY CONSTRUCTION. That is the point of
including them: they are the bar the GAN has to clear. A correlation matrix
captures all LINEAR dependence, so the GAN's only possible edge is non-elliptical
structure - heavy days clustering more than light days do.
"""

import numpy as np
from scipy.stats import norm, t as tdist, chi2, multivariate_t
from scipy.linalg import toeplitz
from scipy.optimize import minimize

import rainfall_wgan as rw
from marginal import SemiParametricMarginal, fit_gpd, GAUGE_RES

DAYS_PER_YEAR = 365.25
NSIM = 500_000


def _windows(series, L, stride=1):
    n = (len(series) - L) // stride + 1
    idx = np.arange(L)[None, :] + (np.arange(n) * stride)[:, None]
    return series[idx]


def _nearest_pd(S, eps=1e-8):
    w, V = np.linalg.eigh((S + S.T) / 2)
    w = np.maximum(w, eps)
    S = V @ np.diag(w) @ V.T
    d = np.sqrt(np.diag(S))
    return S / np.outer(d, d)


class BaseModel:
    name = "base"

    def __init__(self, L=7):
        self.L = L
        self._cache = None

    @property
    def windows_per_year(self):
        return DAYS_PER_YEAR / self.L

    def _sim_totals(self, n=NSIM):
        if self._cache is None or len(self._cache) < n:
            self._cache = self.simulate(n).sum(1)
        return self._cache

    def return_level(self, T):
        p = 1.0 / (T * self.windows_per_year)
        tot = self._sim_totals()
        if p < 5.0 / len(tot):
            return np.nan
        return float(np.quantile(tot, 1.0 - p))


# ---------------------------------------------------------------------------
# M2 / M3 : the log1p GAN, and its GPD splice
# ---------------------------------------------------------------------------

class RawGAN(BaseModel):
    name = "M2 raw GAN"

    def __init__(self, L=7, steps=4000, zdim=32, ch=64, seed=0):
        super().__init__(L)
        self.steps, self.zdim, self.ch, self.seed = steps, zdim, ch, seed

    def fit(self, series):
        W = _windows(series, self.L, 1)
        self.target_wet = float((W > 0).mean())
        self.sc = rw.Scaler().fit(W)
        self.G = rw.train_wgan_gp(self.sc.forward(W), L=self.L, zdim=self.zdim,
                                  ch=self.ch, steps=self.steps, seed=self.seed,
                                  verbose=False)
        return self

    def simulate(self, n):
        return rw.sample_windows(self.G, self.sc, n, zdim=self.zdim,
                                 target_wet=self.target_wet)


class SpliceGAN(BaseModel):
    """GAN below the splice point, GPD above it, on the distribution of TOTALS.

    Uses the threshold-stability property: exceedances above u1 > u0 are GPD with
    the SAME xi and sigma1 = sigma0 + xi*(u1 - u0). This lets a fit made where
    there is data (u0 = q95) be moved to where there is not (u1 = the 10-yr level).
    """
    name = "M3 GAN+GPD splice"

    def __init__(self, gan, splice_T=10.0):
        super().__init__(gan.L)
        self.gan, self.splice_T = gan, splice_T

    def fit(self, series):
        S = _windows(series, self.L, self.L).sum(1)      # disjoint, honest counting
        self.u0 = float(np.quantile(S, 0.95))
        exc = S[S > self.u0] - self.u0
        self.xi, self.sig0 = fit_gpd(exc, x0=(0.1, 10.0))
        self.zeta0 = float((S > self.u0).mean())
        self.ustar = self.gan.return_level(self.splice_T)
        self.zeta_star = 1.0 / (self.splice_T * self.windows_per_year)
        self.sig_star = self.sig0 + self.xi * (self.ustar - self.u0)
        return self

    def simulate(self, n):
        return self.gan.simulate(n)

    def return_level(self, T):
        p = 1.0 / (T * self.windows_per_year)
        if p >= self.zeta_star:
            return self.gan.return_level(T)              # body: trust the GAN
        return float(self.ustar + self.sig_star / self.xi *
                     ((p / self.zeta_star) ** (-self.xi) - 1.0))


# ---------------------------------------------------------------------------
# M4 : normal-score GAN  (GAN learns the copula, EVT owns the marginal)
# ---------------------------------------------------------------------------

class NormalScoreGAN(BaseModel):
    name = "M4 normal-score GAN"

    def __init__(self, L=7, steps=4000, zdim=32, ch=64, seed=0,
                 tail_q=0.95, zbound=5.0):
        super().__init__(L)
        self.steps, self.zdim, self.ch, self.seed = steps, zdim, ch, seed
        self.tail_q, self.zbound = tail_q, zbound

    def fit(self, series):
        self.marg = SemiParametricMarginal().fit(series, tail_q=self.tail_q,
                                                 seed=self.seed)
        z = self.marg.to_normal(series, seed=self.seed)
        Z = _windows(z, self.L, 1)
        # Bound at +/- zbound, NOT at the data range. z = 5 is a 1-in-9500-year
        # day, so the ceiling is far outside anything we will report - the
        # truncation that crippled M2 is no longer binding.
        self.G = rw.train_wgan_gp(Z, L=self.L, zdim=self.zdim, ch=self.ch,
                                  steps=self.steps, seed=self.seed, verbose=False,
                                  bounds=(-self.zbound, self.zbound))
        return self

    def simulate(self, n):
        import torch
        out = []
        with torch.no_grad():
            for i in range(0, n, 8192):
                zz = torch.randn(min(8192, n - i), self.zdim, device=rw.DEVICE)
                out.append(self.G(zz).squeeze(1).cpu().numpy())
        Z = np.concatenate(out)
        return self.marg.from_normal(Z.ravel()).reshape(Z.shape)


# ---------------------------------------------------------------------------
# M5 : Markov weather generator (occurrence chain + semi-parametric amounts)
# ---------------------------------------------------------------------------

class MarkovWeatherGen(BaseModel):
    """Classical hydrology benchmark. Clustering is explicit in the transition
    matrix; the tail is unbounded because amounts come from a GPD-tailed
    marginal. Order selected by BIC."""
    name = "M5 Markov weather gen"

    def __init__(self, L=7, max_order=4, tail_q=0.95, seed=0):
        super().__init__(L)
        self.max_order, self.tail_q, self.seed = max_order, tail_q, seed

    def _fit_chain(self, wet, k):
        n = len(wet)
        idx = np.zeros(n - k, dtype=int)
        for j in range(k):
            idx += wet[j:n - k + j].astype(int) << (k - 1 - j)
        nxt = wet[k:].astype(int)
        cnt = np.zeros((2 ** k, 2))
        np.add.at(cnt, (idx, nxt), 1.0)
        P = (cnt + 0.5) / (cnt.sum(1, keepdims=True) + 1.0)   # Jeffreys smoothing
        ll = float((cnt * np.log(P)).sum())
        bic = -2 * ll + (2 ** k) * np.log(n - k)
        return P, ll, bic

    def fit(self, series):
        wet = np.asarray(series) > 0
        scores = {}
        for k in range(1, self.max_order + 1):
            P, ll, bic = self._fit_chain(wet, k)
            scores[k] = (P, ll, bic)
        self.order = min(scores, key=lambda k: scores[k][2])
        self.P = scores[self.order][0]
        self.bic_table = {k: scores[k][2] for k in scores}

        # Amounts, split by whether the previous day was wet. acf_amount(1) is
        # ~0.17, so intensity persists too, not just occurrence.
        r = np.asarray(series, dtype=float)
        prev_wet = np.concatenate([[False], wet[:-1]])
        self.m_ww = SemiParametricMarginal().fit(
            r[wet & prev_wet], tail_q=self.tail_q, seed=self.seed)
        self.m_dw = SemiParametricMarginal().fit(
            r[wet & ~prev_wet], tail_q=self.tail_q, seed=self.seed)
        self.p_wet = float(wet.mean())
        return self

    def simulate(self, n, burn=30):
        rng = np.random.default_rng(self.seed + 1)
        k, T = self.order, self.L + burn
        wet = np.zeros((n, T), dtype=bool)
        wet[:, :k] = rng.random((n, k)) < self.p_wet
        for t in range(k, T):
            idx = np.zeros(n, dtype=int)
            for j in range(k):
                idx += wet[:, t - k + j].astype(int) << (k - 1 - j)
            wet[:, t] = rng.random(n) < self.P[idx, 1]
        wet = wet[:, burn:]

        prev = np.concatenate([np.zeros((n, 1), bool), wet[:, :-1]], axis=1)
        out = np.zeros((n, self.L))
        for mask, marg in ((wet & prev, self.m_ww), (wet & ~prev, self.m_dw)):
            k_ = int(mask.sum())
            if k_:
                # marginal is conditional on wet, so draw p above the dry atom
                p = marg.p_dry + (1 - marg.p_dry) * rng.random(k_)
                out[mask] = marg.ppf(p)
        return out


# ---------------------------------------------------------------------------
# M6 : elliptical copula on the window, with the same EVT marginal
# ---------------------------------------------------------------------------

class CopulaWindow(BaseModel):
    """Gaussian (df=None) or Student-t copula with a Toeplitz correlation matrix
    estimated from the normal scores. Stationarity implies Toeplitz.

    The t copula has upper tail dependence; the Gaussian has none. Comparing the
    two tests directly whether London's extreme days co-occur more than linear
    correlation alone would imply."""
    name = "M6 copula"

    def __init__(self, L=7, df=None, tail_q=0.95, seed=0, select_df=True):
        super().__init__(L)
        self.df, self.tail_q, self.seed, self.select_df = df, tail_q, seed, select_df

    def _copula_ll(self, U, nu):
        T = tdist.ppf(np.clip(U, 1e-9, 1 - 1e-9), nu)
        joint = multivariate_t.logpdf(T, shape=self.R, df=nu,
                                      loc=np.zeros(self.L))
        marg = tdist.logpdf(T, nu).sum(1)
        return float((joint - marg).sum())

    def fit(self, series):
        self.marg = SemiParametricMarginal().fit(series, tail_q=self.tail_q,
                                                 seed=self.seed)
        z = self.marg.to_normal(series, seed=self.seed)
        rho = [1.0] + [float(np.corrcoef(z[:-k], z[k:])[0, 1])
                       for k in range(1, self.L)]
        self.R = _nearest_pd(toeplitz(rho))
        self.rho = rho

        if self.select_df:
            U = norm.cdf(_windows(z, self.L, self.L))     # disjoint windows
            grid = [3, 4, 5, 7, 10, 15, 25, 50]
            self.ll_table = {nu: self._copula_ll(U, nu) for nu in grid}
            best = max(self.ll_table, key=self.ll_table.get)
            # Gaussian is the nu -> inf limit; treat large nu as Gaussian.
            self.df = None if best >= 50 else best
        self.chol = np.linalg.cholesky(self.R)
        return self

    def simulate(self, n):
        rng = np.random.default_rng(self.seed + 2)
        Y = rng.standard_normal((n, self.L)) @ self.chol.T
        if self.df is None:
            U = norm.cdf(Y)
        else:
            W = rng.chisquare(self.df, size=(n, 1))
            U = tdist.cdf(Y * np.sqrt(self.df / W), self.df)
        return self.marg.ppf(U.ravel()).reshape(U.shape)


# ---------------------------------------------------------------------------
# M7 : GEV on annual maxima (classical reference, no simulation)
# ---------------------------------------------------------------------------

class GEVAnnualMax(BaseModel):
    name = "M7 GEV annual max"

    def __init__(self, L=7, start_year=1979):
        super().__init__(L)
        self.start_year = start_year

    def fit(self, series, years=None):
        roll = np.convolve(np.asarray(series, float), np.ones(self.L), "valid")
        if years is None:
            years = self.start_year + np.arange(len(roll)) // int(DAYS_PER_YEAR)
        self.maxima = np.array([roll[years == y].max() for y in np.unique(years)])

        def nll(par):
            mu, sig, xi = par
            if sig <= 0:
                return 1e10
            z = 1 + xi * (self.maxima - mu) / sig
            if np.any(z <= 0):
                return 1e10
            return (len(self.maxima) * np.log(sig)
                    + (1 + 1 / xi) * np.log(z).sum() + (z ** (-1 / xi)).sum())
        res = minimize(nll, [self.maxima.mean(), self.maxima.std(), 0.1],
                       method="Nelder-Mead", options={"maxiter": 8000})
        self.mu, self.sigma, self.xi = res.x
        return self

    def simulate(self, n):
        raise NotImplementedError("GEV is fitted to maxima, not simulated daily")

    def return_level(self, T):
        # annual maxima: P(annual max <= x) = 1 - 1/T
        y = -np.log(1 - 1 / T)
        return float(self.mu + self.sigma / self.xi * (y ** (-self.xi) - 1))
