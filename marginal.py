"""
Semi-parametric marginal for daily rainfall, and the transform to normal scores.

The marginal has three pieces:
    P(X = 0) = p_dry                              atom at zero
    empirical CDF                                 wet amounts up to threshold u
    GPD                                           wet amounts above u   <- extrapolates

This is the object that lets a dependence model (GAN, copula, Markov chain) stop
worrying about the tail: the model produces normal scores, and the inverse
transform maps them to millimetres through a marginal that already extrapolates.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

GAUGE_RES = 0.1


def fit_gpd(excesses, x0=(0.1, 5.0)):
    """MLE for GPD(xi, sigma) on exceedances y = x - u > 0."""
    def nll(par):
        xi, sig = par
        if sig <= 0:
            return 1e10
        z = 1.0 + xi * excesses / sig
        if np.any(z <= 0):
            return 1e10
        if abs(xi) < 1e-8:
            return len(excesses) * np.log(sig) + excesses.sum() / sig
        return len(excesses) * np.log(sig) + (1.0 + 1.0 / xi) * np.log(z).sum()
    res = minimize(nll, x0, method="Nelder-Mead",
                   options={"xatol": 1e-9, "fatol": 1e-9, "maxiter": 8000})
    return float(res.x[0]), float(res.x[1])


class SemiParametricMarginal:
    """Zero atom + empirical body + GPD tail. Supports cdf, ppf, and both
    directions of the normal-score transform."""

    def fit(self, series, tail_q=0.95, seed=0):
        r = np.asarray(series, dtype=float)
        self.p_dry = float((r <= 0).mean())
        wet = np.sort(r[r > 0])

        # Rain gauges record to 0.1 mm, so the data is full of exact ties.
        # Jitter by half the gauge resolution: physically the true value really
        # is somewhere in that interval, and it makes the CDF continuous so the
        # probability integral transform gives exact uniforms.
        rng = np.random.default_rng(seed)
        self._jitter_scale = GAUGE_RES / 2.0
        self.wet = np.sort(wet + rng.uniform(-self._jitter_scale,
                                             self._jitter_scale, len(wet)))
        self.n_wet = len(self.wet)

        self.u = float(np.quantile(self.wet, tail_q))
        exc = self.wet[self.wet > self.u] - self.u
        self.xi, self.sigma = fit_gpd(exc)
        self.zeta_w = len(exc) / self.n_wet        # P(X > u | wet)
        self.n_exc = len(exc)
        return self

    # ---- CDF / PPF on the wet-day distribution -----------------------------

    def _cdf_wet(self, x):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        out = np.searchsorted(self.wet, x, side="right") / self.n_wet
        hi = x > self.u
        if np.any(hi):
            z = 1.0 + self.xi * (x[hi] - self.u) / self.sigma
            out[hi] = 1.0 - self.zeta_w * np.power(np.maximum(z, 1e-12),
                                                   -1.0 / self.xi)
        return np.clip(out, 0.0, 1.0 - 1e-12)

    def _ppf_wet(self, pw):
        pw = np.atleast_1d(np.asarray(pw, dtype=float))
        out = np.empty_like(pw)
        lo = pw <= 1.0 - self.zeta_w
        if np.any(lo):
            out[lo] = np.interp(pw[lo], np.arange(1, self.n_wet + 1) / self.n_wet,
                                self.wet)
        hi = ~lo
        if np.any(hi):
            # GPD quantile. Grows only polynomially, so even a large input
            # cannot explode the way expm1() did under the log transform.
            out[hi] = self.u + self.sigma / self.xi * (
                np.power((1.0 - pw[hi]) / self.zeta_w, -self.xi) - 1.0)
        return out

    # ---- full marginal including the zero atom -----------------------------

    def ppf(self, p):
        """p in (0,1) -> rainfall in mm. Values of p below p_dry give 0."""
        p = np.atleast_1d(np.asarray(p, dtype=float))
        out = np.zeros_like(p)
        wet = p > self.p_dry
        if np.any(wet):
            pw = (p[wet] - self.p_dry) / (1.0 - self.p_dry)
            out[wet] = np.maximum(self._ppf_wet(np.clip(pw, 0, 1 - 1e-12)),
                                  GAUGE_RES)
        return out

    # ---- normal-score transform -------------------------------------------

    def to_normal(self, series, seed=0):
        """Randomised PIT then Phi^{-1}. Zeros are spread uniformly across
        (0, p_dry), which is what makes the transformed variable exactly
        standard normal despite the atom."""
        rng = np.random.default_rng(seed)
        r = np.asarray(series, dtype=float)
        u = np.empty(len(r))
        dry = r <= 0
        u[dry] = rng.uniform(0.0, self.p_dry, dry.sum())
        if np.any(~dry):
            xw = r[~dry] + rng.uniform(-self._jitter_scale,
                                       self._jitter_scale, (~dry).sum())
            u[~dry] = self.p_dry + (1.0 - self.p_dry) * self._cdf_wet(xw)
        return norm.ppf(np.clip(u, 1e-9, 1 - 1e-9))

    def from_normal(self, z):
        return self.ppf(norm.cdf(np.asarray(z, dtype=float)))

    def __repr__(self):
        return (f"Marginal(p_dry={self.p_dry:.4f}, u={self.u:.2f}mm, "
                f"n_exc={self.n_exc}, xi={self.xi:.4f}, sigma={self.sigma:.3f})")
