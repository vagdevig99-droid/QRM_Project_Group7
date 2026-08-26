"""
M1 - POT/GPD directly on window totals (classical benchmark).

This model is described in the project report (Section 6, "M1 - POT/GPD on
window totals") and appears in the report's return-level table, but it is not
actually implemented as a class in the uploaded models.py (which only defines
M2-M7). It is added here, in the same style as the other model classes, so
that M1 can be included in the bootstrap alongside M2-M7.

No changes were made to marginal.py, models.py, or rainfall_wgan.py to add
this - it is purely additive, built out of the same fit_gpd() used everywhere
else in the project.
"""

import numpy as np

from models import BaseModel, _windows
from marginal import fit_gpd


class POT_GPD(BaseModel):
    """Fit a GPD directly to exceedances of the L-day total over its 95th
    percentile (disjoint windows, honest counting - same convention SpliceGAN
    uses for its own threshold). No simulate() - this model characterises the
    tail analytically, so VaR/CVaR/return levels are computed in closed form
    from the fitted (u0, xi, sigma), exactly mirroring the worked example in
    the report (Section 6, M1)."""

    name = "M1 POT/GPD"

    def fit(self, series):
        S = _windows(series, self.L, self.L).sum(1)
        self.S_obs = S
        self.u0 = float(np.quantile(S, 0.95))
        exc = S[S > self.u0] - self.u0
        self.xi, self.sigma = fit_gpd(exc, x0=(0.1, 10.0))
        self.zeta0 = float((S > self.u0).mean())
        self.n_exc = len(exc)
        return self

    def _gpd_level(self, p):
        """Level exceeded with per-window probability p < zeta0."""
        return float(self.u0 + self.sigma / self.xi *
                     ((p / self.zeta0) ** (-self.xi) - 1.0))

    def return_level(self, T):
        p = 1.0 / (T * self.windows_per_year)
        if p >= self.zeta0:
            return float(np.quantile(self.S_obs, 1.0 - p))
        return self._gpd_level(p)

    def var(self, alpha=0.99):
        p = 1.0 - alpha
        if p >= self.zeta0:
            return float(np.quantile(self.S_obs, alpha))
        return self._gpd_level(p)

    def cvar(self, alpha=0.99):
        v = self.var(alpha)
        if v <= self.u0:
            tail = self.S_obs[self.S_obs >= v]
            return float(tail.mean()) if len(tail) else np.nan
        if self.xi >= 1.0:
            return np.nan
        # Threshold-stability: excess over v (v > u0) is GPD(xi, sigma + xi*(v-u0));
        # its mean is sigma_v / (1 - xi). E[S | S > v] = v + mean excess.
        sigma_v = self.sigma + self.xi * (v - self.u0)
        return float(v + sigma_v / (1.0 - self.xi))

    def __repr__(self):
        return (f"POT_GPD(u0={self.u0:.2f}mm, n_exc={self.n_exc}, "
                f"xi={self.xi:.4f}, sigma={self.sigma:.3f})")
