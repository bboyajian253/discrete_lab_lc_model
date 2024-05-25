# lifecycle_labor: model
# contains basic model functions, e.g., utility functions, budget constraints, or earnings technologies

import numpy as np
from numba import njit
from interpolation.splines import eval_linear  # see, https://github.com/EconForge/interpolation.py
from interpolation import interp
import toolbox


@njit
def earn(wh, n, par):
    """
    Earnings given wage rate w, human capital h, labor n.
    """
    return wh * n


@njit
def dearn_dwork(wh, n):
    """
    Marginal change in earnings due to marginal change in hours.
    """
    return wh


@njit
def aftertaxearn(e, par):
    """
    average tax rate given pre-tax earnings e.
    """
    return par.ybar * (par.τ0 * (e / par.ybar) ** (1 - par.τ1))


@njit
def daftertaxearn_dearn(e, par, mine=1e-5):
    """
    marginal tax rate given pre-tax earnings e.
    """
    e = np.where(e > mine, e, mine)
    return par.τ0 * (1 - par.τ1) * (e / par.ybar) ** (-par.τ1)


@njit
def u(c, n, par):
    """
    Utility function over consumption, c, and labor, n.
    """
    u_c = (c ** (1 - par.ω) - 1) / (1 - par.ω)
    u_n = - (par.ψ / (1 + 1 / par.γ)) * (n ** (1 + 1 / par.γ))
    return u_c + u_n


@njit
def du_dc(c, par):
    """
    Derivative of utility function wrt consumption.
    """
    return c ** (-par.ω)


@njit
def invert_c(mb_kk, par):
    """
    Returns the value of c that satisfies Euler equation for consumption given mb_kk.
    """
    return max(par.minc, (par.β * mb_kk) ** (-1 / par.ω))


@njit
def du_dn(n, ψ, par):
    """
    Derivative of utility function wrt non-leisure time.
    """
    return ψ * n ** (1 / par.γ)


@njit
def invert_n(c, wh, minn, maxn, par, I=13, tol=1e-4):
    """
    Returns the value of n that generates marginal benefit mb_n.
    focn = mb(n) - mc(n)
    mb(n) = du_dc * daftertaxearn_dearn * dearn_dn
    """

    # if linear tax, have closed form solution for labor
    if np.abs(par.τ1) < tol:
        mb_n = (du_dc(c, par) / (1 + par.τc)) * par.τ0 * wh
        return max(minn, min(maxn, (mb_n / par.ψ) ** par.γ))

    # if non-linear tax, need to iterate to solve for labor
    focn = lambda n: (du_dc(c, par) / (1 + par.τc)) \
                     * daftertaxearn_dearn(earn(wh, n, par), par) \
                     * dearn_dwork(wh, n) - par.ψ * n ** (1 / par.γ)

    nlo, nhi = minn, maxn
    if focn(nlo) <= 0:
        return nlo
    if focn(nhi) >= 0:
        return nhi
    iter, err = 1, tol + 1
    while iter < I and np.abs(err) > tol:
        n = (nlo + nhi) / 2
        err = focn(n)
        nlo = nlo * (err < -tol) + n * (err >= -tol)
        nhi = n * (err <= tol) + nhi * (err > tol)
        iter += 1

    return n


@njit
def ValueFn(c, n, hh, mVV_flat, par):
    """
    Compute value function.
    """
    VV = interp(par.gridh, mVV_flat, hh)
    return u(c, n, par) + par.β * VV


@njit
def income(rk, e):
    """
    Computes income given capital k, net interest rate r, and earnings e.
    """
    return max(0.0, rk) + e


@njit
def infer_k(c, kk, e, r, par):
    """
    Infers what existing capital stock k balances the individual's budget.
    """
    return ((1 + par.τc) * c + kk - aftertaxearn(e, par) - par.T) / (1 + r)


@njit
def solve_n_k_yt(kk, c, wh, r, nlo, nhi, par):
    """
    Solves for labor, current capital, and income given consumption c, and savings kk.
    """
    n = invert_n(c, wh, nlo, nhi, par)
    e = earn(wh, n, par)
    k = infer_k(c, kk, e, r, par)

    return n, k


@njit
def consumption(kk, r, k, e, par):
    """
    Computes consumption given savings kk, capital k, interest rate r,
    ... earnings e.
    """
    return max(par.minc, (aftertaxearn(e, par) + par.T + k * (1 + r) - kk) / (1 + par.τc))


@njit
def foc_n(n, k, kk, r, wh, par):
    """
    Labor FOC.
    """
    e = earn(wh, n, par)
    c = consumption(kk, r, k, e, par)
    focL = (du_dc(c, par) / (1 + par.τc)) \
           * daftertaxearn_dearn(earn(wh, n, par), par) \
           * dearn_dwork(wh, n)  # MB
    focR = du_dn(n, par)   # MC
    return focL - focR, c  # MB - MC


@njit
def human_capital(j, na, par):
    """
    Human capital given ability type na and age j.
    """
    h = toolbox.quadratic(j, par.grida[na])
    return np.where(h > par.minc, h, par.minc)
