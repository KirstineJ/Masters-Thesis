# =====================================================
# Imports
# =====================================================

import numpy as np
from scipy import stats
from scipy.integrate import quad, fixed_quad
from scipy.interpolate import interp1d

# =====================================================
# Definitions
# =====================================================

L = 38
n = 400

# =====================================================
# CDF and PDF
# =====================================================

_gumbel = stats.gumbel_r(0, 1)
gumbel_pdf = _gumbel.pdf
gumbel_cdf = _gumbel.cdf

def F_eps(z):
    return gumbel_cdf(z)

def f_eps(z):
    return gumbel_pdf(z)

def F_u(z, delta_j):
    return F_eps(z - delta_j)

def f_u(z, delta_j):
    return f_eps(z - delta_j)

def F_w(z, delta_k, mu_k):
    lambda_k = delta_k - mu_k
    return np.exp(-np.exp(-(z - lambda_k)))

def f_w(z, delta_j, mu_j):
    lambda_j = delta_j - mu_j
    return f_eps(z - lambda_j)

def F_r(z, delta_j, mu_j):
    z = np.asarray(z, dtype=float)

    t1 = z - delta_j
    t2 = z - delta_j + mu_j

    S1 = -np.expm1(-np.exp(-t1))
    S2 = -np.expm1(-np.exp(-t2))

    return 1.0 - S2 / S1


def f_r_cont(z, delta_j, mu_j):
    z = np.asarray(z, dtype=float)

    t1 = z - delta_j
    t2 = z - delta_j + mu_j

    S1 = -np.expm1(-np.exp(-t1))
    S2 = -np.expm1(-np.exp(-t2))

    f1 = f_eps(t1)
    f2 = f_eps(t2)

    numerator = f2 * S1 - S2 * f1
    denominator = S1 ** 2

    return numerator / denominator

# =====================================================
# H0 and Inverse
# =====================================================

def H0(z):
    integrand = lambda e: np.fmax(0.0, e - z) * f_eps(e)
    return quad(integrand, -np.inf, np.inf)[0]

def H0_vec(z):
    N = len(z)
    h = np.zeros(N)
    for i in range(N):
        h[i] = H0(z[i])
    return h

def build_H0_inverse(z_min=-10, z_max=20, n_grid=500):
    z_grid = np.linspace(z_min, z_max, n_grid)
    H0_vals = H0_vec(z_grid)
    return interp1d(H0_vals, z_grid, fill_value="extrapolate")


H0_inv = build_H0_inverse()


def H0_inv_vec(c):
    return np.asarray(H0_inv(c))

# =====================================================
# Search cost distribution
# =====================================================

def Fc(c, mu):
    assert mu >= 0.0, "mu must be non-negative"

    if c < 0.0:
        return 0.0
    elif mu == 0.0:
        return 1.0 
    else:
        H_inv_c = H0_inv(c)
        numer = 1.0 - F_eps(H_inv_c + mu)
        denom = 1.0 - F_eps(H_inv_c)
        return numer / denom

def get_fast_Fc_inv(mu, cmin, cmax, ngrid=1000):
    c_grid = np.linspace(cmin, cmax, ngrid)
    p_grid = np.array([Fc(c, mu) for c in c_grid])
    Fc_inv_interp = interp1d(p_grid, c_grid, fill_value="extrapolate")

    def fast_Fc_inv(p_values):
        N = len(p_values)
        c = np.zeros(N)
        p_cutoff = Fc(0.0, mu) 
        I = (p_values <= p_cutoff)
        c[I] = 0.0  
        c[~I] = Fc_inv_interp(p_values[~I])
        return c

    return fast_Fc_inv

# =====================================================
# Model-implied search probabilities
# =====================================================

def psi_integrand(z, j, delta, mu):
    outside = F_eps(z)
    prod = 1.0

    for k in range(1, len(delta)): 
        if k != j:
            prod *= F_w(z, delta[k], mu[k])

    return outside * prod * f_r_cont(z, delta[j], mu[j])


def psi_numeric_one_product_quad(j, delta, mu):
    atom = np.exp(-mu[j])
    integral_value, _ = fixed_quad(lambda z: psi_integrand(z, j, delta, mu), -L, L, n=n)
    psi = atom + integral_value

    return psi

def psi_numeric_one_product_quad_benchmark(j, delta, mu):
    atom = np.exp(-mu[j])
    integral_value, _ = quad(lambda z: psi_integrand(z, j, delta, mu), -L, L)
    psi = atom + integral_value

    return psi

def pi0_integrand(z, delta, mu):
    prod = 1.0

    for j in range(1, len(delta)): 
        prod *= F_r(z, delta[j], mu[j])

    return prod * f_eps(z)

def prob_zero_searches_quad(delta, mu):
    pi0_quad, _ = fixed_quad(lambda z: pi0_integrand(z, delta, mu), -L, L, n=n)

    return pi0_quad

def F_minus_j_r(z, j, delta, mu):
    prod = 1.0

    for k in range(1, len(delta)): 
        if k != j:
            prod *= F_r(z, delta[k], mu[k])

    return prod

def pi1_integrand_term1(z, j, delta, mu):
    return ((F_w(z, delta[j], mu[j]) - F_r(z, delta[j], mu[j]))* F_minus_j_r(z, j, delta, mu)* f_eps(z))

def pi1_integrand_term2(z, j, delta, mu):
    return (F_minus_j_r(z, j, delta, mu)* F_eps(z)* f_w(z, delta[j], mu[j]))

def prob_one_search_quad(delta, mu):
    total_quad = 0.0

    for j in range(1, len(delta)):  
        term1_quad, _ = fixed_quad(lambda z: pi1_integrand_term1(z, j, delta, mu), -L, L, n=n)
        term2_quad, _ = fixed_quad(lambda z: pi1_integrand_term2(z, j, delta, mu), -L, L, n=n)

        total_quad += term1_quad + term2_quad

    return total_quad

# =====================================================
# Expected number of searches
# =====================================================

def expected_number_searches_quad(psi):
    return np.sum(psi)

# =====================================================
# Choice probability
# =====================================================

def choice_probs_theory(deltas, mus):
    num = np.exp(deltas - mus)
    return num / num.sum()
