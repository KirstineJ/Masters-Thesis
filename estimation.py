# =====================================================
# Imports
# =====================================================

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from joblib import Parallel, delayed
import search_model  as sm
import simulation as sim

# =====================================================
# Theoretical moments
# =====================================================

def compute_deltas(theta, data):
    beta_const, beta_price, beta_star = theta[:3]
    delta = beta_const + beta_price * data["price"].to_numpy() + beta_star * data["star"].to_numpy()
    delta = np.array(delta, dtype=float)
    delta[0] = 0.0
    return delta

def compute_mus(theta, data):
    lambda_const, lambda_pos = theta[3:]
    mu = np.log(1 + np.exp(lambda_const + lambda_pos * data["position"].to_numpy()))
    mu = np.array(mu, dtype=float)
    mu[0] = 0.0
    return mu

def compute_market_theory(theta, market_obj):
    info_t = market_obj["info"].sort_values("product_id")

    delta = compute_deltas(theta, info_t)
    mu = compute_mus(theta, info_t)

    choice_model = sm.choice_probs_theory(delta, mu)

    psi_model = np.array([
        sm.psi_numeric_one_product_quad(j, delta, mu)
        for j in range(1, len(delta))
    ])

    exp_search_model = sm.expected_number_searches_quad(psi_model)
    pi0_model = sm.prob_zero_searches_quad(delta, mu)
    pi1_model = sm.prob_one_search_quad(delta, mu)

    return {
        "delta": delta,
        "mu": mu,
        "choice": choice_model,
        "psi": psi_model,
        "exp_search": exp_search_model,
        "pi0": pi0_model,
        "pi1": pi1_model
    }

# =====================================================
# Moment construction
# =====================================================

def choice_moments(market_obj, theory):
    diff = market_obj["choice_data"] - theory["choice"]
    return diff[1:]   # drop outside option

def at_least_two_search_moments(market_obj, theory):
    pi2plus_data = 1.0 - market_obj["pi0_data"] - market_obj["pi1_data"]
    pi2plus_model = 1.0 - theory["pi0"] - theory["pi1"]
    return np.array([pi2plus_data - pi2plus_model])

def no_search_moments(market_obj, theory):
    return np.array([market_obj["pi0_data"] - theory["pi0"]])

def one_search_moments(market_obj, theory):
    return np.array([market_obj["pi1_data"] - theory["pi1"]])

def searched_product_moments(market_obj, theory):
    return market_obj["psi_data"] - theory["psi"]

def exp_search_moments(market_obj, theory):
    return np.array([market_obj["exp_search_data"] - theory["exp_search"]])

def build_moment_matrix(theta, data, selected_moments):
    market_blocks = []

    for market_obj in data:
        theory = compute_market_theory(theta, market_obj)
        blocks = []

        if "choice" in selected_moments:
            blocks.append(np.atleast_1d(choice_moments(market_obj, theory)))

        if "two_search" in selected_moments:
            blocks.append(np.atleast_1d(at_least_two_search_moments(market_obj, theory)))

        if "no_search" in selected_moments:
            blocks.append(np.atleast_1d(no_search_moments(market_obj, theory)))

        if "one_search" in selected_moments:
            blocks.append(np.atleast_1d(one_search_moments(market_obj, theory)))

        if "searched_product" in selected_moments:
            blocks.append(np.atleast_1d(searched_product_moments(market_obj, theory)))

        if "exp_search" in selected_moments:
            blocks.append(np.atleast_1d(exp_search_moments(market_obj, theory)))

        market_blocks.append(np.concatenate(blocks))

    return np.array(market_blocks)

def build_moment_vector(theta, data, selected_moments):
    M = build_moment_matrix(theta, data, selected_moments)
    return M.mean(axis=0)

# =====================================================
# GMM estimation
# =====================================================

def gmm_objective(theta, data, selected_moments, W):
    g = build_moment_vector(theta, data, selected_moments)
    return g.T @ W @ g

def estimate_one_step_gmm(all_moments, theta0, selected_moments):
    g0 = build_moment_vector(theta0, all_moments, selected_moments)
    W1 = np.eye(len(g0))

    res1 = minimize(
        gmm_objective,
        x0=theta0,
        args=(all_moments, selected_moments, W1),
        method="L-BFGS-B"
    )


    return {
        "res1": res1,
        "W1": W1
    }

# =====================================================
# Monte Carlo
# =====================================================

def _run_one_replication(r, T, N, J, beta_true, lambda_true, theta0, selected_moments, seed_start):
    product_id = np.arange(J + 1)
    
    all_moments = sim.simulate_dataset(
        T=T, N=N, J=J,
        beta=beta_true, lambda_=lambda_true,
        product_id=product_id,
        seed_base=seed_start + r * 1000
    )
    
    est = estimate_one_step_gmm(
        all_moments=all_moments,
        theta0=theta0,
        selected_moments=selected_moments
    )
    
    return {
        "replication": r,
        "theta1": est["res1"].x,
        "success1": est["res1"].success,
        "fun1": est["res1"].fun,
        "message1": est["res1"].message,
        "nit1": est["res1"].nit,
        "nfev1": est["res1"].nfev
    }

def monte_carlo_gmm_one_step(R, T, N, J, beta_true, lambda_true, theta0, selected_moments, seed_start=1000):
    results = Parallel(n_jobs=-1)(
        delayed(_run_one_replication)(
            r, T, N, J, beta_true, lambda_true, theta0, selected_moments, seed_start
        )
        for r in range(R)
    )
    return results