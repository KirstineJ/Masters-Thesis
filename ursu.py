# =====================================================
# Imports
# =====================================================

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import simulation as sim
import estimation as est

# =====================================================
# Prepare Ursu data
# =====================================================

def prepare_ursu_data(df):
    df = df.copy()

    df["session_id"] = df["srch_id"]
    df["product_id"] = df["position"]
    df["i"] = df["session_id"]

    df["price"] = df["price_usd"] / 100
    df["star"] = df["prop_starrating"]

    df["searched"] = df["click_bool"]
    df["booked"] = df["booking_bool"]

    return df


# =====================================================
# Select sessions
# =====================================================

def select_ursu_sessions(df, n_sessions=None, top_k=7, seed=123):
    df_random = df.query("random_bool == 1").copy()

    valid_session_ids = []

    for session_id, df_s in df_random.groupby("session_id"):
        df_s = df_s.sort_values("position")

        chosen = df_s.loc[df_s["booked"] == 1, "product_id"]
        chosen = chosen.iloc[0] if len(chosen) > 0 else None

        top_products = df_s["product_id"].unique()[:top_k]

        if len(top_products) < top_k:
            continue

        if chosen is None or chosen in top_products:
            valid_session_ids.append(session_id)

    valid_session_ids = np.array(valid_session_ids)

    if n_sessions is not None:
        np.random.seed(seed)
        session_ids = np.random.choice(
            valid_session_ids,
            size=n_sessions,
            replace=False
        )
    else:
        session_ids = valid_session_ids

    return session_ids


# =====================================================
# Add outside option
# =====================================================

def add_outside_option(df_s):
    outside = pd.DataFrame([{
        "session_id": df_s["session_id"].iloc[0],
        "i": df_s["i"].iloc[0],
        "product_id": 0,
        "price": 0.0,
        "star": 0.0,
        "position": 0.0,
        "searched": 0,
        "booked": 0,
        "d": df_s["d"].iloc[0]
    }])

    return pd.concat([outside, df_s], ignore_index=True)


# =====================================================
# Build sessions
# =====================================================

def build_sessions(df, n_sessions=None, top_k=7, seed=123):
    sessions = []

    df = prepare_ursu_data(df)

    session_ids = select_ursu_sessions(
        df,
        n_sessions=n_sessions,
        top_k=top_k,
        seed=seed
    )

    for session_id in session_ids:
        df_s = df[df["session_id"] == session_id].copy()
        df_s = df_s.sort_values("position")

        # keep top K hotels
        top_products = df_s["product_id"].unique()[:top_k]
        
        if len(top_products) < top_k:
            continue

        df_s = df_s[df_s["product_id"].isin(top_products)].copy()

        # reindex hotels as 1,...,K
        products = np.sort(df_s["product_id"].unique())
        mapping = {p: i + 1 for i, p in enumerate(products)}
        df_s["product_id"] = df_s["product_id"].map(mapping)

        # construct chosen product ID
        chosen = df_s.loc[df_s["booked"] == 1, "product_id"]

        if len(chosen) > 0:
            chosen_product = chosen.iloc[0]
        else:
            chosen_product = 0

        df_s["d"] = chosen_product

        # add outside option as product_id = 0
        df_s = add_outside_option(df_s)

        J = top_k

        info = (
            df_s[["product_id", "price", "star", "position"]]
            .drop_duplicates()
            .sort_values("product_id")
        )

        dat = df_s.set_index(["i", "product_id"])

        choice_data = sim.choice_probs_data(dat, J)
        pi0_data = sim.prob_zero_searches_data(dat)
        pi1_data = sim.prob_one_search_data(dat)
        psi_data = sim.prob_searched_by_product_data(dat)
        exp_search_data = sim.expected_number_searches_data(dat)

        sessions.append({
            "session_id": session_id, 
            "info": info,
            "choice_data": choice_data,
            "pi0_data": pi0_data,
            "pi1_data": pi1_data,
            "psi_data": psi_data,
            "exp_search_data": exp_search_data
        })

    return sessions


# =====================================================
# Simulate one session (model moments)
# =====================================================

def simulate_session(theta, session, R):
    seed = 123 + int(session["session_id"])

    info = session["info"]

    delta = est.compute_deltas(theta, info)
    mu = est.compute_mus(theta, info)

    J = len(delta) - 1
    product_id = info["product_id"].to_numpy()

    dat = sim.simulate_single(
        N=R,
        J=J,
        deltas=delta,
        mus=mu,
        product_id=product_id,
        seed=seed 
    )

    choice = sim.choice_probs_data(dat, J)
    pi0 = sim.prob_zero_searches_data(dat)
    pi1 = sim.prob_one_search_data(dat)
    psi = sim.prob_searched_by_product_data(dat)
    exp_search = sim.expected_number_searches_data(dat)

    return {
        "choice": choice,
        "pi0": pi0,
        "pi1": pi1,
        "psi": psi,
        "exp_search": exp_search
    }


# =====================================================
# Moment construction
# =====================================================

def compute_model_moments(theta, sessions, R, selected_moments):
    moment_list = []

    for session in sessions:
        model = simulate_session(theta, session, R)
        data = session

        blocks = []

        if "choice" in selected_moments:
            diff = data["choice_data"] - model["choice"]
            blocks.append(diff[:-1])

        if "two_search" in selected_moments:
            pi2_data = 1.0 - data["pi0_data"] - data["pi1_data"]
            pi2_model = 1.0 - model["pi0"] - model["pi1"]
            blocks.append(np.array([pi2_data - pi2_model]))

        if "no_search" in selected_moments:
            blocks.append(np.array([data["pi0_data"] - model["pi0"]]))

        if "one_search" in selected_moments:
            blocks.append(np.array([data["pi1_data"] - model["pi1"]]))

        if "searched_product" in selected_moments:
            blocks.append(data["psi_data"] - model["psi"])

        if "exp_search" in selected_moments:
            blocks.append(np.array([data["exp_search_data"] - model["exp_search"]]))

        moment_list.append(np.concatenate(blocks))

    return np.mean(moment_list, axis=0)


# =====================================================
# GMM objective
# =====================================================

def gmm_objective(theta, sessions, R, selected_moments, W):
    g = compute_model_moments(theta, sessions, R, selected_moments)
    return g.T @ W @ g


# =====================================================
# Estimation
# =====================================================

def estimate_ursu_gmm(sessions, theta0, selected_moments, R):
    g0 = compute_model_moments(theta0, sessions, R, selected_moments)
    W = np.eye(len(g0))

    res = minimize(
        gmm_objective,
        x0=theta0,
        args=(sessions, R, selected_moments, W),
        method="L-BFGS-B",
    )

    return res