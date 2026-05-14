# =====================================================
# Imports
# =====================================================

import numpy as np
import pandas as pd
import search_model as sm

# =====================================================
# Parameter mapping 
# =====================================================

def mu_func(X_mu, lambda_):
    return np.array(np.log(1 + np.exp(X_mu @ lambda_)), dtype=float)

def delta_func(X_delta, beta, xi):
    return np.array(X_delta @ beta, dtype=float) + xi

# =====================================================
# Data-implied search probabilities
# =====================================================

def prob_searched_by_product_data(dat):
    dat_inside = dat.query("product_id != 0")
    return dat_inside["searched"].groupby(level="product_id").mean()

def prob_zero_searches_data(dat):
    dat_inside = dat.query("product_id != 0")
    searches_per_consumer = dat_inside["searched"].groupby(level="i").sum()
    return (searches_per_consumer == 0).mean()

def prob_one_search_data(dat):
    dat_inside = dat.query("product_id != 0")
    searches_per_consumer = dat_inside["searched"].groupby(level="i").sum()
    return (searches_per_consumer == 1).mean()

def expected_number_searches_data(dat):
    dat_inside = dat[dat.index.get_level_values("product_id") != 0]
    return dat_inside["searched"].groupby(level="i").sum().mean()

def choice_probs_data(dat, J):
    return (
        dat["d"].value_counts(normalize=True)
        .reindex(range(J + 1), fill_value=0.0)
        .to_numpy()
    )

# =====================================================
# Single simulation
# =====================================================

def simulate_single(N, J, deltas, mus, product_id, seed=None): 
    if seed is not None:
        np.random.seed(seed)

    assert deltas.size == J+1, "deltas must have size J+1"
    assert mus.size == J+1, "mus must have size J+1"
    assert len(product_id) == J+1, "product_id must have length J+1"

    # draws
    eps = np.random.gumbel(0.0, 1.0, size=(N, J+1))
    unis = np.random.uniform(0.0, 1.0, size=(N, J+1))


    # long-format
    idx = pd.MultiIndex.from_product([range(N), product_id], names=['i', 'product_id'])

    # allocating u, c, r, and w
    u = deltas[None, :] + eps
    c = np.zeros((N, J+1))
    r = np.zeros((N, J+1))
    w = np.zeros((N, J+1))

    # repeat mus for each i
    mu_mat = mus[None, :].repeat(N, axis=0)

    #c = Fc_inv_vec(unis.flatten(),mu_mat.flatten()).reshape(N,J+1)
    for j in range(J+1):
        Fc_inv_fast = sm.get_fast_Fc_inv(mu=mus[j], cmin=0.0, cmax=50.0, ngrid=1000)
        c[:, j] = Fc_inv_fast(unis[:, j])

    dat = pd.DataFrame({
        'mu': mu_mat.flatten(),
        'delta': deltas[None, :].repeat(N, axis=0).flatten(),
        'uni': unis.flatten(),
        'eps': eps.flatten(),
        'c': c.flatten(),
        'u': u.flatten()
    }, index=idx)

    # solve model
    dat['r'] = dat['delta'] + sm.H0_inv_vec(dat['c'].values)
    dat['w'] = np.fmin(dat['r'], dat['u'])
    w_mat = dat['w'].values.reshape(N, J+1)
    chosen_idx = np.argmax(w_mat, axis=1)
    chosen_product = product_id[chosen_idx]
    dat['d'] = np.repeat(chosen_product, J+1)

    # searched indicator
    dat['searched'] = dat['r'] >= dat.groupby(level='i')['w'].transform('max')
    return dat

# =====================================================
# Market simulation
# =====================================================

def simulate_onemarket(N, J, beta, lambda_, product_id, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # Product covariates
    price = np.random.uniform(20, 30, size=J+1)
    star = np.random.uniform(1, 5, size=J+1)
    xi = np.zeros(J+1)
    position = np.arange(J+1).astype(float)

    # Outside option normalization
    price[0] = 0.0
    star[0] = 0.0
    xi[0] = 0.0
    position[0] = 0.0

    # Build delta
    X_delta = pd.DataFrame({
        'Constant': np.ones(J+1),
        'Price': price,
        'Star': star
    })

    delta = delta_func(X_delta, beta, xi)
    delta[0] = 0.0

    # Build mu
    X_mu = pd.DataFrame({
        'Constant': np.ones(J+1),
        'Position': position
    })

    mu = mu_func(X_mu, lambda_)
    mu[0] = 0.0

    # Simulate individual data
    dat = simulate_single(
        N=N,
        J=J,
        deltas=delta,
        mus=mu,
        product_id=product_id)

    # Keep long format, but add market-level covariates to each row
    dat = dat.copy()
    dat['price'] = np.tile(price, N)
    dat['star'] = np.tile(star, N)
    dat['xi'] = np.tile(xi, N)
    dat['position'] = np.tile(position, N)

    # Separate market-level info
    market_info = pd.DataFrame({
        'product_id': product_id,
        'price': price,
        'star': star,
        'xi': xi,
        'position': position,
        'delta': delta,
        'mu': mu
    })

    return dat, market_info

# =====================================================
# Dataset simulation
# =====================================================

def simulate_dataset(T, N, J, beta, lambda_, product_id, seed_base=1000):
    all_moments = []

    for t in range(T):
        dat_t, info_t = simulate_onemarket(
            N=N,
            J=J,
            beta=beta,
            lambda_=lambda_,
            product_id=product_id,
            seed=seed_base + t
        )

        market_obj = {
            "market": t,
            "info": info_t,
            "choice_data": choice_probs_data(dat_t, J),
            "pi0_data": prob_zero_searches_data(dat_t),
            "pi1_data": prob_one_search_data(dat_t),
            "psi_data": prob_searched_by_product_data(dat_t),
            "exp_search_data": expected_number_searches_data(dat_t),
        }

        all_moments.append(market_obj)

    return all_moments