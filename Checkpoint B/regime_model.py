"""
Hidden Markov Model of biotech market regimes.

A two-state Gaussian HMM is fit on daily observations of
    [XBI log return, log VIX level]
using training dates only. The state with the higher VIX level is labeled
"bear" (stress); the other is "bull" (calm).

The HMM is initialized explicitly (calm/stress means at the 25th/75th
percentiles, sticky transitions) because hmmlearn's default k-means
initialization collapses both states onto the same mean with current
numpy/scikit-learn, after which EM never separates them.

The feature passed to the LSTM is the *filtered* bear probability
P(state_t = bear | observations up to t), computed with the forward
recursion. hmmlearn's ``predict_proba`` returns smoothed probabilities that use
future observations, which would leak information into a backtest, so it is
not used.

``filtered_probs`` is vectorized over paths so monte_carlo.py can continue
the recursion on simulated markets from the real state at the last date.
"""

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

N_STATES = 2
SEED = 451


def observations(adj_close):
    """(T, 2) observations from a wide adj_close frame with XBI and ^VIX."""
    return pd.DataFrame({
        "xbi_log_return": np.log(adj_close["XBI"]).diff(),
        "log_vix": np.log(adj_close["^VIX"]),
    })


def fit(obs_train):
    """Fit the HMM on standardized training observations (NaN rows dropped)."""
    x = obs_train.dropna().to_numpy()
    mean, std = x.mean(0), x.std(0)
    z = (x - mean) / std
    hmm = GaussianHMM(n_components=N_STATES, covariance_type="full", init_params="",
                      params="stmc", n_iter=500, tol=1e-6, random_state=SEED)
    hmm.startprob_ = np.full(N_STATES, 1 / N_STATES)
    hmm.transmat_ = np.array([[0.98, 0.02], [0.05, 0.95]])
    hmm.means_ = np.quantile(z, [0.25, 0.75], axis=0)
    hmm.covars_ = np.stack([np.eye(z.shape[1])] * N_STATES)
    hmm.fit(z)
    # Bear = state with the higher VIX level (last observation column)
    bear = int(np.argmax(hmm.means_[:, -1]))
    return {
        "obs_mean": mean,
        "obs_std": std,
        "startprob": hmm.startprob_,
        "transmat": hmm.transmat_,
        "means": hmm.means_,
        "covars": hmm.covars_,
        "bear_state": bear,
    }


def _log_emission(x, params):
    """Gaussian log-likelihood of x (..., d) under each state -> (..., K)."""
    d = x.shape[-1]
    out = []
    for mu, cov in zip(params["means"], params["covars"]):
        diff = x - mu
        inv = np.linalg.inv(cov)
        quad = np.einsum("...i,ij,...j->...", diff, inv, diff)
        out.append(-0.5 * (quad + np.log(np.linalg.det(cov)) + d * np.log(2 * np.pi)))
    return np.stack(out, axis=-1)


def filtered_probs(obs, params, init=None):
    """Forward-filtered state probabilities.

    obs: (T, d) or (P, T, d) raw observations; NaN rows carry the prior forward.
    init: (K,) or (P, K) state probabilities before the first observation
          (defaults to the HMM start distribution).
    Returns probabilities of the same leading shape with K states last.
    """
    single = obs.ndim == 2
    x = obs[None] if single else obs
    x = (x - params["obs_mean"]) / params["obs_std"]
    p, t, _ = x.shape
    log_b = _log_emission(np.nan_to_num(x), params)
    missing = np.isnan(x).any(-1)

    alpha = np.broadcast_to(params["startprob"] if init is None else init, (p, N_STATES)).copy()
    out = np.empty((p, t, N_STATES))
    for step in range(t):
        prior = alpha @ params["transmat"] if step or init is not None else alpha
        lb = log_b[:, step]
        post = prior * np.exp(lb - lb.max(-1, keepdims=True))
        post = post / post.sum(-1, keepdims=True)
        alpha = np.where(missing[:, step, None], prior, post)
        out[:, step] = alpha
    return out[0] if single else out


def bear_prob(probs, params):
    return probs[..., params["bear_state"]]
