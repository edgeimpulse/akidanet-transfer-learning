from typing import Tuple

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from .translate import translate_function

import numpy as np
import jax.numpy as jnp


class GaussianMixtureAnomalyScorer(object):

    def __init__(self, n_components: int, seed: int):
        self.gmm = GaussianMixture(
            n_components=n_components, random_state=seed,
            covariance_type='full')
        self.scaler = StandardScaler()
        self.fit_called = False

    def fit(self, x: np.array):
        # fit GMM
        self.gmm.fit(x)
        scores = self.gmm.score_samples(x)
        # use scores to fit scalar
        # note: scalar requires trailing dimension
        scores = np.expand_dims(scores, axis=-1)
        self.scaler.fit(scores)
        self.fit_called = True

    def score(self, x: np.array, use_jax: bool):
        if not self.fit_called:
            raise Exception("Must call fit() before score()")
        if use_jax:
            gmm_score_fn = translate_function(
                self.gmm, GaussianMixture.score_samples)
            scores = gmm_score_fn(x)
            # note: we add trailing dimension ONLY to match shape of non
            # jax version
            scores = jnp.expand_dims(scores, axis=-1)
            standardise_fn = translate_function(
                self.scaler, StandardScaler.transform)
            scores = standardise_fn(scores)
            return jnp.abs(scores)
        else:
            scores = self.gmm.score_samples(x)
            # recall: scalar requires trailing dimension
            scores = np.expand_dims(scores, axis=-1)
            scores = self.scaler.transform(scores)
            return np.abs(scores)
