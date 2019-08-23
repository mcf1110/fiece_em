import sys
import warnings
from copy import deepcopy
import itertools
import numpy as np
from scipy.stats import entropy, multivariate_normal
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.mixture.gaussian_mixture import _compute_precision_cholesky,\
    _check_precision_matrix
import pandas as pd

from .utilities import ranking_based_roulette


def _diag_only(x):
    return np.diag(np.diag(x))


def _is_not_singular(a):
    return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]


def is_singular(a):
    try:
        _check_precision_matrix(a, 'full')
    except Exception:
        return True
    return not _is_not_singular(a)


def _singular_prevent(c):
    if is_singular(c):
        c = _diag_only(c)
        sp = np.identity(c.shape[0])
        while is_singular(c):
            c += sp
    return c


def _singular_prevent_multiple(covs):
    new_covs = np.zeros(covs.shape)
    if len(covs.shape) == 3:
        for i, c in enumerate(covs):
            new_covs[i] = _singular_prevent(c)
    return new_covs


def _create_gmm(k, means, weights, precisions=None, covariances=None):
    if covariances is None:
        precisions = np.array(precisions)
        covariances = np.linalg.pinv(precisions)
    elif precisions is None:
        covariances = np.array(covariances)
        precisions = np.linalg.pinv(covariances)

    gmm = GaussianMixture(n_components=k,
                          weights_init=weights,
                          means_init=means,
                          reg_covar=1e-2,
                          precisions_init=precisions,
                          max_iter=1,
                          warm_start=True)

    try:
        gmm.precisions_cholesky_ = _compute_precision_cholesky(covariances,
                                                               'full')
    except Exception:
        c2 = covariances.copy()
        covariances = _singular_prevent_multiple(covariances)
        precisions = np.linalg.pinv(covariances)
        try:
            gmm.precisions_cholesky_ = _compute_precision_cholesky(covariances,
                                                                   'full')
        except Exception:
            c2.dump('cov.npy')
            raise Exception('Problema na matriz! Dump no arquivo cov.npy')

    gmm.weights_ = weights
    gmm.means_ = means
    gmm.covariances_ = covariances
    gmm.precisions_ = precisions

    return gmm


def partial_log_likelihood(gmm, X):
    respons = gmm.predict_proba(X)
    pis = gmm.weights_
    log_pi_mat = np.tile(np.log(pis), (respons.shape[0], 1))

    try:
        N = np.array([multivariate_normal.pdf(X, gmm.means_[g],
                                              gmm.covariances_[g],
                                              allow_singular=True)
                      for g in range(gmm.n_components)]
                     ).T
    except Exception:
        N = np.array([multivariate_normal.pdf(X, gmm.means_[g],
                                              _singular_prevent(
            gmm.covariances_[g]))
            for g in range(gmm.n_components)]
        ).T
    N += sys.float_info.min
    log_N = np.log(N)

    plls = np.sum((log_N + log_pi_mat) * respons, axis=0)
    return plls


def _responsability_entropy(gmm, X):
    respons = entropy(gmm.predict_proba(X).T)
    return respons


def _closest_chunklet(obj, chunklets, X):
    return np.argmin([np.min([np.linalg.norm(X[idx] - obj)
                              for idx in c]) for c in chunklets])


class Individual:
    @staticmethod
    def covariance_matrices(X, y, n_clusters):
        X = np.array(X)
        y = np.array(y)
        clusters = (X[y == i] for i in range(n_clusters))
        n_attrs = len(X[0])

        return [(np.cov(g, rowvar=False)
                if len(g) > 1 else np.zeros((n_attrs, n_attrs)))
                for g in clusters]

    @staticmethod
    def random_mapping_(chunklets, k, X):
        nc = len(chunklets)
        n = len(X)

        # Map each chunklet to a cluster
        mapping = list(range(nc))
        seed_indexes = [np.random.choice(chunklets[i]) for i in mapping]

        # Fill remaining with random
        for _ in range(k - nc):
            random_index = np.random.choice(n)
            while random_index in seed_indexes:
                random_index = np.random.choice(n)

            mapping.append(_closest_chunklet(X[random_index], chunklets, X))
            seed_indexes.append(random_index)

        return seed_indexes, mapping

    @staticmethod
    def generate_from_kmeans(X, k, r, km_method, max_iter):
        chunklets = r.get_chunklets()
        seed_indexes, mapping = Individual.random_mapping_(chunklets, k, X)

        if not chunklets:
            km_method = 'random'

        if km_method == 'chunklets':
            seeds = X[seed_indexes]
            km = KMeans(n_clusters=k, init=seeds, n_init=1,
                        random_state=0, max_iter=max_iter + 1)
        else:
            km = KMeans(n_clusters=k, init=km_method, n_init=1,
                        random_state=0, max_iter=max_iter + 1)

        km.fit(X)
        y = km.predict(X)

        pis = np.bincount(y, minlength=k) / len(X)
        mus = km.cluster_centers_

        covs = Individual.covariance_matrices(X, y, km.n_clusters)

        return Individual(_create_gmm(k, mus, pis, covariances=covs), mapping)

    @staticmethod
    def generate_feasible(X, k, r, k_means_max_iter=2):
        return Individual.generate_from_kmeans(
            X, k, r, 'chunklets', k_means_max_iter)

    @staticmethod
    def generate_infeasible(X, k, r, k_means_max_iter=2):
        return Individual.generate_from_kmeans(
            X, k, r, 'random', k_means_max_iter)

    def reset_caches(self):
        self._infeasible_fitness = None
        self._ff_and_llk = None
        self._is_feasible = None

    def __init__(self, gmm, clusters_to_chunklets):
        if len(clusters_to_chunklets) != gmm.n_components:
            raise ValueError(
                "Clusters to chunklets should be of length n_components")
        self.gmm = deepcopy(gmm)
        self.clusters_to_chunklets = np.array(
            clusters_to_chunklets).astype(int)
        self.reset_caches()

    def em(self, number_iterations, X):
        self.gmm.set_params(max_iter=number_iterations)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            try:
                self.gmm.fit(X)
            except Exception:
                pass
        self.reset_caches()

    def is_feasible(self, constraints, X):
        if self._is_feasible is None:
            self._is_feasible = constraints.is_feasible(self.predict(X))
        return self._is_feasible

    def is_infeasible(self, constraints, X):
        return not self.is_feasible(constraints, X)

    def infeasible_fitness(self, constraints, X):
        if self._infeasible_fitness is None:
            cost = 0
            for chunklet, indexes in enumerate(constraints.get_chunklets()):
                allowed_clusters = self.clusters_to_chunklets == chunklet
                objects = X[indexes]
                predicted_chunklet =\
                    self.clusters_to_chunklets[self.gmm.predict(objects)]
                for idx, pred_chunk in zip(indexes, predicted_chunklet):
                    if pred_chunk != chunklet:
                        if not np.any(allowed_clusters):
                            cost += 1
                        else:
                            cost += 1 - \
                                np.max(self.gmm.predict_proba(
                                    [X[idx]])[0, allowed_clusters])
            self._infeasible_fitness = cost
        return self._infeasible_fitness

    def llk(self, X):
        return self.gmm.score(X) * X.shape[0]

    def ff_and_llk(self, X):
        if self._ff_and_llk is None:
            n, m = X.shape
            llk = self.llk(X)
            k = self.gmm.n_components
            # Minumim description Length
            penalization = k / 2 * (1 + m + (m * (m + 1) / 2)) * np.log(n)
            self._ff_and_llk = ((penalization - llk), llk)
        return self._ff_and_llk

    def feasible_fitness(self, X):
        return self.ff_and_llk(X)[0]

    def mutate_create(self, constraints, X, seeds_indexes):
        seeds = X[seeds_indexes]
        w, m, p, k = self.gmm.weights_.copy(), self.gmm.means_.copy(
        ), self.gmm.precisions_.copy(), self.gmm.n_components
        c = _singular_prevent_multiple(self.gmm.covariances_.copy())

        cov = _singular_prevent(np.diag(np.var(X, axis=0)) / 10)
        prec = np.linalg.inv(cov)

        mapping = self.clusters_to_chunklets.copy()
        chunklets = constraints.get_chunklets()
        slen = len(seeds)
        try:
            normals = [wi * multivariate_normal.pdf(seeds, mean=mi,
                                                    cov=ci,
                                                    allow_singular=True)
                       for mi, ci, wi in zip(m, c, w)]
            predicts = np.reshape(
                np.argmax(normals, axis=0),
                slen)
        except np.linalg.LinAlgError as e:
            print(e)
            np.save('./singular.npy', self.gmm.covariances_)
            raise Exception("Matriz singular salva no arquivo 'singular.npy'")
        closest_chunklets = []
        pis = []
        for s, cluster_to_split in zip(seeds, predicts):
            half_pi = w[cluster_to_split] / 2

            w[cluster_to_split] = half_pi
            pis.append(half_pi)

            closest_chunklets.append(_closest_chunklet(s, chunklets, X))
        k += slen
        w = np.append(w, pis)
        m = np.append(m, seeds, axis=0)
        p = np.append(p, [prec] * slen, axis=0)
        c = np.append(c, [cov] * slen, axis=0)
        mapping = np.append(mapping, closest_chunklets)
        gmm = _create_gmm(k, m, w, p, covariances=c)
        return Individual(gmm, mapping)

    def mutate_remove(self, clusters_indexes):
        w, m, p, k = self.gmm.weights_.copy(), self.gmm.means_.copy(
        ), self.gmm.precisions_.copy(), self.gmm.n_components.copy()
        mapping = self.clusters_to_chunklets.copy()

        should_really_remove = []
        _, counts = np.unique(mapping, return_counts=True)
        counts = list(counts)
        for g in clusters_indexes:
            if counts[mapping[g]] > 1:
                should_really_remove.append(g)
                counts[mapping[g]] -= 1

        if len(should_really_remove) > 0:
            w = np.delete(w, should_really_remove, axis=0)
            m = np.delete(m, should_really_remove, axis=0)
            p = np.delete(p, should_really_remove, axis=0)
            mapping = np.delete(mapping, should_really_remove, axis=0)
            k -= len(should_really_remove)

            w = w / np.sum(w)

        return Individual(_create_gmm(k, m, w, p), mapping)

    def _decide_clusters_to_remove(self, kmin, X):
        k = self.gmm.n_components
        n_clusters = np.random.randint(1, (k - kmin) + 1)
        pll = partial_log_likelihood(self.gmm, X)
        _, counts = np.unique(self.clusters_to_chunklets, return_counts=True)
        counts = list(counts)

        to_remove = []
        roulette = ranking_based_roulette(
            pll, descending=True, reposition=False)
        while len(to_remove) < n_clusters:
            try:
                g = next(roulette)
                chunk = self.clusters_to_chunklets[g]
                # prevents removing only cluster of chunklet
                if counts[chunk] > 1:
                    to_remove.append(g)
                    counts[chunk] -= 1
            except StopIteration:
                break
        return to_remove

    def mutate_feasible(self, kmin, kmax, constraints, X):
        k = self.gmm.n_components
        prob = (k - kmin) / (kmax - kmin)

        if np.random.rand() > prob:  # create
            n_clusters = np.random.randint(1, (kmax - k) + 1)
            ent = _responsability_entropy(self.gmm, X)

            roulette = itertools.islice(ranking_based_roulette(
                ent,
                descending=True,
                reposition=False),
                n_clusters)
            to_create = list(roulette)
            return self.mutate_create(constraints, X, to_create)
        # remove
        to_remove = self._decide_clusters_to_remove(kmin, X)
        return self.mutate_remove(to_remove)

    def _find_objects_out_of_chunklet(self, constraints, X):
        objects_out_of_chunklet = []
        costs = []
        for chunk, idxs in enumerate(constraints.get_chunklets()):
            idxs = np.array(idxs)
            wrong = idxs[self.clusters_to_chunklets[self.gmm.predict(
                X[idxs])] != chunk]
            if wrong.size:
                wrong_costs = 1 - self.gmm.predict_proba(X[wrong]).max(axis=1)

                objects_out_of_chunklet.extend(wrong)
                costs.extend(wrong_costs)
        return objects_out_of_chunklet, costs

    def mutate_infeasible(self, kmin, kmax, constraints, X):
        k = self.gmm.n_components
        if k < kmax:  # create
            n_clusters = np.random.randint(1, (kmax - k) + 1)
            objects_out_of_chunklet, costs = \
                self._find_objects_out_of_chunklet(constraints, X)
            roulette = itertools.islice(ranking_based_roulette(
                costs,
                descending=True,
                reposition=False),
                n_clusters)
            seeds_indexes = [objects_out_of_chunklet[i] for i in roulette]
            return self.mutate_create(constraints, X, seeds_indexes)
        # remove
        to_remove = self._decide_clusters_to_remove(kmin, X)
        return self.mutate_remove(to_remove)

    def mutate(self, kmin, kmax, constraints, X):
        if self.is_feasible(constraints, X):
            return self.mutate_feasible(kmin, kmax, constraints, X)
        return self.mutate_infeasible(kmin, kmax, constraints, X)

    def remove_empty_clusters(self, X):
        k = self.gmm.n_components
        predictions = np.unique(self.gmm.predict(X))
        all_clusters = set(range(k))
        to_remove = [x for x in all_clusters if x not in predictions]
        return self.mutate_remove(list(to_remove))

    def update_mapping(self, r, X):
        d = r.get_chunklet_dict()
        objs_in_chunks = list(d.keys())

        respons = self.gmm.predict_proba(X[objs_in_chunks])

        predicts = np.unique(respons.argmax(axis=1))

        _, counts = np.unique(self.clusters_to_chunklets, return_counts=True)

        for cluster, resp in enumerate(respons.T):
            mapped_chunklet = self.clusters_to_chunklets[cluster]
            not_only_representant = counts[mapped_chunklet] > 1
            if not_only_representant and cluster not in predicts:
                closest_object = objs_in_chunks[np.argmax(resp)]
                self.clusters_to_chunklets[cluster] = d[closest_object]
                _, counts = np.unique(
                    self.clusters_to_chunklets, return_counts=True)

        self.reset_caches()

    def predict_cluster(self, X):
        return self.gmm.predict(X)

    def predict(self, X):
        return self.clusters_to_chunklets[self.gmm.predict(X)]

    def _predict_proba_fn(self, fn, X):
        clusters_proba = self.gmm.predict_proba(X)
        class_proba = np.vstack([fn(clusters_proba[:, df.index], axis=1)
                                 for _, df in pd.DataFrame(
                                         self.clusters_to_chunklets
                                         ).clusterby(0)]).T
        return class_proba

    def predict_proba(self, X):
        prob = self._predict_proba_fn(np.max, X)
        return (prob.T / prob.sum(axis=1)).T

    def predict_proba_sum(self, X):
        return self._predict_proba_fn(np.sum, X)
