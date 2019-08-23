"""
This file contains active learning strategies.

A strategy is a functions which, given a FieceEm instance, the current
Constraints, a Dataset and an Integer N, returns N indexes from the dataset
that have no constraints yet.
"""

import numpy as np
import scipy as sc
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def random(fiece_em, r, X, size):
    """Selects unqueries objects at random.

    Parameters
    ----------
    fiece_em : FieceEm
        A trained FieceEm instance.
    r : Constraints
        The current Constraints.
    X : array_like
        The dataset.
    size : int
        How many object should be sampled.

    Returns
    -------
    list of ints
        The chosen indexes.

    """
    n = len(X)
    p = np.ones(n)
    p[r.get_flat_chunklets()] = 0
    p /= p.sum()
    return np.random.choice(n, size, p=p, replace=False)


def best_feasible_classification_uncertainty(fiece_em, r, X, size):
    """Select the objects which, according to the best individual in FieceEm,
    have the highest class probability entropy (which isn't already in a Chunklet).

    Parameters
    ----------
    fiece_em : FieceEm
        A trained FieceEm instance.
    r : Constraints
        The current Constraints.
    X : array_like
        The dataset.
    size : int
        How many object should be sampled.

    Returns
    -------
    list of ints
        The chosen indexes.

    """
    already_in_chunk = r.get_flat_chunklets()
    proba = fiece_em.predict_proba(X)
    entropies = np.apply_along_axis(sc.stats.entropy, 1, proba)

    argsrt = entropies.argsort()
    argsrt_without_chunk = [a for a in argsrt if a not in already_in_chunk]
    return argsrt_without_chunk[-size:][::-1]


def feasible_classification_uncertainty(fiece_em, r, X, size):
    """Selects the objects on which the feasible population disagrees the most, i.e.,
    objects that are predicted in the least homogeneous form among feasible individuals.

    Goes through all the feasible individuals and gets their class prediction
    for all objects. Then, for each object, gets the entropy of their predictions,
    and returns those with the largest (which isn't already in a Chunklet).

    Parameters
    ----------
    fiece_em : FieceEm
        A trained FieceEm instance.
    r : Constraints
        The current Constraints.
    X : array_like
        The dataset.
    size : int
        How many object should be sampled.

    Returns
    -------
    list of ints
        The chosen indexes.

    """
    def _observation_to_entropy(obs):
        _, counts = np.unique(obs, return_counts=True)
        prob = counts / counts.sum()
        return sc.stats.entropy(prob)

    if len(fiece_em.feasible) <= 1:
        return best_feasible_classification_uncertainty(fiece_em, r, X, size)
    already_in_chunk = r.get_flat_chunklets()
    y_hats = np.stack([ind.predict(X) for ind in fiece_em.feasible])
    entropies = np.apply_along_axis(_observation_to_entropy, 0, y_hats)

    argsrt = entropies.argsort()
    argsrt_without_chunk = [a for a in argsrt if a not in already_in_chunk]
    return argsrt_without_chunk[-size:][::-1]


def distance_to_violated_objects(fiece_em, r, X, size):
    """Let X be an object in a violated constraint, 
    and Y its nearest neighbor which is not in a chunklet.
    Select objects based on the following score:
        dist(X,Y)/(Sum of (cost_vio(X)) across all infeasible individuals)

    Parameters
    ----------
    fiece_em : FieceEm
        A trained FieceEm instance.
    r : Constraints
        The current Constraints.
    X : array_like
        The dataset.
    size : int
        How many object should be sampled.

    Returns
    -------
    list of ints
        The chosen indexes.

    """
    flat = np.array(r.get_flat_chunklets())
    d = r.get_chunklet_dict()
    correct = np.array([d[f] for f in flat])
    X_chunk = X[flat]
    eps = np.finfo(float).eps
    obj_costs = {x: eps for x in flat}
    for ind in fiece_em.infeasible:
        problematic = ind.predict(X_chunk) != correct
        for obj_idx, correct_chunk in zip(
                flat[problematic], correct[problematic]):
            allowed_groups = ind.groups_to_chunklets == correct_chunk
            cost = 1 - \
                np.max(ind.gmm.predict_proba([X[obj_idx]])[0, allowed_groups])
            obj_costs[obj_idx] += cost

    obj_costs = np.array(list(obj_costs.values()))
    mask = np.full(X.shape[0], True)
    mask[flat] = False
    nbrs = NearestNeighbors(n_neighbors=1).fit(X[mask])
    distances, indices = nbrs.kneighbors(X_chunk)
    indices = np.arange(len(X))[mask][indices]  # mapping to original indexes
    return indices[:, 0][(distances[:, 0] / obj_costs).argsort()[:size]]


def simple_largest_unlabeled_clusters(fiece_em, r, X, size):
    """Finds unlabeled groups in feasible individuals and labels the ones
    with the greatest support.

    Parameters
    ----------
    fiece_em : FieceEm
        A trained FieceEm instance.
    r : Constraints
        The current Constraints.
    X : array_like
        The dataset.
    size : int
        How many object should be sampled.

    Returns
    -------
    list of ints
        The chosen indexes.
    """
    objs_in_chunks = r.get_flat_chunklets()

    indexes, supports = [], []
    for ind in fiece_em.feasible:

        unlabeled = list(set(range(ind.gmm.n_components)) -
                         set(ind.gmm.predict(X[objs_in_chunks])))
        probas = ind.gmm.predict_proba(X)
        supports.extend(probas.sum(axis=0)[unlabeled])
        indexes.extend(probas[:, unlabeled].argmax(axis=0))

    return pd.unique(np.array(indexes)[np.argsort(supports)[::-1]])[:size]


def make_fallback_count():
    return {
        'lug': 0,
        'bfcu': 0,
        'fcu': 0
    }


# Counts how many times lug's fallbacks have been called
fallback_count = make_fallback_count()


def largest_unlabeled_clusters(fiece_em, r, X, size):
    """Finds unlabeled groups in feasible individuals and labels the ones
    with the greatest support. The difference between this and simple_unlabeled_groups
    is in the fact that this one is an iterator that repeats the whole process of
    finding the largest groups, allowing for mutability to happen on fiece_em and
    the constraints. Should be used only with `active_learning.lug_custom_ask`.

    Parameters
    ----------
    fiece_em : FieceEm
        A trained FieceEm instance.
    r : Constraints
        The current Constraints.
    X : array_like
        The dataset.
    size : int
        How many object should be sampled.

    Returns
    -------
    iterator of ints
        The chosen indexes.
    """
    for i in range(size):
        objs_in_chunks = r.get_flat_chunklets()

        indexes, supports = [], []
        for ind in fiece_em.feasible:
            ind = ind.remove_empty_groups(X)
            unlabeled = list(set(range(ind.gmm.n_components)) -
                             set(ind.gmm.predict(X[objs_in_chunks])))
            probas = ind.gmm.predict_proba(X)
            supports.extend(probas.sum(axis=0)[unlabeled])
            indexes.extend(probas[:, unlabeled].argmax(axis=0))

        if supports:
            fallback_count['lug'] += 1
            yield indexes[np.argmax(supports)]
        else:  # no unlabeled groups
            remaining = size - i
            if len(fiece_em.feasible) > 1:
                fallback_count['fcu'] += remaining
                results = feasible_classification_uncertainty(
                    fiece_em, r, X, remaining)
            else:
                fallback_count['bfcu'] += remaining
                results = best_feasible_classification_uncertainty(
                    fiece_em, r, X, remaining)
            for res in results:
                yield res
            break
