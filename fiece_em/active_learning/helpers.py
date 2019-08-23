"""
Active Learning module.

This module contains all basic active learning functions.
Its submodules contain premade oracles and strategies, but a developer may
come up with new ones that, as long as they respect the signatures, can be used here.
"""

from copy import deepcopy
import numpy as np

from .strategies import largest_unlabeled_clusters


def to_pairwise(fiece_em, r, X, indexes):
    """Given a list of indexes, generates a list of pairwise queries (tuples),
    which should be used as input with `ask_oracle`.

    Parameters
    ----------
    fiece_em : FieceEm
        A FieceEm instance.
    r : Constraints
        The current constraints.
    X : array_like
        The dataset.
    indexes : array_like
        The indexes to be made into pairwise queries.

    Returns
    -------
    list
        List of tuples representing pairwise queries, in descending order of
        proability by object, according to the model's current prediction.

    """
    chunklets = r.get_chunklets()
    X = X[indexes]
    proba = fiece_em.predict_proba(X)
    return [(ind, np.random.choice(chunklets[i]))
            for ind, row in zip(indexes, proba) for i in np.argsort(row)[::-1]]


def ask_oracle(r, queries, oracle, copy=True):
    """Performs the queries to the oracle, and returns the updated constraints.

    Parameters
    ----------
    r : Constraints
        The current constraints.
    queries : list of tuples
        A list of pairwise queries.
    oracle : function
        An oracle function to answer the given queries.

    Returns
    -------
    Constraints, int
        The updated constraints based on the oracle's answers,
        and the number of queries made

    """
    if copy:
        new_r = deepcopy(r)
    else:
        new_r = r
    count = 0
    for a, b in queries:
        if new_r.is_a_constraint(a, b):
            continue
        ans = oracle(a, b)
        if ans:
            new_r.add_must_link(a, b)
        else:
            new_r.add_cannot_link(a, b)
        count += 1
    return new_r, count

def ask_with_strategy(fiece_em, old_r, X, oracle, strategy, size):
    objects_to_label = strategy(fiece_em, old_r, X, size)
    pairwise_queries = to_pairwise(fiece_em, old_r, X, objects_to_label)
    return ask_oracle(old_r, pairwise_queries, oracle)


def luc_custom_ask(fiece_em, old_r, X, oracle, size):
    new_r = deepcopy(old_r)
    fiece_em = deepcopy(fiece_em)
    luc = largest_unlabeled_clusters(fiece_em, new_r, X, size)
    count = 0
    for xq in luc:
        queries = to_pairwise(fiece_em, new_r, X, [xq])
        _, count_add = ask_oracle(new_r, queries, oracle, False)
        count += count_add

        for f in fiece_em.feasible:
            f.update_mapping(new_r, X)
        fiece_em.update_constraints(new_r, X)

    return new_r, count
