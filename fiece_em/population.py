"""Contains functions that deal with populations"""
from itertools import cycle, islice
import warnings

import numpy as np

from .individual import Individual
from .utilities import ranking_based_roulette


def initialize_feasible_infeasible(X, r, fsize, isize, kmin, kmax, max_tries,
                                   k_means_max_iter):
    """Initializes both populations.

    Parameters
    ----------
    X : array_like
        Data to be fitted.
    r : Constraints
        Constraints to target.
    fsize : int
        Size of the Feasible population.
    isize : int
        Size of the Infeasible population.
    kmin : int
        Minimum number of clusters.
    kmax : int
        Maximum number of clusters.
    max_tries : int
        Maximum number of tries.
    k_means_max_iter : int
        Maximum kmeans iterations.

    Returns
    -------
    (list, list)
        The feasible and infeasible populations, respectively.

    """
    f, i = [], []
    msg = "Discarding negative feasible individual!"
    if fsize > 0:
        spacing = np.linspace(kmin, kmax, fsize, dtype=int)
        for k in cycle(spacing):
            if max_tries <= 0 or len(f) >= fsize:
                break
            ind = Individual.generate_feasible(X, k, r, k_means_max_iter)
            if ind.is_feasible(r, X):
                if ind.llk(X) < 0:
                    f.append(ind)
                else:
                    warnings.warn(msg, stacklevel=2)
                    max_tries -= 1

            else:
                i.append(ind)
                max_tries -= 1

    if isize > 0:
        spacing = np.linspace(kmin, kmax, isize, dtype=int)
        for k in cycle(spacing):
            if len(i) >= isize:
                break
            ind = Individual.generate_infeasible(X, k, r, k_means_max_iter)
            if ind.is_feasible(r, X) and ind.llk(X) < 0:
                f.append(ind)
            else:
                i.append(ind)

    return f[:fsize], i[:isize]


def feasible_select(population, pool, pop_size, X):
    """Given an population and a pool, return a new population.

    Parameters
    ----------
    population : list
        The current feasible population.
    pool : list
        The current feasible pool.
    pop_size : int
        How many individuals should be selected.
    X : array_like
        List of n_features-dimensional data points.
        Each row corresponds to a single data point.

    Returns
    -------
    list
        The new feasible population.

    """
    total = np.array(population + pool)
    if total.size == 0:
        return []
    ff_llk = [ind.ff_and_llk(X) for ind in total]
    fitness, llk = tuple(zip(*ff_llk))
    fitness = np.array(fitness)
    valid = np.array(llk) < 0
    if not np.all(valid):
        msg = "%d negative feasible individuals were found!" % (
            len(valid) - np.sum(valid))
        warnings.warn(msg, stacklevel=2)
        total, fitness = total[valid], fitness[valid]

    asort = np.argsort(fitness)
    ret = total[asort[:pop_size]]
    return list(ret)


def infeasible_select(population, pool, pop_size, constraints, X):
    """Given an population and a pool, return a new population.

    Parameters
    ----------
    population : list
        The current infeasible population.
    pool : list
        The current infeasible pool.
    pop_size : int
        How many individuals should be selected.
    constraints : Constraints
        The constraints to target.
    X : array_like
        List of n_features-dimensional data points.
        Each row corresponds to a single data point.

    Returns
    -------
    list
        The new infeasible population.

    """
    total = np.array(population + pool)
    if total.size == 0:
        return []
    if len(total) <= pop_size:  # prevents unecessary computations
        return list(total)
    fitness = [ind.infeasible_fitness(constraints, X) for ind in total]
    roulette = ranking_based_roulette(
        fitness, descending=False, reposition=False)
    return list(total[list(islice(roulette, pop_size))])
