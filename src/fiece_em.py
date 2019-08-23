"""The main FIECE-EM algorith, complying to sklearn's interfaces"""

from copy import deepcopy

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array
from sklearn.exceptions import NotFittedError

from .population import feasible_select, infeasible_select, initialize_feasible_infeasible


class FieceEm(BaseEstimator, ClassifierMixin):
    """The main FIECE-EM algorith, complying to sklearn's interfaces

    Parameters
    ----------
    constraints : Constraints
        An object representing the constraints
    kmax : int
        Maximum number of clusters.
    population_size : int
        Size of the populations to generate.
    max_init_tries : int
        Maximum number of allowed failures to generate a feasible individual.
    p_min : int
        Minumum size of the population.
    max_iter_em : int
        Maximum number of EM iterations.
    max_generations : int
        Maximum number of generations.
    min_improv: int
        An integer between 0-1, defines how much the fitness must improve
        before we stop.
    min_improv_gens: int
        Number of generations that fitness must improve at least "min_improv"
    can_return_infeasible : bool
        If no feasible solution is found, is it ok to return an infeasible?
    """

    def __init__(self, constraints, kmax, population_size=20, max_init_tries=3,
                 p_min=2, max_iter_em=5, max_generations=20, random_state=42,
                 min_improv=None, min_improv_gens=None,
                 can_return_infeasible=False,
                 k_means_max_iter=2, store_fitness_curve=False):
        self.constraints = deepcopy(constraints)
        self.kmin = len(self.constraints.get_chunklets())
        if kmax <= self.kmin:
            raise ValueError(
                "Kmax must be larger than the number of chunklets")
        self.kmax = kmax
        self.population_size = population_size
        self.max_init_tries = max_init_tries
        self.p_min = p_min
        self.max_iter_em = max_iter_em
        self.max_generations = max_generations
        self.k_means_max_iter = k_means_max_iter

        if min_improv and min_improv_gens:
            if 0 < min_improv < 1:
                self.min_improv = min_improv
                self.min_improv_gens = min_improv_gens
            else:
                raise ValueError("min_improv must be between 0 and 1")
        else:
            self.min_improv = None
            self.min_improv_gens = None

        self.best_individual_ = None
        self.is_best_feasible_ = False
        self.feasible = []
        self.infeasible = []

        self.can_return_infeasible = can_return_infeasible
        self.total_generations_run = 0

        self.store_fitness_curve = store_fitness_curve
        self.feasible_fitness_curve = []
        self.infeasible_fitness_curve = []
        self.total_parents_killed = 0
        self.potential_parents_killed = 0

        self.random_state = random_state

    def fill_remaining(self, feasible, infeasible, X, to_size):
        extra_feasible, extra_infeasible = \
            initialize_feasible_infeasible(
                X,
                self.constraints,
                max(0, to_size - len(feasible)),
                max(0, to_size - len(infeasible)),
                self.kmin,
                self.kmax,
                self.max_init_tries,
                self.k_means_max_iter
            )

        return feasible + extra_feasible, infeasible + extra_infeasible

    def fit(self, X):
        """Runs the FIECE-EM algorithm.

        Parameters
        ----------
        X : array_like
            List of n_features-dimensional data points.
            Each row corresponds to a single data point.

        Returns
        -------
        self

        """
        np.random.seed(self.random_state)
        X = check_array(X)
        feasible, infeasible = \
            self.fill_remaining([], [], X, self.population_size)
        self.total_parents_killed = 0
        self.potential_parents_killed = 0

        def add_to_pool(ind):
            """Adds the object to the correct pool"""
            if ind.is_feasible(self.constraints, X):
                fpool.append(ind)
            else:
                ipool.append(ind)

        t = 0
        last_best_fitness = []

        def stop_criteria():
            nonlocal last_best_fitness
            if t >= self.max_generations:
                return True
            if self.min_improv and self.min_improv_gens:
                if feasible:
                    fitness = feasible[0].feasible_fitness(X)
                    last_best_fitness = \
                        (last_best_fitness + [fitness])[-self.min_improv_gens:]
                    if len(last_best_fitness) == self.min_improv_gens:
                        return last_best_fitness[0] * (1 - self.min_improv) \
                            <= last_best_fitness[-1]
            return False

        while not stop_criteria():
            fpool, ipool = [], []
            self.potential_parents_killed += len(feasible)

            for feas in feasible:
                updated = feas.remove_empty_clusters(X)
                updated.update_mapping(self.constraints, X)
                updated.em(self.max_iter_em, X)
                add_to_pool(updated)

                if not updated.is_feasible(self.constraints, X):
                    fpool.append(feas)  # save parents with infeasible children
                else:
                    self.total_parents_killed += 1

                child = updated.mutate_feasible(self.kmin, self.kmax,
                                                self.constraints, X)
                child.em(self.max_iter_em, X)
                add_to_pool(child)
            feasible = []  # kill all parents

            for inf in infeasible:
                child = inf.mutate_infeasible(self.kmin, self.kmax,
                                              self.constraints, X)
                add_to_pool(child)

            feasible = feasible_select(feasible, fpool,
                                                  self.population_size, X)
            infeasible = infeasible_select(infeasible, ipool,
                                                      self.population_size,
                                                      self.constraints, X)

            feasible, infeasible = self.fill_remaining(feasible, infeasible, X,
                                                       self.p_min)
            t += 1

            if self.store_fitness_curve:
                self.store_curves(feasible, infeasible, X)

        self.feasible, self.infeasible = feasible, infeasible
        if feasible:
            best_index = np.argmax([f.feasible_fitness(X) for f in feasible])
            self.best_individual_ = feasible[best_index]
            self.is_best_feasible_ = True
        elif self.can_return_infeasible:
            best_index = np.argmax([i.infeasible_fitness(self.constraints, X)
                                    for i in infeasible])
            self.best_individual_ = infeasible[best_index]
            self.is_best_feasible_ = False

        if self.store_fitness_curve:
            self.store_curves(feasible, infeasible, X)

        self.total_generations_run += t
        return self

    def store_curves(self, feasible, infeasible, X):
        ffc = [f.feasible_fitness(X)
               for f in feasible] if feasible else [np.inf]
        ifc = [i.infeasible_fitness(self.constraints, X)
               for i in infeasible] if infeasible else [np.inf]
        self.feasible_fitness_curve.append(ffc)
        self.infeasible_fitness_curve.append(ifc)

    def _check_before_predict(self, X):
        X = check_array(X)
        if not self.best_individual_:
            raise NotFittedError

    def predict(self, X):
        """Predict the labels for the data samples in X using trained model.

        Parameters
        ----------
        X : array_like
            List of n_features-dimensional data points.
            Each row corresponds to a single data point.

        Returns
        -------
        labels : array, shape (n_samples,)
            Chunklet labels as predicted by the best feasible individual.

        """
        self._check_before_predict(X)
        return self.best_individual_.predict(X)

    def predict_cluster(self, X):
        self._check_before_predict(X)
        return self.best_individual_.predict_cluster(X)

    def predict_proba(self, X):
        self._check_before_predict(X)
        return self.best_individual_.predict_proba(X)

    def predict_proba_sum(self, X):
        self._check_before_predict(X)
        return self.best_individual_.predict_proba_sum(X)

    def update_constraints(self, constraints, X):
        """Update the constraints

        Parameters
        ----------
        constraints : Constraints
                An object representing the constraints
        X : array_like
            List of n_features-dimensional data points.
            Each row corresponds to a single data point.
        """
        self.constraints = deepcopy(constraints)

        individuals = self.feasible + self.infeasible
        self.feasible, self.infeasible = [], []
        for ind in individuals:
            if ind.is_feasible(self.constraints, X):
                self.feasible.append(ind)
            else:
                self.infeasible.append(ind)
            ind.reset_caches()
