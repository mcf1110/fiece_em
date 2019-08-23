# FIECE-EM

FIECE-EM stands for Feasible-Infeasible Evolutionary Create & Eliminate Algorithm for Expectation Maximization.

If you find this code useful in your research, please consider citing:

```
@inproceedings{covoes2018classification,
  author    = {Cov{\~o}es, Thiago F and Hruschka, Eduardo R},
  title     = {Classification with Multi-Modal Classes Using Evolutionary Algorithms and Constrained Clustering},
  booktitle = {IEEE Congress on Evolutionary Computation (CEC)},
  year      = {2018},
  pages     = {1-8},
  doi       = {10.1109/CEC.2018.8477858},
}
```

# How to use

First, make you sure you have Python 3.x installed, with Numpy, Scipy, Pandas and Scikit-Learn.

Then, clone or download this repository and place the fiece_em directory in your working directory.

You can then use FIECE-EM by doing:

```python
from fiece_em import FieceEm, Constraints

X, y = read_my_dataset()

# extract 3 random objects from each class and generate corresponding constraints
constraints = Constraints.from_classes(3, y)
# create a FIECE-EM instance with a maximum of 20 clusters in a single individual
clf = FieceEm(constraints, 20)
# runs FIECE-EM
clf.fit(X)
```

# Active Learning

We also provide Active Learning strategies for FIECE-EM.

They are avaiable in the submodule `fiece_em.active_learning.strategies`.

Example:

```python
import fiece_em.active_learning as al

# get 5 objects
objects_to_label = al.strategies.best_feasible_classification_uncertainty(clf, constraints, X, 5)
# transform them into pairwise queries
pairwise_queries = al.to_pairwise(clf, constraints, X, objects_to_label)

# create a perfect oracle, which knows every object's label
oracle = al.oracles.make_oracle_from_labels(y)
# perform queries to the perfect oracle
new_constraints, number_of_questions = al.ask_oracle(constraints, pairwise_queries, oracle)

# OR perform queries to a human oracle in the python console
# new_constraints, number_of_questions = al.ask_oracle(constraints, pairwise_queries, al.oracles.simple_human_oracle)

# updates FIECE-EM to take into account new constraints
clf.update_constraints(new_constraints, X)
# fits it again
clf.fit(X)
```

For convenience, we provide the method `ask_with_strategy`, as follows:

```python
# create a perfect oracle, which knows every object's label
oracle = al.oracles.make_oracle_from_labels(y)

# performs queries about 5 objects
new_constraints, number_of_questions = al.ask_with_strategy(clf, constraints, X, oracle, al.strategies.best_feasible_classification_uncertainty, 5)

# updates FIECE-EM to take into account new constraints
clf.update_constraints(new_constraints, X)
# fits it again
clf.fit(X)
```

## Observation about LUC

When using LUC, the answer to each query influences which object will be the next one queried.
Therefore, a different function should be used:

```python
oracle = al.oracles.make_oracle_from_labels(y)
# labels 5 objects with LUC
new_constraints, number_of_questions = al.luc_custom_ask(clf, constraints, X, oracle, 5)

# updates FIECE-EM to take into account new constraints
clf.update_constraints(new_constraints, X)
# fits it again
clf.fit(X)
```

# Documentation

- [FieceEm](docs/fiece_em.md)
- [Constraints](docs/constraints.md)
