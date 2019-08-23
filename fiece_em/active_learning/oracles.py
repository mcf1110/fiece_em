"""
This file contains all oracles or oracle generators.

An oracle is a function which, given two object indexes, returns a boolean where
True is a ML and False is a CL.

All functions whose name begins with `make_` are higher-order functions that
return an oracle. They may take any kind of parameter.
"""


def make_oracle_from_labels(class_labels):
    """Given a list of class labels, generates an oracle that will answer
    posed queries queries based on the labels.

    Parameters
    ----------
    class_labels : array_like
        A list of class labels, which the oracle will use to answer queries.

    Returns
    -------
    function
        An oracle function.

    """
    return lambda a, b: class_labels[a] == class_labels[b]


def simple_human_oracle(object_a, object_b):
    """This oracle will prompt a human user via Python terminal with the object's ids.

    Parameters
    ----------
    object_a : int
        First object index.
    object_b : int
        Second object index.

    Returns
    -------
    bool
        Is it a must-link?

    """
    ans = input("Do the objects %d and %d represent the same concept? (Y/N): "
                % (object_a, object_b))
    return ans in ('Y', 'y')
