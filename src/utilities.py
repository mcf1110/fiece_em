"""This module has shared functions. Just because its name is `utils`, it does
not mean the rest of this project has no utility (:"""
import numpy as np


def ranking_based_roulette(scores, descending=False, reposition=True):
    """Given an array of scores, returns an generator with probabilities
    proportional to the ranking positions

    Parameters
    ----------
    scores : array_like
        Scores to be ranked and calculate the probabilities.
    descending : bool
        Whether the ranking should be descending.
    reposition : bool
        Whether to use roulette with reposition.

    Returns
    -------
    generator
        An generator that returns indexes. The probability of an index being
        returned is proportional to its ranking on the `scores` array.
    """
    size = len(scores)
    if descending:
        scores = -1 * np.array(scores)
    sort = np.argsort(scores)
    probs = np.arange(size, 0, -1) / (size * (size + 1) / 2)
    intervals = probs.cumsum()
    while True:
        rnd = np.random.rand()
        index = np.argmax(intervals >= rnd)
        yield sort[index]

        if not reposition:
            probs[index] = 0
            probsum = probs.sum()
            if probsum <= 0:
                return
            probs = probs / probsum
            intervals = probs.cumsum()
