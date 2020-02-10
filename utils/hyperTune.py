"""
Scripts enable to read in the hyper-parameters
and return a Run namedtuple containing
all the combinations of hyper-parameters
"""
from collections import namedtuple
from itertools import product


class RunBuilder:
    """
    RunBuilder get the combinations of hyper-parameters
    """

    @staticmethod
    def get_runs(params):
        """
        Parameters:
        param (Orderdict): contains all network param that we need to tune
        Returns:
        runs: list containing all runs with different parameters
       """
        Run = namedtuple('Run', params.keys())
        runs = []
        for val in product(*params.values()):
            runs.append(Run(*val))

        return runs
