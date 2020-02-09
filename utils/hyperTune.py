from collections import namedtuple
from itertools import product


# Read in the hyper-parameters and return a Run namedtuple containing all the
# combinations of hyper-parameters
class RunBuilder:
  @staticmethod
  def get_runs(params):
    Run = namedtuple('Run', params.keys())
    runs = []
    for v in product(*params.values()):
      runs.append(Run(*v))

    return runs