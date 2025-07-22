# This file helps with data and files

import os
from dataclasses import dataclass
from enum import Enum
from numbers import Number
from typing import Optional, Annotated

import numpy.typing as npt
import numpy as np

import uproot

from startrace import *



########################################################################################################################

# Synthetic Error plot options

########################################################################################################################



"""
Each plot will show, averaging each level below it:

1. 1/syn
    - x: percent
2. 1/syn & 1/percent
    - x: iteration
3. 1/syn & 1/percent & 1/iteration
    - x: test
4. 1/syn & 1/percent & 1/iteration & 1/test
    - x: bin
"""
@dataclass
class PlotOver(Enum):
    Set = 1
    Percent = 2
    Iteration = 3
    Test = 4

    Custom = 5



@dataclass
class OmnifoldSetting:
    average: bool
    name: str
    values: Optional[np.array] = None

    start: Optional[Number] = None
    end: Optional[Number] = None
    step: Optional[Number] = None

    def __post_init__(self):
        if self.values is None and (self.start is not None or self.end is not None or self.step is not None):
            self.values = np.arange(self.start, self.end + 1, self.step)
        elif self.values is not None and (self.start is not None or self.end is not None or self.step is not None):
            print("Warning: OmnifoldSetting.values is set while .start, .end, or .step are set. Values will ignore the "
                  "numbers.")
        # elif self.values is None and (self.start is None or self.end is None or self.step is None):
        #     raise ValueError("Error: OmnifoldSetting.values cannot be None if .start, .end, and .step are not set")


@dataclass
class Point:
    name: str
    color: str
    plot_error_bars: bool
    error_color: str
    shift: float

    syn_dir: str
    weight_dir: str
    syn_pat: Pattern
    weight_pat: Pattern

    sets: OmnifoldSetting
    percents: OmnifoldSetting
    iterations: OmnifoldSetting
    tests: OmnifoldSetting
    num_datapoints: int



    syn_data: Optional[Annotated[npt.NDArray[np.float64], "shape: (sets, percents, num_datapoints)"]] = None
    syn_weight: Optional[Annotated[npt.NDArray[np.float64], "shape: (sets, percents, num_datapoints)"]] = None
    omnifold_weight: Optional[Annotated[npt.NDArray[np.float64], "shape: (sets, percents, iterations, tests, num_datapoints)"]] = None
    sum_omnifold_weight: Optional[Annotated[npt.NDArray[np.float64], "shape: (sets, percents, iterations, tests, num_datapoints)"]] = None

    # The shapes for these will be generated on runtime, to be (sets, percents, iterations, tests, num_bins)
    percent_error: Optional[np.ndarray] = None
    std_error: Optional[np.ndarray] = None
    percent_error_avg: Optional[np.ndarray] = None
    std_error_avg: Optional[np.ndarray] = None
    x_values: Optional[np.ndarray] = None



@dataclass
class PlotOptions:
    plot_over: PlotOver

    data_dir: str

    nat_dir: str
    nat_file_name: str

    bins: np.linspace

    shift: float
    points: list[Point]

    plot_dir: str
    plot_pat: Pattern





############################################################
# Main options
############################################################



# These are left here so the user can't accidentally erase the type-hints
plot_options: list[PlotOptions] = []
plot_options_to_use: list[int] = []



########################################################################################################################

# Data utilities

########################################################################################################################



############################################################
# Base functions
############################################################



_cached_base_dir = None
def get_base_dir(new_base_dir: str=''):
    global _cached_base_dir

    if new_base_dir != '':
        _cached_base_dir = new_base_dir

    if _cached_base_dir is not None:
        return _cached_base_dir

    path = os.path.dirname(os.path.abspath(__file__))
    _cached_base_dir = os.path.join(path, '../../')
    return _cached_base_dir


def does_file_exist(filepath):
    return os.path.isfile(filepath)


def get_file_path(dirs: list[str]):
    path = get_base_dir()
    for d in dirs:
        path = os.path.join(path, d)
    return path



############################################################
# Data retrieval
############################################################



def get_nat_data_and_weights(filepath):
    if not does_file_exist(filepath):
        raise Exception("Nat file does not exist: " + filepath)

    with uproot.open(filepath) as f:
        tree = f["tree_nat_weight"]
        arrays = tree.arrays(["nat_pt_gen", "nat_pt_weight"], library="np")

    return arrays["nat_pt_gen"], arrays["nat_pt_weight"]

def get_syn_data_and_weights(filepath):
    if not does_file_exist(filepath):
        raise Exception("Syn file does not exist: " + filepath)

    with uproot.open(filepath) as f:
        tree = f["tree_syn_weight"]
        arrays = tree.arrays(["syn_pt_gen", "syn_pt_weight"], library="np")

    return arrays["syn_pt_gen"], arrays["syn_pt_weight"]

def get_omnifold_weights(filepath):
    if not does_file_exist(filepath):
        raise Exception("Syn weights file does not exist: ", str(filepath))

    return np.load(filepath)

