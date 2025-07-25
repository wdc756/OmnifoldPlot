# Written by William Dean Coker
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
    Test = 3
    Iteration = 4

    Custom = 5



@dataclass
class Dimension:
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
            print("Warning: Dimension.values is set while .start, .end, or .step are set. Values will ignore the "
                  "numbers.")



@dataclass
class Point:
    name: str
    color: str
    plot_error_bars: bool
    error_color: str
    shift: float

    data_dir: str
    nat_dir: str
    nat_file_name: str
    syn_dir: str
    weight_dir: str
    syn_pat: Pattern
    weight_pat: Pattern

    bins: np.linspace

    sets: Dimension
    percents: Dimension
    tests: Dimension
    iterations: Dimension
    num_datapoints: int
    average_bins: bool



    nat_data: Optional[np.ndarray] = None
    nat_weight: Optional[np.ndarray] = None
    syn_data: Optional[np.ndarray] = None
    syn_weight: Optional[np.ndarray] = None
    omnifold_weight: Optional[np.ndarray] = None

    sum_nat_weight: Optional[np.ndarray] = None
    sum_omnifold_weight: Optional[np.ndarray] = None
    percent_error: Optional[np.ndarray] = None
    std_dev: Optional[np.ndarray] = None

    percent_error_avg: Optional[np.ndarray] = None
    std_dev_avg: Optional[np.ndarray] = None
    x_values: Optional[np.ndarray] = None
    x_label: Optional[str] = None



@dataclass
class PlotOptions:
    points: list[Point]
    shift: float

    plot_dir: str
    plot_pat: Pattern
    plot_title_pat: Pattern

    verbose: int

    use_numpy_histogram: bool = False
    use_symmetric_percent_error: bool = False,
    calculate_std_dev_using_datapoints: bool = False
    normalize_std_dev: bool = False
    normalize_y_axis: bool = False
    use_symlog_yscale: bool = False





############################################################
# Main options
############################################################



# This is left here so the user can't accidentally erase the type-hints
plot_options: list[PlotOptions] = []



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



############################################################
# Data processing
############################################################



def _verify_weight_file(file_path):
    try:
        # Load the file
        data = np.load(file_path)
        data = data[:, 1, :]  # Only look at the second element of dim 1

        # Basic checks
        print(f"File successfully loaded: {file_path}")
        print(f"Array shape: {data.shape}")
        print(f"Data type: {data.dtype}")

        # Value checks
        print("\nValue analysis:")
        print(f"Number of non-zero elements: {np.count_nonzero(data)}")
        print(f"Number of inf values: {np.sum(np.isinf(data))}")
        print(f"Number of nan values: {np.sum(np.isnan(data))}")

        # Distribution of non-zero values
        nonzero = data[data != 0]
        if len(nonzero) > 0:
            print("\nNon-zero values statistics:")
            print(f"Min non-zero: {np.min(nonzero)}")
            print(f"Max non-zero: {np.max(nonzero)}")
            print(f"Mean non-zero: {np.mean(nonzero)}")
            print(f"Median non-zero: {np.median(nonzero)}")

        return True
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return False

def _compare_weight_files(problem_file, good_file):
    prob_data = np.load(problem_file)[:, 1, :]
    good_data = np.load(good_file)[:, 1, :]

    print("Shape comparison:")
    print(f"Problem file: {prob_data.shape}")
    print(f"Good file: {good_data.shape}")

    print("\nValue ranges:")
    print("Problem file:")
    print(f"Min: {np.min(prob_data)}, Max: {np.max(prob_data)}")
    print("Good file:")
    print(f"Min: {np.min(good_data)}, Max: {np.max(good_data)}")

    # Compare distribution characteristics
    print("\nDistribution characteristics:")
    for data, name in [(prob_data, "Problem"), (good_data, "Good")]:
        nonzero = data[data != 0]
        print(f"\n{name} file:")
        print(f"Number of non-zero values: {len(nonzero)}")
        if len(nonzero) > 0:
            print(f"Non-zero value percentiles:")
            for p in [0, 25, 50, 75, 100]:
                print(f"{p}th percentile: {np.percentile(nonzero, p)}")
