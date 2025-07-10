# This file holds methods to process the data located in root/data

# Dictionary:
"""



"""

# Used to test dirs and filepaths
import os

# Used for file patterns
from startrace import *

# Used to help pass large amounts of data around
from dataclasses import dataclass
from typing import Optional

# Used to read from files
import uproot
import numpy as np


_cached_base_dir = None
def get_base_dir():
    global _cached_base_dir
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


@dataclass
class PlotSEIOptions():
    plot_syn_error: bool
    syn_color: str

    plot_weighted_syn_error: bool
    weighted_color: str

    plot_re_weighted_syn_error: bool
    re_weighted_color: str

    shift_distance: float

    plot_combined: bool

    plot_error_bars: bool

    bins_start: int
    bins_end: int
    bins_step: int

    num_syn_datasets: int
    num_percent_deviations: int
    num_tests: int
    num_iterations: int
    num_datapoints: int

    nat_data_dir: str
    syn_data_dir: str
    weight_dir: str
    re_weight_dir: str
    plot_dir: str

    nat_file_pat: Pattern
    syn_file_pat: Pattern
    weight_file_pat: Pattern
    re_weight_file_pat: Pattern
    error_plot_file_pat: Pattern



def get_nat_data_and_weights(filepath):
    if not does_file_exist(filepath):
        raise Exception("Nat file does not exist: " + filepath)

    with uproot.open(filepath) as f:
        tree = f["tree_nat_weight"]
        arrays = tree.arrays(["nat_pt_gen", "nat_pt_weight"], library="np")

    return arrays["nat_pt_gen"], arrays["nat_pt_weight"]



def get_syn_data(filepath):
    if not does_file_exist(filepath):
        raise Exception("Syn file does not exist: " + filepath)

    with uproot.open(filepath) as f:
        tree = f["tree_syn_weight"]
        arrays = tree.arrays(["syn_pt_gen"], library="np")

    return arrays["syn_pt_gen"]

def get_syn_weights(filepath):
    if not does_file_exist(filepath):
        raise Exception("Syn weights file does not exist: " + filepath)

    return np.load(filepath)

