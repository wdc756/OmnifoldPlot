# This file helps with data and files

import os
from dataclasses import dataclass
import uproot
import numpy as np
from startrace import *



########################################################################################################################

# Synthetic Error plot options

########################################################################################################################



# @dataclass
# class File:
#     dir: str
#
#     pat: Pattern
#
#     set_index: int
#     percent_deviation_index: int
#     iteration_index: int
#
#     def
#
#     def increment(self):
#


@dataclass
class Point:
    name: str
    color: str
    plot_error_bars: bool
    error_color: str
    shift: float

    num_tests: int
    num_iterations: int
    num_datapoints: int

    syn_dir: str
    syn_pat: Pattern

    weight_dir: str
    weight_pat: Pattern



@dataclass
class PlotSEOptions:
    data_dir: str

    nat_dir: str
    nat_file_pat: Pattern

    points: list[Point]

    bins_start: int
    bins_end: int
    bins_step: int

    num_syn_datasets: int
    num_percent_deviations: int

    items_to_plot: list[int]
    plot_combined: bool
    shift: float

    plot_dir: str
    plot_file_pat: Pattern



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

