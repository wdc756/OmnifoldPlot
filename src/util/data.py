# This file holds methods to process the data located in root/data

# Dictionary:
"""



"""

# Used to test dirs and filepaths
import os

# Used to help pass large amounts of data around
from dataclasses import dataclass
from typing import Optional

# Used to read from files
import uproot
import numpy as np


def get_base_dir():
    path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(path, '../../')


def does_file_exist(filepath):
    return os.path.isfile(filepath)


@dataclass
class plot_syn_error_data():
    plot_syn_error: bool
    syn_error_color: str

    plot_weighted_syn_error: bool
    weighted_syn_error_color: str

    plot_re_weighted_syn_error: bool
    re_weighted_syn_error_color: str

    plot_combined_syn_error: bool
    plot_error_bars: bool
    syn_error_bins_start: int
    syn_error_bins_end: int



# def get_nat_150000_data(filepath):
#     if not does_file_exist(filepath):
#         raise Exception("File does not exist: " + filepath)
#
#     with uproot.open(filepath) as f:
#         tree = f["tree_nat_weight"]
#         arrays = tree.arrays(["nat_pt_gen", "nat_pt_weight"], library="np")
#
#     return arrays["nat_pt_gen"], arrays["nat_pt_weight"]
#
# def get_syn_150000_data(filepath):
#     if not does_file_exist(filepath):
#         raise Exception("File does not exist: " + filepath)
#
#     with uproot.open(filepath) as f:
#         tree = f["tree_syn_weight"]
#         arrays = tree.arrays(["syn_pt_gen"], library="np")
#
#     return arrays["syn_pt_gen"]
#
# def get_syn_150000_weights(filepath):
#     if not does_file_exist(filepath):
#         raise Exception("File does not exist: " + filepath)
#
#     return np.load(filepath)

