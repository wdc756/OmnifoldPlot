# This file holds methods to process the data located in root/data

# Dictionary:
"""



"""

# Used to test dirs and filepaths
import os

# Used to read from files
#import ROOT
import uproot
import numpy as np

# Used for control flow based on user configuration
import src.util.options as op


def does_file_exist(filepath):
    return os.path.isfile(filepath)


def get_nat_150000_data(filepath):
    if not does_file_exist(filepath):
        raise Exception("File does not exist: " + filepath)

    with uproot.open(filepath) as f:
        tree = f["tree_nat_weight"]
        arrays = tree.arrays(["nat_pt_gen", "nat_pt_weight"], library="np")

    return arrays["nat_pt_gen"], arrays["nat_pt_weight"]

def get_syn_150000_data(filepath):
    if not does_file_exist(filepath):
        raise Exception("File does not exist: " + filepath)

    with uproot.open(filepath) as f:
        tree = f["tree_syn_weight"]
        arrays = tree.arrays(["syn_pt_gen"], library="np")

    return arrays["syn_pt_gen"]

def get_syn_150000_weights(filepath):
    if not does_file_exist(filepath):
        raise Exception("File does not exist: " + filepath)

    return np.load(filepath)



# # Get data from files
# ##############################
# nat_data_file = ROOT.TFile(nat_data_file_name)
# nat_data_tree = nat_data_file.Get('tree_nat_weight')
# nat_data_array = []
# nat_weight_array = []
# for entry in nat_data_tree:
#     nat_data_array.append(entry.nat_pt_gen)
#     nat_weight_array.append(entry.nat_pt_weight)
# nat_data_file.Close()
# nat_data = np.array(nat_data_array)
# nat_weights = np.array(nat_weight_array)
#
# syn_data_file = ROOT.TFile(syn_data_file_name)
# syn_data_tree = syn_data_file.Get('tree_syn_weight')
# syn_data_array = []
# for entry in syn_data_tree:
#     syn_data_array.append(entry.syn_pt_gen)
# syn_data = np.array(syn_data_array)
#
# syn_weights_raw = np.load(syn_weight_data_file_name)
# syn_weights = syn_weights_raw[0, 1]
#
# print("syn_data shape:", syn_data.shape)
# print("syn_weights shape:", syn_weights.shape)