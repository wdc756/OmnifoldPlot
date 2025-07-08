# This is the main file for making omnifold plots

# Dictionary:
"""



"""

# Used to create dynamic file names
from startrace import *

# Do this to check if the running version of python has all the right libraries
from src.util.imports import check_imports
if not check_imports():
    exit(1)

# Used to get the working dir
from src.util.data import get_base_dir



# Main options
############################################################

# graph_all = True

base_dir = get_base_dir()
data_dir = 'data'


# Error graphs - syn vs nat
##############################

# No nat bools or color settings because it's the frame of reference for the syn data

# Bool to graph unweighted synthetic data
plot_syn_error = False
syn_error_color = '#2ecc71'

plot_weighted_syn_error = True
weighted_syn_error_color = '#3498db'

plot_re_weighted_syn_error = True
re_weighted_syn_error_color = '#e74c3c'

# Combined Error graphs - syn vs nat
##############################

# This bool will re-use the above settings to graph the combined results
plot_combined_syn_error = True

# General graphing vars
##############################

graph_error_bars = True
syn_error_bins_start = 0
syn_error_bins_end = 100
syn_error_bins_step = 20

nat_data_dir = 'mock'
syn_data_dir = 'mock'
weight_dir = 'weights'
re_weights_dir = 're_weights'

nat_file_pat = Pattern([
    Token('mockdata.nat.Logweighted2.N150000.root'),
])
syn_file_pat = Pattern([
    Token('mockdata.syn', 1, Iter(1, 2, 1)),
    Token('.', 1, Iter(1, 5, 1)),
    Token('Percent.Logweighted2.N150000.root')
])
weight_file_pat = Pattern([
    Token('Syn', 1, Iter(1, 2, 1)),
    Token('_', 1, Iter(1, 5, 1)),
    Token('Percent_Test', 1, Iter(1, 10, 1)),
    Token('.npy')
])
re_weight_file_pat = Pattern([
    Token('Syn', 1, Iter(1, 2, 1)),
    Token('_', 1, Iter(1, 5, 1)),
    Token('Percent_Test', 1, Iter(1, 10, 1)),
    Token('.npy')
])

# NOTE FOR FUTURE WILL: MAKE SURE TO AUTOMATE BINS IN TOKEN!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# pls don't forget
error_plot_file_pat = Pattern([
    Token('syn', 1, Iter(1, 2, 1)),
    Token('.', 1, Iter(1, 5, 1)),
    Token('Percent.'),
    Token(['0-20', '20-40', '40-60', '60-80', '80-100']),
    Token('GeV.png')
])



# Function calls
############################################################

