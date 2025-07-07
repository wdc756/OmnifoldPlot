# This is the main file for making omnifold plots

# Dictionary:
"""



"""

"""# Setup
############################################################

# Run util/imports.py to make sure the user has all libraries installed
import util.imports

# Used to build file paths
import os

# Used to convert ROOT data to numpy arrays for plotting
# import util.bridge as bridge
import ROOT

# Used for data containers
import numpy as np

# Used for plotting
import matplotlib.pyplot as plt



# Data files paths
##############################
# Get dir this script is in
base_dir = os.path.dirname(os.path.abspath(__file__))

nat_data_file_name = os.path.join(base_dir, "..", "data", "mock", "mockdata.nat.Logweighted2.N150000.root")
syn_data_file_name = os.path.join(base_dir, "..", "data", "mock", "mockdata.syn1.1Percent.Logweighted2.N150000.root")
syn_weight_data_file_name = os.path.join(base_dir, "..", "data", "weights", "Syn1_1Percent_Test1.npy")



# Data loading and processing
############################################################



# Get data from files
##############################
nat_data_file = ROOT.TFile(nat_data_file_name)
nat_data_tree = nat_data_file.Get('tree_nat_weight')
nat_data_array = []
nat_weight_array = []
for entry in nat_data_tree:
    nat_data_array.append(entry.nat_pt_gen)
    nat_weight_array.append(entry.nat_pt_weight)
nat_data_file.Close()
nat_data = np.array(nat_data_array)
nat_weights = np.array(nat_weight_array)

syn_data_file = ROOT.TFile(syn_data_file_name)
syn_data_tree = syn_data_file.Get('tree_syn_weight')
syn_data_array = []
for entry in syn_data_tree:
    syn_data_array.append(entry.syn_pt_gen)
syn_data = np.array(syn_data_array)

syn_weights_raw = np.load(syn_weight_data_file_name)
syn_weights = syn_weights_raw[0, 1]

print("syn_data shape:", syn_data.shape)
print("syn_weights shape:", syn_weights.shape)


# Process data
##############################
# Bin data - size to 10, maybe 20 for testing
bins = np.linspace(0, 15, 20)

# Apply weights to nat/syn

nat_hist, bin_edges = np.histogram(nat_data, bins=bins, weights=nat_weights)
syn_hist, _ = np.histogram(syn_data, bins=bins, weights=syn_weights)

# Compute bin centers for plotting
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

# Get error%
# 1. Boolean mask of bins where natural count ≠ 0
mask = nat_hist != 0
# 2. Allocate percent_error array (default to NaN where nat_hist==0)
percent_error = np.full_like(nat_hist, fill_value=np.nan, dtype=float)
# 3. Compute percent error only in “valid” bins
percent_error[mask] = ((syn_hist[mask] - nat_hist[mask]) / nat_hist[mask]) * 100.0
# percent_error = syn_hist - nat_hist


# Get std. deviation




# Plotting
############################################################



plt.plot(bin_centers, nat_hist, drawstyle='steps-mid', label='Natural', color='blue')
plt.plot(bin_centers, syn_hist, drawstyle='steps-mid', label='Synthetic', color='orange')

plt.xlabel('Some kind of observable')
plt.ylabel('Count I think')
plt.title('Some title should go here')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



plt.figure(figsize=(6,3))
plt.plot(bin_centers[mask], percent_error[mask],
         drawstyle='steps-mid', marker='o', label='% Error')
# plt.plot(bin_centers, percent_error,
#          drawstyle='steps-mid', marker='o', label='% Error')
plt.axhline(0, color='k', linestyle='--', linewidth=1)
plt.xlabel('Energy difference?')
plt.ylabel('Some error thing')
plt.title('Should really get around to naming these')
plt.grid(True)
plt.tight_layout()
plt.show()"""

# Do this to check if the running version of python has all the right libraries
import src.util.imports




# Execution options
############################################################
import src.util.options as op

# Used to create dynamic file names
from src.util.file_patterns import TokenType, Token, FilePattern, get_file_path


# Plot type, choose from the options in PlotType()
##############################

# Plot Type Options:
"""
ALL   : 1 : Used to plot all main options
debug : 0 : Only used for testing purposes
"""

# If this is set to something not in PlotType(), the program will exit
plot_type = op.PlotType.ALL


# Binning options
##############################

# Binning range, end must be greater than start
# Note also that these numbers are in terms of Gev
op.bin_start = 0
op.bin_end = 80
op.bin_step = 20


# File path options
##############################
op.data_dir = "../../data"
op.nat_data_dir = "mock"
op.syn_data_dir = "mock"
op.weights_dir = "weights"
op.plots_dir = "plots"

# File Pattern Options:
"""
File Patterns are used to create dynamic file names that allow you to increment through a large amount of similarly-
named files. 

File Patterns work on a Token system. Each Token has a type, a string, and an optional increment integer, along with a
range tuple. Additionally, the class supports two functions. get_full_pattern() returns a string combining all the 
Tokens, and increment() increments the Token's integer by 1, moving right to left, and wrapping around at the end of 
the range. It also returns True if the Pattern could be incremented, and False if it was not. 

For more information, see the file_names.py file for more documentation, or look at options.py for the default values
"""
op.nat_data_file_pattern = FilePattern([
    Token(TokenType.STATIC, 'mockdata.nat.Logweighted2.N150000.root'),
])
op.syn_data_file_pattern = FilePattern([
    Token(TokenType.STATIC, 'mockdata.syn'),
    Token(TokenType.INCREMENTAL, '', 1, (1, 2, 1)),
    Token(TokenType.STATIC, '.'),
    Token(TokenType.INCREMENTAL, '', 1, (1, 5, 1)),
    Token(TokenType.STATIC, 'Percent.Logweighted2.N150000.root'),
])
op.weights_file_pattern = FilePattern([
    Token(TokenType.STATIC, 'Syn'),
    Token(TokenType.INCREMENTAL, '', 1, (1, 2, 1)),
    Token(TokenType.STATIC, '_'),
    Token(TokenType.INCREMENTAL, '', 1, (1, 5, 1)),
    Token(TokenType.STATIC, 'Percent_Test'),
    Token(TokenType.INCREMENTAL, '', 1, (1, 10, 1)),
    Token(TokenType.STATIC, '.npy')
])
op.plots_file_pattern = FilePattern([
    Token(TokenType.STATIC, 'plot.'),
    Token(TokenType.INCREMENTAL, '', 1, (1, 2, 1)),
    Token(TokenType.STATIC, '.'),
    Token(TokenType.INCREMENTAL, '', 1, (1, 5, 1)),
    Token(TokenType.STATIC, 'Percent.Logweighted2.N150000.png'),
])



# Execution
############################################################
# Used to pull data
from src.util.data import get_nat_150000_data, get_syn_150000_data, get_syn_150000_weights

# Used for plotting
import matplotlib.pyplot as plt


# Get nat data (in this case there's only one nat file
nat_data, nat_weights = get_nat_150000_data(get_file_path(op.nat_data_file_pattern, op.nat_data_dir))


# Loop through all files generating plots
##############################
while (True):
    # Get syn data
    syn_data = get_syn_150000_data(get_file_path(op.syn_data_file_pattern, op.syn_data_dir))

    print('\n' + op.syn_data_file_pattern.get_full_pattern())

    while (True):
        # Loop through all the syn weights and calculate avg
        iteration = 0
        step = 0
        syn_weight = get_syn_150000_weights(get_file_path(op.weights_file_pattern, op.weights_dir))

        print('   ' + op.weights_file_pattern.get_full_pattern())

        while (True):
            weight = syn_weight[iteration, step]

            # Basic checks before moving on
            if syn_data.shape != weight.shape:
                raise Exception('Shape mismatch between syn data and weights. ' +
                                op.syn_data_file_pattern.get_full_pattern() + ' ' +
                                op.weights_file_pattern.get_full_pattern())

            print('        ' + str(iteration) + '  ' + str(step))

            if iteration >= 4 and step >= 1:
                break
            elif step >= 1:
                step = 0
                iteration += 1
            else:
                step += 1

        if not op.weights_file_pattern.increment():
            break

    if not op.syn_data_file_pattern.increment():
        break

