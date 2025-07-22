# This is the main launch file, containing settings, documentation, and execution calls

"""

Plotting steps:

1. Get nat and syn data, for bin mapping
2. Get nat and syn weights
3. Get omnifold weights
4. Multiply syn weights by omnifold weights
5. Bin unweighted nat and syn data, to get bin mappings
6. Sum nat weights and syn weights
7. Calculate percent error between nat and syn weights
8. Apply bins to % error calculations
9. Plot each point's data and save

"""

########################################################################################################################

# Imports

########################################################################################################################



# imports.check_imports will attempt to import all packages and throw errors if any are missing
from src.util.imports import check_imports
if not check_imports():
    exit(1)

from startrace import *

from src.util.data import *
from src.util.plot import *



########################################################################################################################

# Settings

########################################################################################################################



############################################################
# Main settings
############################################################



# This list allows us to create multiple different plots each time we execute
# plot_options = list[PlotOptions] - defined in data.py, so we only need to call plot_options.add(PlotOptions)

# This optional list tells the program which plot_options to use by index. If left empty, all plot_options will be
#   executed
# plot_options_to_use = [int] - also defined in data.py, so we can use it like plot_options



############################################################
# Plot over Set
############################################################



t_plot_over = PlotOver.Test

t_data_dir = 'data'

t_nat_dir = 'mock'
t_nat_file_name = 'mockdata.nat.Logweighted2.N150000.root'

t_bins_start = 0
t_bins_end = 100
t_bins_step = 20

t_sets = OmnifoldSetting(average=False, name='set', values=None, start=1, end=2, step=1)
t_percents = OmnifoldSetting(False, 'percent', None, 1, 5, 1)
t_iterations = OmnifoldSetting(False, 'iteration', None, 1, 5, 1)
t_tests = OmnifoldSetting(True, 'test', None, 1, 10, 1)
t_datapoints = 150000

t_shift = 0.1
t_points = [
    Point(
        name='weighted syn', color='#5dade2', plot_error_bars=True, error_color='#5dade2', shift=0.0,
        syn_dir='mock', weight_dir='weights',
        # mockdata.syn1.1Percent.Logweighted2.N150000.root
        syn_pat=Pattern([Token('mockdata.syn', t_sets.values), Token('.', t_percents.values),
                         Token('Percent.Logweighted2.N' + str(t_datapoints) + '.root')]),
        # Syn1_1Percent_Test1.npy
        weight_pat=Pattern([Token('Syn', t_sets.values), Token('_', t_percents.values),
                            Token('Percent_Test', t_tests.values), Token('.npy')]),
        sets=t_sets, percents=t_percents, iterations=t_iterations, tests=t_tests, num_datapoints=t_datapoints
    ),
    Point(
        're weighted syn', '#45b39d', True, '#45b39d', 0.0, 'mock', 're_weights',
        Pattern([Token('mockdata.syn', t_sets.values), Token('.', t_percents.values),
                 Token('Percent.Logweighted2.N' + str(t_datapoints) + '.root')]),
        Pattern([Token('Syn', t_sets.values), Token('_', t_percents.values), Token('Percent_Test', t_tests.values),
                 Token('.npy')]),
        t_sets, t_percents, t_iterations, t_tests, t_datapoints
    )
]

t_plot_dir = 'plots/sets'
t_plot_pat = Pattern([Token('syn'), Token(t_sets.values), Token('.png')])



plot_options.append(PlotOptions(plot_over=t_plot_over,
                             data_dir=t_data_dir,
                             nat_dir=t_nat_dir, nat_file_name=t_nat_file_name,
                             bins_start=t_bins_start, bins_end=t_bins_end, bins_step=t_bins_step,
                             shift=t_shift, points=t_points,
                             plot_dir=t_plot_dir, plot_pat=t_plot_pat))



########################################################################################################################

# Execution

########################################################################################################################



# if plot_options_to_use is empty, add all values
if len(plot_options_to_use) == 0:
    plot_options_to_use = list(range(len(plot_options)))

for plot_option in plot_options_to_use:
    plot(plot_options[plot_option])