# This is the main launch file, containing settings, documentation, and execution calls

########################################################################################################################

# Imports

########################################################################################################################



# imports.check_imports will attempt to import all packages and throw errors if any are missing
from src.util.imports import check_imports
if not check_imports():
    exit(1)

from src.util.plot import *



########################################################################################################################

# Default plotting

########################################################################################################################



# If you're using the default project structure as laid out in the ReadMe, then set these average bools and run the
#   program. If your setup is different, or you want more control, disable use_default_options and scroll down to the
#   manual plotting setup
average_sets = False
average_percents = False
average_iterations = False
average_bins = True
verbose = 2

use_default_options = True

if use_default_options:
    plot_defaults(average_sets, average_percents, average_iterations, average_bins, verbose)



########################################################################################################################

# Manual plotting

########################################################################################################################



############################################################
# Manual vars example
############################################################



ex_verbose = 2


# Hack-y variables, don't touch unless you know what you're doing
##############################


# Determines if the program should average over tests. If this is false, then it's almost required to enable
#   calculate_std_dev_using_datapoints
ex_average_tests = True
# Determines if the program should use the default numpy histogram() function. This is disabled by default because
#   the function seemed unstable at high energy bins, returning a bunch of 0s
ex_use_numpy_histogram = False
# Determines if the program should use symmetric percent error. It could be useful if you're ever dealing with
#   nat bins that have no data, as this calculation will give you real values
ex_use_symmetric_percent_error = False
# Determines if the program should calculate standard deviation using datapoints. By default this is false because
#   calculating std dev over points was unstable, and deemed unnecessary in the long run
ex_calculate_std_dev_using_datapoints = False
# Determines if the program should normalize wights when calculating standard deviation. This is false by default
#   because it's not as accurate. Note: this will only have an effect when calculate_std_dev_using_datapoints is True
ex_normalize_std_dev = False
# Determines if the program should normalize all plots' y-axes to the min and max values. This is set to True by
#   default because it helps with comparison
ex_normalize_y_axis = True
# Determines if the program should use symlog to scale on the y-axis. This is generally recommended when
#   normalize_y_axis is true
ex_use_symlog_yscale = True


# File vars
##############################


ex_data_dir = 'data'
ex_nat_dir = 'mock'
ex_nat_file_name = 'mockdata.nat.Logweighted2.N150000.root'
ex_syn_dir = 'mock'
ex_weight_dir = 'weights'
ex_re_weight_dir = 're_weights'
ex_spline_weight_dir = 'spline_weights'
ex_plot_dir = ex_data_dir + '/plots'


# Omnifold settings
##############################


ex_bins_start = 0
ex_bins_end = 100

ex_use_bins_step = False
ex_bins_step = 20
ex_bins_count = 20

if ex_use_bins_step:
    ex_bins = np.arange(ex_bins_start, ex_bins_end + ex_bins_step, ex_bins_step)
else:
    ex_bins = np.linspace(ex_bins_start, ex_bins_end, ex_bins_count + 1)
if verbose > 2:
    print('bins:', ex_bins)

ex_sets = Dimension(False, 'Set', None, 1, 2, 1)
ex_percents = Dimension(False, 'Percent', None, 1, 5, 1)
ex_tests = Dimension(ex_average_tests, 'Test', None, 1, 10, 1)
ex_iterations = Dimension(False, 'Iteration', None, 1, 5, 1)
ex_average_bins = False
ex_datapoints = 150000


# Points to plot
##############################


ex_plot_error_bars = True
ex_points = [
    Point(
        'weighted syn', '#5dade2', ex_plot_error_bars, '#5dade2', 0.0, ex_data_dir, ex_nat_dir,
        ex_nat_file_name, ex_syn_dir, ex_weight_dir,
        Pattern([Token('mockdata.syn', ex_sets.values), Token('.', ex_percents.values),
                 Token('Percent.Logweighted2.N' + str(ex_datapoints) + '.root')]),
        Pattern([Token('Syn', ex_sets.values), Token('_', ex_percents.values), Token('Percent_Test', ex_tests.values),
                 Token('.npy')]),
        ex_bins, ex_sets, ex_percents, ex_tests, ex_iterations, ex_datapoints, average_bins
    ),
    Point(
        're weighted syn', '#FB2C36', ex_plot_error_bars, '#FB2C36', 0.0, ex_data_dir, ex_nat_dir,
        ex_nat_file_name, ex_syn_dir, ex_re_weight_dir,
        Pattern([Token('mockdata.syn', ex_sets.values), Token('.', ex_percents.values),
                 Token('Percent.Logweighted2.N' + str(ex_datapoints) + '.root')]),
        Pattern([Token('Syn', ex_sets.values), Token('_', ex_percents.values), Token('Percent_Test', ex_tests.values),
                 Token('.npy')]),
        ex_bins, ex_sets, ex_percents, ex_tests, ex_iterations, ex_datapoints, average_bins
    ),
    Point(
        'spline weighted', '#31C950', ex_plot_error_bars, '#31C950', 0.0, ex_data_dir, ex_nat_dir,
        ex_nat_file_name, ex_syn_dir, ex_spline_weight_dir,
        Pattern([Token('mockdata.syn', ex_sets.values), Token('.', ex_percents.values),
                 Token('Percent.Logweighted2.N' + str(ex_datapoints) + '.root')]),
        Pattern([Token('Syn', ex_sets.values), Token('_', ex_percents.values), Token('Percent_Test', ex_tests.values),
                 Token('.npy')]),
        ex_bins, ex_sets, ex_percents, ex_tests, ex_iterations, ex_datapoints, ex_average_bins
    )
]
# The distance between points on the x-axis, measured by the difference between x-ticks
ex_shift = 0.25


# Plotting
##############################



ex_plot_pat = Pattern([Token('syn', ex_sets.values), Token('percent', ex_percents.values),
                       Token('iteration', ex_iterations.values), Token('.png')])
ex_plot_title_pat = Pattern([Token('Syn ', ex_sets.values), Token(' Percent ', ex_percents.values),
                             Token(' Iteration', ex_iterations.values)])



# Compile data
##############################


plot_options.append(PlotOptions(ex_points, ex_shift, ex_plot_dir, ex_plot_pat, ex_plot_title_pat, ex_verbose,
                                ex_use_numpy_histogram, ex_use_symmetric_percent_error,
                                ex_calculate_std_dev_using_datapoints, ex_normalize_std_dev, ex_normalize_y_axis,
                                ex_use_symlog_yscale))



############################################################
# Testing
############################################################



test_verbose = 3


# Hack-y variables, don't touch unless you know what you're doing
##############################


# Determines if the program should average over tests. If this is false, then it's almost required to enable
#   calculate_std_dev_using_datapoints
test_average_tests = True
# Determines if the program should use the default numpy histogram() function. This is disabled by default because
#   the function seemed unstable at high energy bins, returning a bunch of 0s
test_use_numpy_histogram = False
# Determines if the program should use symmetric percent error. It could be useful if you're ever dealing with
#   nat bins that have no data, as this calculation will give you real values
test_use_symmetric_percent_error = False
# Determines if the program should calculate standard deviation using datapoints. By default this is false because
#   calculating std dev over points was unstable, and deemed unnecessary in the long run
test_calculate_std_dev_using_datapoints = False
# Determines if the program should normalize wights when calculating standard deviation. This is false by default
#   because it's not as accurate. Note: this will only have an effect when calculate_std_dev_using_datapoints is True
test_normalize_std_dev = False
# Determines if the program should normalize all plots' y-axes to the min and max values. This is set to True by
#   default because it helps with comparison
test_normalize_y_axis = True
# Determines if the program should use symlog to scale on the y-axis. This is generally recommended when
#   normalize_y_axis is true
test_use_symlog_yscale = True


# File vars
##############################


test_data_dir = 'data'
test_nat_dir = 'mock'
test_nat_file_name = 'mockdata.nat.Logweighted2.N150000.root'
test_syn_dir = 'mock'
test_weight_dir = 'weights'
test_re_weight_dir = 're_weights'
test_spline_weight_dir = 'spline_weights'
test_plot_dir = test_data_dir + '/plots/test'


# Omnifold settings
##############################


test_bins_start = 0
test_bins_end = 100

test_use_bins_step = False
test_bins_step = 20
test_bins_count = 20

if test_use_bins_step:
    test_bins = np.arange(test_bins_start, test_bins_end + test_bins_step, test_bins_step)
else:
    test_bins = np.linspace(test_bins_start, test_bins_end, test_bins_count + 1)
if verbose > 2:
    print('bins:', test_bins)

test_sets = Dimension(False, 'Set', None, 2, 2, 1)
test_percents = Dimension(False, 'Percent', None, 5, 5, 1)
test_tests = Dimension(test_average_tests, 'Test', None, 1, 10, 1)
test_iterations = Dimension(False, 'Iteration', None, 3, 3, 1)
test_average_bins = False
test_datapoints = 150000


# Points to plot
##############################


test_plot_error_bars = True
test_points = [
    Point(
        'weighted syn', '#5dade2', test_plot_error_bars, '#5dade2', 0.0, test_data_dir, test_nat_dir,
        test_nat_file_name, test_syn_dir, test_weight_dir,
        Pattern([Token('mockdata.syn', test_sets.values), Token('.', test_percents.values),
                 Token('Percent.Logweighted2.N' + str(test_datapoints) + '.root')]),
        Pattern([Token('Syn', test_sets.values), Token('_', test_percents.values), Token('Percent_Test', test_tests.values),
                 Token('.npy')]),
        test_bins, test_sets, test_percents, test_tests, test_iterations, test_datapoints, average_bins
    ),
    Point(
        're weighted syn', '#FB2C36', test_plot_error_bars, '#FB2C36', 0.0, test_data_dir, test_nat_dir,
        test_nat_file_name, test_syn_dir, test_re_weight_dir,
        Pattern([Token('mockdata.syn', test_sets.values), Token('.', test_percents.values),
                 Token('Percent.Logweighted2.N' + str(test_datapoints) + '.root')]),
        Pattern([Token('Syn', test_sets.values), Token('_', test_percents.values), Token('Percent_Test', test_tests.values),
                 Token('.npy')]),
        test_bins, test_sets, test_percents, test_tests, test_iterations, test_datapoints, average_bins
    ),
    Point(
        'spline weighted', '#31C950', test_plot_error_bars, '#31C950', 0.0, test_data_dir, test_nat_dir,
        test_nat_file_name, test_syn_dir, test_spline_weight_dir,
        Pattern([Token('mockdata.syn', test_sets.values), Token('.', test_percents.values),
                 Token('Percent.Logweighted2.N' + str(test_datapoints) + '.root')]),
        Pattern([Token('Syn', test_sets.values), Token('_', test_percents.values), Token('Percent_Test', test_tests.values),
                 Token('.npy')]),
        test_bins, test_sets, test_percents, test_tests, test_iterations, test_datapoints, test_average_bins
    )
]
# The distance between points on the x-axis, measured by the difference between x-ticks
test_shift = 0.25


# Plotting
##############################


test_plot_pat = Pattern([Token('set', test_sets.values), Token('percent', test_percents.values),
                       Token('iteration', test_iterations.values), Token('.png')])
test_plot_title_pat = Pattern([Token('Syn ', test_sets.values), Token(' Percent ', test_percents.values),
                             Token(' Iteration', test_iterations.values)])



# Compile data
##############################


# plot_options.append(PlotOptions(test_points, test_shift, test_plot_dir, test_plot_pat, test_plot_title_pat,
#                                 test_verbose, test_use_numpy_histogram, test_use_symmetric_percent_error,
#                                 test_calculate_std_dev_using_datapoints, test_normalize_std_dev, test_normalize_y_axis,
#                                 test_use_symlog_yscale))



############################################################
# Plot all manual entries
############################################################


if not use_default_options:
    for option in plot_options:
        plot_manual(option.points, option.shift, option.plot_dir, option.plot_pat, option.plot_title_pat, option.verbose,
                    option.use_numpy_histogram, option.use_symmetric_percent_error,
                    option.calculate_std_dev_using_datapoints, option.normalize_std_dev,
                    option.normalize_y_axis, option.use_symlog_yscale)
