# Written by William Dean Coker
# This is the main launch file, containing settings, documentation, and execution calls

########################################################################################################################

# Imports

########################################################################################################################



# imports.check_imports will attempt to import all packages and throw errors if any are missing
from util.imports import check_imports
if not check_imports():
    exit(1)

from util.plot import *



########################################################################################################################

# Default plotting

########################################################################################################################



# If you're using the default project structure as laid out in the ReadMe, then set these average bools and run the
#   program. If your setup is different, or you want more control, disable use_default_options and scroll down to the
#   manual plotting setup
average_sets = False
average_percents = False
average_iterations = False
average_bins = False
verbose = 2
show_plots = False

use_default_options = True

if use_default_options:
    plot_defaults(average_sets, average_percents, average_iterations, average_bins, show_plots, verbose)



########################################################################################################################

# Manual plotting

########################################################################################################################



############################################################
# Manual vars example
############################################################



ex_verbose = 2
ex_show_plots = False


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


# You can redefine these to be whatever you want, or get rid of them and set everything individually in the Point
#   constructor, I've just placed these here because I keep my files in the default locations (see ReadMe)
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
ex_bins_count = 10
if ex_use_bins_step:
    ex_bins = np.arange(ex_bins_start, ex_bins_end + ex_bins_step, ex_bins_step)
else:
    ex_bins = np.linspace(ex_bins_start, ex_bins_end, ex_bins_count + 1)
if ex_verbose > 2:
    print('bins:', ex_bins)

ex_sets = Dimension(
    # Determines if this dimension should be averaged
    average=False,
    # The display name (used if this value is the x-axis)
    name='Set',
    # The values (indexes) used - when set to None, it will be filled according to start, end, and step, otherwise the
    #   values can be entered manually, such as [2, 3, 7], which will access sets 2, 3, and 7. Note that all array
    #   indexing is handled automatically, so in the case of [2, 3, 7], the array will have a size [3], and will be
    #   indexed as such, while all other display references will use the inputted values
    values=None, start=1, end=2, step=1)
ex_percents = Dimension(False, 'Percent', None, 1, 5, 1)
ex_tests = Dimension(ex_average_tests, 'Test', None, 1, 10, 1)
ex_iterations = Dimension(False, 'Iteration', None, 5, 5, 1)
ex_average_bins = False
ex_datapoints = 150000


# Points to plot
##############################


ex_plot_error_bars = True
ex_points = [
    Point(
        # The display name for this point, used by the legend
        name='weighted syn',
        # The color of the point
        color='#5dade2',
        # Determines if this point should have error bars - Note that even if this is false, the program will still
        #   calculate std deviation, it just won't show it
        plot_error_bars=ex_plot_error_bars,
        # The color of the error bars, when shown
        error_color='#5dade2',
        # The horizontal shift of this point. When set to 0, the program will automatically calculate the position of
        #   this point to be relative to all other points, but when it's set to any other value, the program will leave it
        shift=0.0,
        # Where to get files from
        data_dir=ex_data_dir, nat_dir=ex_nat_dir, nat_file_name=ex_nat_file_name, syn_dir=ex_syn_dir, weight_dir=ex_weight_dir,
        # This is a File Pattern from StarTrace, for more info, see StarTrace documentation. The files to get syn data
        #   from
        syn_pat=Pattern([Token('mockdata.syn', ex_sets.values), Token('.', ex_percents.values),
                 Token('Percent.Logweighted2.N' + str(ex_datapoints) + '.root')]),
        # Files to get weights from
        weight_pat=Pattern([Token('Syn', ex_sets.values), Token('_', ex_percents.values), Token('Percent_Test', ex_tests.values),
                 Token('.npy')]),
        # Average/Values
        bins=ex_bins, sets=ex_sets, percents=ex_percents, tests=ex_tests, iterations=ex_iterations, num_datapoints=ex_datapoints, average_bins=average_bins
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
ex_shift = 0.2

# This function will generate both plot patterns, but you can also create your own and overwrite this call
ex_plot_tokens, ex_plot_title_tokens = get_plot_patterns(ex_sets, ex_percents, ex_tests, ex_iterations, ex_average_bins, ex_bins)
ex_plot_pat = Pattern(ex_plot_tokens)
ex_plot_title_pat = Pattern(ex_plot_title_tokens)



# Compile data
##############################


plot_options.append(PlotOptions(ex_points, ex_shift, ex_plot_dir, ex_plot_pat, ex_plot_title_pat, ex_verbose, ex_show_plots,
                                ex_use_numpy_histogram, ex_use_symmetric_percent_error,
                                ex_calculate_std_dev_using_datapoints, ex_normalize_std_dev, ex_normalize_y_axis,
                                ex_use_symlog_yscale))



############################################################
# Plot all manual entries
############################################################



if not use_default_options:
    for option in plot_options:
        plot_manual(option.points, option.shift, option.plot_dir, option.plot_pat, option.plot_title_pat, option.verbose,
                    option.use_numpy_histogram, option.use_symmetric_percent_error,
                    option.calculate_std_dev_using_datapoints, option.normalize_std_dev,
                    option.normalize_y_axis, option.use_symlog_yscale, option.show_plots)
