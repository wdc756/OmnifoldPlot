# This is the main launch file, containing settings, documentation, and execution calls



########################################################################################################################

# Imports

########################################################################################################################



from startrace import *

from src.util.imports import check_imports
if not check_imports():
    exit(1)

import src.util.data as data
import src.util.plot as plot



########################################################################################################################

# Settings

########################################################################################################################



############################################################
# Main settings
############################################################


# Note that this, along with other if 'str' statements, are just to allow IDEs to collapse code for readability
if 'Main settings':

    # Overrides all bools in this section to graph everything when set to True
    plot_all = False

    # Will be passed to other functions, by default returns the dir above this file (omnifold_plot.py)
    # To set a custom base dir, pass it into the function as a string
    base_dir = data.get_base_dir()
    # Where your data files are kept
    data_dir = 'data'

    # Plots % error of syn data compared to nat data by averaging selected iterations
    """
    y-axis: % error
    x-axis: iteration
    
    Plotted by bin, averaging selected iterations for each bin, percent deviation, and syn dataset. You may also 
    plot unweighted syn error and re-weighted syn error on the same graph. This plot will average all tests!
    
    This will mainly give insight on weighted vs re-weighted performance, focusing on iteration performance.
    """
    plot_syn_error_by_iteration = True

    # Plots % error of syn data compared to nat data by bin and iteration
    """
    y-axis: % error
    x-axis: bin
    
    Plotted by iteration, averaging all bins for each of the selected iterations. You may also plot unweighted syn error 
    and re-weighted syn error on the same graph. This plot will average all tests!
    
    This plot will mainly give insight on weighted vs re-weighted performance, focusing on bin performance
    """
    plot_syn_error_by_bin = False



# SEI (Synthetic Error by Iteration) plotting options
############################################################



if 'SEI options':
    
    # SEI Main control vars
    ##############################

    
    # Binning vars
    sei_bins_start = 0
    sei_bins_end = 100
    sei_bins_step = 20
    
    # Omnifold training vars
    sei_num_syn_datasets = 2
    sei_num_percent_deviations = 5
    sei_num_tests = 10
    sei_num_iterations = 5
    sei_num_datapoints = 150000

    # Bins to plot - note each value is the index, not the actual bin values
    # To automatically plot all, uncomment the for loop below
    sei_bins_to_plot = []
    for i in range(int((sei_bins_end - sei_bins_start) / sei_bins_step)):
        sei_bins_to_plot.append(i)

    # The distance between any two data points per iteration, if there are any
    # This just shifts the iteration points on the x-axis to make the plot readable
    sei_shift_distance = 0.5

    # To plot std. deviation error bars
    sei_plot_error_bars = True

    # This bool will re-use the above settings to graph the combined results (average all bins)
    sei_plot_combined = False

    # Bool to graph standard deviation error bars
    sei_graph_error_bars = True


    # Nat data
    ##############################


    sei_nat_dir = 'mock'
    sei_nat_file_pat = Pattern([
        Token('mockdata.nat.Logweighted2.N150000.root'),
    ])


    # Datapoints
    ##############################


    sei_points = []

    # Weighted synthetic data
    sei_points.append(data.Point(
        'weighted syn',  # Title, for the legend
        '#3498db',  # Point color
        sei_plot_error_bars,
        '#3498db',  # Error bar color
        0,  # Shift amount, leave 0 to be automatically set based on sei_shift_distance
        'weights',  # Dir to get files from
        Pattern([
            Token('Syn', 1, Iter(1, sei_num_syn_datasets, 1)),
            Token('_', 1, Iter(1, sei_num_percent_deviations, 1)),
            Token('Percent_Test', 1, Iter(1, sei_num_tests, 1)),
            Token('.npy')
        ]),  # Weights file pattern
        sei_num_tests,  # The number of tests, used to average when plotting
        sei_num_iterations  # The number of iterations, sometimes used to change plots
    ))

    # Re-Weighted synthetic data
    sei_points.append(data.Point(
        're-weighted syn',
        '#e74c3c',
        sei_plot_error_bars,
        '#e74c3c',
        0,
        're_weights',
        Pattern([
            Token('Syn', 1, Iter(1, sei_num_syn_datasets, 1)),
            Token('_', 1, Iter(1, sei_num_percent_deviations, 1)),
            Token('Percent_Test', 1, Iter(1, sei_num_tests, 1)),
            Token('.npy')
        ]),
        sei_num_tests,
        sei_num_iterations
    ))


    # Syn data
    ##############################


    sei_syn_dir = 'mock'
    sei_syn_file_pat = Pattern([
        Token('mockdata.syn', 1, Iter(1, sei_num_syn_datasets, 1)),
        Token('.', 1, Iter(1, sei_num_percent_deviations, 1)),
        Token('Percent.Logweighted2.N150000.root')
    ])


    # Plot file pattern
    ##############################


    sei_plot_dir = 'plots/bins'

    sei_plot_file_bins_str = []
    sei_bins = []
    for i in range(sei_bins_start, sei_bins_end + sei_bins_step, sei_bins_step):
        sei_bins.append(i)
    for i in range(len(sei_bins_to_plot)):
        sei_plot_file_bins_str.append(str(sei_bins[sei_bins_to_plot[i]]) + '-' + str(sei_bins[sei_bins_to_plot[i] + 1]))

    sei_plot_file_pat = Pattern([
        Token('syn', 1, Iter(1, sei_num_syn_datasets, 1)),
        Token('.', 1, Iter(1, sei_num_percent_deviations, 1)),
        Token('Percent.'),
        Token(sei_plot_file_bins_str),
        Token('GeV.png')
    ])
    if sei_plot_combined:
        sei_plot_file_pat = Pattern([
            Token('syn', 1, Iter(1, sei_num_syn_datasets, 1)),
            Token('.', 1, Iter(1, sei_num_percent_deviations, 1)),
            Token('Percent.png')
        ])

    
    # SEI compile options
    ##############################
    
    
    # In theory, you (the user) should never have to change this, so don't touch unless you know what you're doing
    plot_sei_options = data.PlotSEIOptions(
        sei_nat_dir, sei_nat_file_pat,
        sei_points,
        sei_bins_start, sei_bins_end, sei_bins_step,
        sei_num_syn_datasets, sei_num_percent_deviations, sei_num_tests, sei_num_iterations, sei_num_datapoints,
        sei_bins_to_plot, sei_plot_combined,
        sei_syn_dir, sei_syn_file_pat,
        sei_plot_dir, sei_plot_file_pat
    )



# SEB (Synthetic Error by Bin) plotting options
############################################################



if 'SEB options':

    # Shared settings
    ##############################


    # Binning vars
    seb_bins_start = 0
    seb_bins_end = 100
    seb_bins_step = 20

    # Omnifold training vars
    seb_num_syn_datasets = 2
    seb_num_percent_deviations = 5
    seb_num_tests = 10
    seb_num_iterations = 5
    seb_num_datapoints = 150000

    # Iterations to graph - to use all, either manually enter or remove all entries and uncomment the for loop
    seb_iterations_to_plot = [5]
    # for i in range(1, seb_num_iterations + 1):
    #     seb_iterations_to_plot.append(i)

    # The distance between any 2 datapoints
    # Note the individual values can be set by hand, but it's easier to do this one because it will be automatically applied
    seb_shift_distance = 2.5

    # If std. error bars should be shown
    seb_plot_error_bars = True


    # Nat data
    ##############################


    seb_nat_data_dir = 'mock'
    seb_nat_file_pat = Pattern([
        Token('mockdata.nat.Logweighted2.N150000.root'),
    ])


    # Datapoints
    ##############################


    seb_points = []

    # Weighted synthetic data
    seb_points.append(data.Point(
        'weighted syn',
        '#3498db',
        seb_plot_error_bars,
        '#3498db',
        0,
        'weights',
        Pattern([
            Token('Syn', 1, Iter(1, seb_num_syn_datasets, 1)),
            Token('_', 1, Iter(1, seb_num_percent_deviations, 1)),
            Token('Percent_Test', 1, Iter(1, seb_num_tests, 1)),
            Token('.npy')
        ]),
        seb_num_tests,
        seb_num_iterations
    ))

    # Re-Weighted synthetic data
    seb_points.append(data.Point(
        're-weighted syn',
        '#e74c3c',
        seb_plot_error_bars,
        '#e74c3c',
        0,
        're_weights',
        Pattern([
            Token('Syn', 1, Iter(1, seb_num_syn_datasets, 1)),
            Token('_', 1, Iter(1, seb_num_percent_deviations, 1)),
            Token('Percent_Test', 1, Iter(1, seb_num_tests, 1)),
            Token('.npy')
        ]),
        seb_num_tests,
        seb_num_iterations
    ))


    # Syn data
    ##############################


    seb_syn_dir = 'mock'
    seb_syn_file_pat = Pattern([
        Token('mockdata.syn', 1, Iter(1, seb_num_syn_datasets, 1)),
        Token('.', 1, Iter(1, seb_num_percent_deviations, 1)),
        Token('Percent.Logweighted2.N150000.root')
    ])


    # Plot file pattern
    ##############################


    seb_plot_dir = 'plots/bins'
    seb_plot_file_iterations_str = []
    for i in seb_iterations_to_plot:
        seb_plot_file_iterations_str.append(str(i))
    seb_plot_file_pat = Pattern([
        Token('syn', 1, Iter(1, seb_num_syn_datasets, 1)),
        Token('.', 1, Iter(1, seb_num_percent_deviations, 1)),
        Token('Percent.Iteration'),
        Token(seb_plot_file_iterations_str),
        Token('.png')
    ])


    # SEB compile options
    ##############################


    # In theory, you (the user) should never have to change this, so don't touch unless you know what you're doing
    plot_seb_options = data.PlotSEBOptions(
        seb_nat_data_dir, seb_nat_file_pat,
        seb_points,
        seb_bins_start, seb_bins_end, seb_bins_step,
        seb_num_syn_datasets, seb_num_percent_deviations, seb_num_tests, seb_num_iterations, seb_num_datapoints,
        seb_iterations_to_plot,
        seb_syn_dir, seb_syn_file_pat,
        seb_plot_dir, seb_plot_file_pat,
    )



########################################################################################################################

# Execution

########################################################################################################################



if plot_syn_error_by_iteration or plot_all:
    plot.plot_sei(plot_sei_options, data_dir)


if plot_syn_error_by_bin or plot_all:
    plot.plot_seb(plot_seb_options, data_dir)

