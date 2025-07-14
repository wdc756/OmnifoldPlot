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
    plot_syn_error_by_bin = True



# SEI (Synthetic Error by Iteration) plotting options
############################################################



if 'SEI options':
    
    # SEI Main control vars
    ##############################
    
    
    # Bool to graph unweighted synthetic data
    sei_plot_unweighted = False
    sei_color = '#2ecc71'
    
    # Bool to graph omnifold-weighted syn data
    sei_plot_weighted = True
    sei_weighted_color = '#3498db'
    
    # Bool to plot pre-weighted omnifold-weighted syn data
    sei_plot_re_weighted = True
    sei_re_weighted_color = '#e74c3c'
    
    # The distance between any two data points per iteration, if there are any
    # This just shifts the iteration points on the x-axis to make the plot readable
    sei_shift_distance = 0.5
    
    # This bool will re-use the above settings to graph the combined results (average all bins)
    sei_plot_combined = True
    
    # Bool to graph standard deviation error bars
    sei_graph_error_bars = True
    
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
    
    
    # SEI File vars
    ##############################
    
    
    sei_nat_data_dir = 'mock'
    sei_syn_data_dir = 'mock'
    sei_weight_dir = 'weights'
    sei_re_weight_dir = 're_weights'
    sei_plot_dir = 'plots/iterations'
    
    sei_nat_file_pat = Pattern([
        Token('mockdata.nat.Logweighted2.N150000.root'),
    ])
    sei_syn_file_pat = Pattern([
        Token('mockdata.syn', 1, Iter(1, sei_num_syn_datasets, 1)),
        Token('.', 1, Iter(1, sei_num_percent_deviations, 1)),
        Token('Percent.Logweighted2.N150000.root')
    ])
    sei_weight_file_pat = Pattern([
        Token('Syn', 1, Iter(1, sei_num_syn_datasets, 1)),
        Token('_', 1, Iter(1, sei_num_percent_deviations, 1)),
        Token('Percent_Test', 1, Iter(1, sei_num_tests, 1)),
        Token('.npy')
    ])
    sei_re_weight_file_pat = Pattern([
        Token('Syn', 1, Iter(1, sei_num_syn_datasets, 1)),
        Token('_', 1, Iter(1, sei_num_percent_deviations, 1)),
        Token('Percent_Test', 1, Iter(1, sei_num_tests, 1)),
        Token('.npy')
    ])
    
    # Automate bin plot names
    bins_str = []
    for i in range(sei_bins_start, sei_bins_end, sei_bins_step):
        bins_str.append(str(i) + '-' + str(i + sei_bins_step))
    sei_plot_file_pat = Pattern([
        Token('syn', 1, Iter(1, sei_num_syn_datasets, 1)),
        Token('.', 1, Iter(1, sei_num_percent_deviations, 1)),
        Token('Percent.'),
        Token(bins_str),
        Token('GeV.png')
    ])

    # Automate alternate file pat name when sei_plot_combined == True
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
        sei_plot_unweighted, sei_color,
        sei_plot_weighted, sei_weighted_color,
        sei_plot_re_weighted, sei_re_weighted_color,
        sei_shift_distance,
        sei_plot_combined,
        sei_graph_error_bars,
        sei_bins_start, sei_bins_end, sei_bins_step,
        sei_num_syn_datasets, sei_num_percent_deviations, sei_num_tests, sei_num_iterations, sei_num_datapoints,
        sei_nat_data_dir, sei_syn_data_dir, sei_weight_dir, sei_re_weight_dir, sei_plot_dir,
        sei_nat_file_pat, sei_syn_file_pat, sei_weight_file_pat, sei_re_weight_file_pat,
        sei_plot_file_pat,
    ) 



# SEB (Synthetic Error by Bin) plotting options
############################################################



if 'SEB options':

    # SEB Main control vars
    ##############################


    # Bool to graph omnifold-weighted syn data
    seb_plot_weighted = True
    seb_weighted_color = '#3498db'

    # Bool to plot pre-weighted omnifold-weighted syn data
    seb_plot_re_weighted = True
    seb_re_weighted_color = '#e74c3c'

    # The distance between any two data points per iteration, if there are any
    # This just shifts the iteration points on the x-axis to make the plot readable
    seb_shift_distance = 5

    # Bool to graph standard deviation error bars
    seb_graph_error_bars = True

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


    # SEB File vars
    ##############################


    seb_nat_data_dir = 'mock'
    seb_syn_data_dir = 'mock'
    seb_weight_dir = 'weights'
    seb_re_weight_dir = 're_weights'
    seb_plot_dir = 'plots/bins'

    seb_nat_file_pat = Pattern([
        Token('mockdata.nat.Logweighted2.N150000.root'),
    ])
    seb_syn_file_pat = Pattern([
        Token('mockdata.syn', 1, Iter(1, seb_num_syn_datasets, 1)),
        Token('.', 1, Iter(1, seb_num_percent_deviations, 1)),
        Token('Percent.Logweighted2.N150000.root')
    ])
    seb_weight_file_pat = Pattern([
        Token('Syn', 1, Iter(1, seb_num_syn_datasets, 1)),
        Token('_', 1, Iter(1, seb_num_percent_deviations, 1)),
        Token('Percent_Test', 1, Iter(1, seb_num_tests, 1)),
        Token('.npy')
    ])
    seb_re_weight_file_pat = Pattern([
        Token('Syn', 1, Iter(1, seb_num_syn_datasets, 1)),
        Token('_', 1, Iter(1, seb_num_percent_deviations, 1)),
        Token('Percent_Test', 1, Iter(1, seb_num_tests, 1)),
        Token('.npy')
    ])

    # Automate iteration names
    iterations_str = []
    for i in seb_iterations_to_plot:
        iterations_str.append(str(i))
    seb_plot_file_pat = Pattern([
        Token('syn', 1, Iter(1, seb_num_syn_datasets, 1)),
        Token('.', 1, Iter(1, seb_num_percent_deviations, 1)),
        Token('Percent.Iteration'),
        Token(iterations_str),
        Token('.png')
    ])


    # SEB compile options
    ##############################


    # In theory, you (the user) should never have to change this, so don't touch unless you know what you're doing
    plot_seb_options = data.PlotSEBOptions(
        seb_plot_weighted, seb_weighted_color,
        seb_plot_re_weighted, seb_re_weighted_color,
        seb_shift_distance,
        seb_graph_error_bars,
        seb_bins_start, seb_bins_end, seb_bins_step,
        seb_num_syn_datasets, seb_num_percent_deviations, seb_num_tests, seb_num_iterations, seb_num_datapoints,
        seb_iterations_to_plot,
        seb_nat_data_dir, seb_syn_data_dir, seb_weight_dir, seb_re_weight_dir, seb_plot_dir,
        seb_nat_file_pat, seb_syn_file_pat, seb_weight_file_pat, seb_re_weight_file_pat,
        seb_plot_file_pat,
    )



########################################################################################################################

# Execution

########################################################################################################################



if plot_syn_error_by_iteration or plot_all:
    plot.plot_sei(plot_sei_options, data_dir)


if plot_syn_error_by_bin or plot_all:
    plot.plot_seb(plot_seb_options, data_dir)

