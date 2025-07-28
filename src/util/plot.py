# Written by William Dean Coker
# This file contains methods to plot omnifold data according to settings from omnifold_plot.py

# Plotting Steps:
"""
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

from collections.abc import Iterable
import copy

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from src.util.data import *



########################################################################################################################

# Back-end functions

########################################################################################################################



############################################################
# Utilities
############################################################


# To be re-worked eventually
##############################


def _get_depth(obj) -> int:
    if isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
        try:
            first = next(iter(obj))
        except StopIteration:
            return 1
        return 1 + _get_depth(first)
    return 0

def _print_depth(message: str, depth: int):
    message = '    ' * depth + message
    print(message)

def _recursive_print_array_stats(array: np.ndarray, name: str, depth: int, top=True):
    if depth == 1:
        if top:
            print(name + ' stats')
            print('  shape:', array.shape)
        print('  array: ', array)
        print('  min:', np.min(array))
        print('  max:', np.max(array))
        print('  mean:', np.mean(array))
        print('  median:', np.median(array))
        print('  std:', np.std(array))
        print('  sum:', np.sum(array))
        print('  num_nan:', np.count_nonzero(np.isnan(array)))
        print('  num_inf:', np.count_nonzero(np.isinf(array)))
    else:
        if top:
            print(name + ' stats')
            print('shape:', array.shape)
        for i in range(array.shape[0]):
            _recursive_print_array_stats(array[i], name, depth - 1, False)


# Fully working
##############################


def _recursive_index(data: np.ndarray, indexes: list[int]):
    if indexes is None or len(indexes) == 0:
        return data
    else:
        if len(indexes) == 1:
            return data[indexes[0]]
        else:
            return _recursive_index(data[indexes[0]], indexes[1:])



############################################################
# Data loading and processing
############################################################



def _fill_omnifold_weight(point: Point):
    # Create an empty array so we can just fill it and not worry about .append()
    point.omnifold_weight = np.zeros((len(point.sets.values), len(point.percents.values), len(point.tests.values),
                                      len(point.iterations.values), point.num_datapoints))

    # Omnifold generates 1 weight file for each set, percent, and test, so loop over all values
    # Note that here we use enumerate() because all Dimension.values arrays can hold non-linear indexes
    for s_idx, set in enumerate(point.sets.values):
        for p_idx, percent in enumerate(point.percents.values):
            for t_idx, test in enumerate(point.tests.values):
                # Put the weights in a temp container because we need to reformat them
                weights = get_omnifold_weights(get_file_path([point.data_dir, point.weight_dir,
                                                              point.weight_pat.get_pattern()]))

                # The raw file stores [iteration][step], but we only need step=1 to plot with, so loop over
                #   iterations.
                for i_idx, iteration in enumerate(point.iterations.values):
                    # Throw an error if the datapoints don't match exactly, because if we don't, the program will
                    #   likely just keep going and give us faulty plots
                    if weights.shape[-1] != point.num_datapoints:
                        raise ValueError(
                            'Error: ' + point.name + '.weight_pat.weights does not match point.num_datapoints'
                        )

                    point.omnifold_weight[s_idx][p_idx][t_idx][i_idx] = (
                            weights[iteration - 1][1]
                    )

                # Throw an error if point runs out of files before we're done filling the arrays, so the user knows
                #   to change the file pattern
                if (not point.weight_pat.increment() and set != point.sets.values[-1] and
                        percent != point.percents.values[-1] and test != point.tests.values[-1]):
                    raise ValueError('Error: ' + point.name + '.weight_pat could not increment')

def _recursive_process_binned_sum(bins: np.linspace, data: np.ndarray, weight: np.ndarray, verbose: int, depth: int,
                                  use_numpy_histogram = False):
    if depth == 1:
        # Base case: weight is 1D array, so we can use data to bin and apply histogram
        bin_sums = np.zeros(len(bins) - 1)

        # Since numpy histogram() can sometimes return trash values, we have a toggle
        if use_numpy_histogram:
            bin_sums, _ = np.histogram(data, bins, weights=weight)

            if verbose > 2:
                _print_depth('NP bin sums:' + str(bin_sums), depth)
                _print_depth('NP histogram bins: ' + str(_), depth)
        else:
            # Manual historgram
            for i in range(len(bins) - 1):
                mask = (data >= bins[i]) & (data < bins[i + 1])
                bin_sums[i] = np.sum(weight[mask])

            if verbose > 2:
                _print_depth("Manual bin sums:" + str(bin_sums), depth)
                _print_depth("Manual histogram bins: " + str(bins), depth)

        return bin_sums
    elif _get_depth(data) > 1:
        # Recursive case 1: go deeper into data and weights
        result = []
        for i in range(weight.shape[0]):
            binned = _recursive_process_binned_sum(bins, data[i], weight[i], verbose, depth - 1, use_numpy_histogram)
            result.append(binned)
        return np.array(result)
    else:
        # Recursive case: process each slice along the first dimension
        result = []
        for i in range(weight.shape[0]):
            binned = _recursive_process_binned_sum(bins, data, weight[i], verbose, depth - 1, use_numpy_histogram)
            result.append(binned)
        return np.array(result)

def _recursive_process_percent_error(nat_weight: np.ndarray, syn_weight: np.ndarray, depth: int,
                                     epsilon: float = 0.000001, symmetric = False):
    if depth == 1:
        # Base case: we're at the bin level, calculate the percent error

        # Symmetric percent error is sometimes useful, so we have a toggle for it
        if symmetric:
            denominator = (nat_weight + syn_weight) / 2
            denominator = np.where(np.abs(denominator) < epsilon, epsilon, denominator)
            percent_error = 100 * (nat_weight - syn_weight) / denominator
        else:
            denominator = nat_weight + epsilon
            percent_error = 100 * (syn_weight - nat_weight) / denominator

        return percent_error
    else:
        # Recursive case: go deeper into nested arrays
        result = []
        for syn_w in syn_weight:
            error = _recursive_process_percent_error(nat_weight, syn_w, depth - 1, epsilon, symmetric)
            result.append(error)
        return np.array(result)

def _recursive_process_std_dev_datapoint(bins: np.linspace, data: np.ndarray, weight: np.ndarray, depth: int, normalize = False):
    if depth == 1:
        # Get bin indices for each data point
        bin_indices = np.digitize(data, bins) - 1
        bin_indices = np.clip(bin_indices, 0, len(bins) - 2)

        # Initialize std dev array for each bin
        std_devs = np.zeros(len(bins) - 1)

        # Calculate std dev for each bin
        for bin_idx in range(len(bins) - 1):
            mask = bin_indices == bin_idx
            weights_in_bin = weight[mask]

            if len(weights_in_bin) > 1:
                if normalize:
                    # Normalize weights within this bin
                    weights_in_bin = weights_in_bin / (np.sum(weights_in_bin) + np.finfo(float).eps)

                # Calculate standard deviation of normalized weights
                std_devs[bin_idx] = np.std(weights_in_bin, ddof=1)
            else:
                std_devs[bin_idx] = 0.0  # Single point or empty bin

        return std_devs * 100.0 # convert to percent
    elif _get_depth(data) > 1:
        # Recursive case 1: go deeper into data and weights
        result = []
        for i in range(weight.shape[0]):
            binned = _recursive_process_std_dev_datapoint(bins, data[i], weight[i], depth - 1, normalize)
            result.append(binned)
        return np.array(result)
    else:
        # Recursive case 2: go deeper into weight arrays
        result = []
        for w in weight:
            binned = _recursive_process_std_dev_datapoint(bins, data, w, depth - 1, normalize)
            result.append(binned)
        return np.array(result)

def _process_std_dev(data: np.ndarray, dimension_averages: list[bool], normalize = False):
    result = data.copy()
    actual_dims = result.ndim


    if normalize and result.size > 0:
        # Normalize each bin separately to preserve relative variations
        bin_sums = np.sum(result, axis=tuple(range(actual_dims-1)))  # Sum all but the last dimension
        bin_sums = np.expand_dims(bin_sums, axis=tuple(range(actual_dims-1)))  # Reshape for broadcasting
        result = result / (bin_sums + np.finfo(float).eps)  # Add small epsilon to avoid division by zero

    # Process dimensions from innermost to outermost
    for i in range(min(len(dimension_averages), actual_dims) - 1, -1, -1):
        if dimension_averages[i]:
            if result.shape[i] > 1:
                result = np.std(result, axis=i, ddof=1)
            else:
                result = np.squeeze(result, axis=i)

    return result

def _average_data(data: np.ndarray, dimension_averages: list[bool]):
    result = data.copy()

    # Get the actual number of dimensions in our data
    actual_dims = result.ndim

    # Process other dimensions from innermost to outermost (reverse order)
    # Only process dimensions that actually exist in our data
    for i in range(min(len(dimension_averages), actual_dims) - 1, -1, -1):
        if dimension_averages[i]:
            result = np.average(result, axis=i)

    return result



############################################################
# Plotting
############################################################



def _calculate_shifts(shift: float, points: list[Point]):
    for p in range(len(points)):
        point = points[p]

        # Only shift if the user hasn't set it manually
        if point.shift == 0:
            point.shift = shift * (p - len(points) // 2)

def _get_y_axis_lim(points: list[Point]):
    # Set a default value so we have a good baseline
    y_max = np.max(points[0].percent_error_avg) + np.max(np.abs(points[0].std_dev_avg))
    y_min = np.min(points[0].percent_error_avg) - np.max(np.abs(points[0].std_dev_avg))

    # Loop over all other points to find true min/max
    for point in points[1:]:
        p_p_max = np.max(point.percent_error_avg)
        p_p_min = np.min(point.percent_error_avg)
        std_dev_abs = np.abs(point.std_dev_avg)
        p_s_max = np.max(std_dev_abs)

        y_max = max(y_max, p_p_max + p_s_max)
        y_min = min(y_min, p_p_min - p_s_max)

    return y_max, y_min

def _get_x_axis(dimensions: list[Dimension], x_ticks):
    x_tick = []
    x_values = []
    x_label = ''

    # Loop over all dimensions, outermost to innermost, setting the new x_values label accordingly
    for i in range(len(dimensions)):
        if not dimensions[i].average:
            x_tick = x_ticks[i]
            x_values = dimensions[i].values
            x_label = dimensions[i].name

    return x_tick, x_values, x_label

def _recursive_plot_points(points: list[Point], plot_dir: str, plot_pat: Pattern, plot_title_pat: Pattern,
                           x_ticks: np.ndarray, x_values: np.ndarray, x_label: str, normalize_y_axis: bool,
                           y_max: float, y_min: float, depth: int, indexes: list[int],
                           _top = True, use_symlog_yscale = False):
    if depth == 1:
        # Base case: we have a 1D array to plot for the current points
        for point in points:
            percent_error = _recursive_index(point.percent_error_avg, indexes)
            if len(x_values) != len(percent_error):
                raise IndexError(f'Error: cannot plot points with unmatched pairs of numbers (x={x_values.shape}, y={percent_error.shape})')
            if point.plot_error_bars:
                # Take the absolute value here to avoid an error with negative std deviations
                std_dev = np.abs(_recursive_index(point.std_dev_avg, indexes))
                plt.errorbar(x_values + point.shift, percent_error, yerr=std_dev, marker='o', markersize=4,
                             linestyle='none', color=point.color, ecolor=point.error_color, label=point.name, capsize=2)
            else:
                plt.plot(x_values + point.shift, percent_error, marker='o', markersize=4, linestyle='none',
                         color=point.color, label=point.name)

        # Plot extras and save
        plt.xticks(x_ticks)
        plt.xlabel(x_label)
        if use_symlog_yscale:
            plt.yscale('symlog', linthresh=1e1)
        plt.ylabel('Percent Error')
        if normalize_y_axis:
            plt.ylim(y_min, y_max)
        plt.grid(axis='x')
        plt.title(plot_title_pat.get_pattern())
        plt.legend()
        plt.savefig(get_file_path([plot_dir, plot_pat.get_pattern()]))
        plt.show()
        plt.close()

        # Increment plot pat
        if not plot_pat.increment():
            raise ValueError('Error: plot.plot_pat could not increment')
        if not plot_title_pat.increment():
            raise ValueError('Error: plot.plot_title_pat could not increment')
    else:
        # Recursive case: we have an ND array and need to go deeper
        current = _recursive_index(points[0].percent_error_avg, indexes)
        for i in range(len(current)):
            # Insert try-catch here with _top to not throw any errors after the last plot has been made
            try:
                _recursive_plot_points(points, plot_dir, plot_pat, plot_title_pat, x_ticks, x_values, x_label,
                                       normalize_y_axis, y_max, y_min, depth - 1,
                                       indexes + [i], False, use_symlog_yscale)
            except ValueError as e:
                if _top and i == len(current) - 1:
                    return
                raise ValueError(e)



########################################################################################################################

# User functions

########################################################################################################################



def get_plot_patterns(sets: Dimension, percents: Dimension, tests: Dimension, iterations: Dimension, average_bins, bins):
    path_tokens = []

    # Determine which variable will be the x-axis
    # Priority order: bins (lowest) -> iterations -> tests -> percents -> sets (highest)
    is_x_axis = {
        'bins': not average_bins,
        'iterations': average_bins and not iterations.average,
        'tests': average_bins and iterations.average and not tests.average,
        'percents': average_bins and iterations.average and tests.average and not percents.average,
        'sets': average_bins and iterations.average and tests.average and percents.average and not sets.average
    }

    # Add tokens based on whether they're averaged or x-axis
    if not sets.average and not is_x_axis['sets']:
        path_tokens.append(Token('Set', sets.values))

    if not percents.average and not is_x_axis['percents']:
        path_tokens.append(Token('Percent', percents.values))

    if not tests.average and not is_x_axis['tests']:
        path_tokens.append(Token('Test', tests.values))

    if not iterations.average and not is_x_axis['iterations']:
        path_tokens.append(Token('Iteration', iterations.values))

    if not average_bins and not is_x_axis['bins']:
        bins_str = [f"{bins[i]}-{bins[i + 1]}" for i in range(len(bins) - 1)]
        path_tokens.append(Token('Bin'))
        path_tokens.append(Token(bins_str))

    path_tokens.append(Token('.png'))
    # Deep copy here to avoid re-using the exact same tokens (overwrite/race)
    title_tokens = copy.deepcopy(path_tokens[:-1])
    return path_tokens, title_tokens



def plot_manual(points: list[Point], shift: float, plot_dir: str, plot_pat: Pattern, plot_title_pat: Pattern, verbose = 1,
                use_numpy_histogram = False, use_symmetric_percent_error = False,
                calculate_std_dev_using_datapoints = False, normalize_std_dev = False,
                normalize_y_axis = False, use_symlog_yscale = False):

    # Data checks & setup
    ##############################


    if len(points) == 0:
        raise ValueError('Error: plot.points cannot be empty')


    # Data gathering and processing
    ##############################


    for point in points:
        if verbose > 0:
            print('processing ' + point.name)

        # Gather data
        ##############################


        point.nat_data, point.nat_weight = get_nat_data_and_weights(get_file_path([point.data_dir, point.nat_dir,
                                                                                   point.nat_file_name]))
        if verbose > 2:
            _recursive_print_array_stats(point.nat_data, 'nat_data', _get_depth(point.nat_data))
        elif verbose > 1:
            print('nat data shape: ', point.nat_data.shape)

        point.syn_data = np.empty((len(point.sets.values), len(point.percents.values), point.num_datapoints))
        point.syn_weight = np.empty((len(point.sets.values), len(point.percents.values), point.num_datapoints))
        # Note that here we use enumerate() because all Dimension.values arrays can hold non-linear indexes
        for s_idx, set in enumerate(point.sets.values):
            for p_idx, percent in enumerate(point.percents.values):
                syn_data, syn_weight = get_syn_data_and_weights(get_file_path([point.data_dir,
                                                                               point.syn_dir,point.syn_pat.get_pattern()]))
                point.syn_data[s_idx - 1][p_idx - 1] = syn_data
                point.syn_weight[s_idx - 1][p_idx - 1] = syn_weight
        if verbose > 2:
            _recursive_print_array_stats(point.syn_data, 'syn_data', _get_depth(point.syn_data))
        elif verbose > 1:
            print('syn data shape: ', point.syn_data.shape)

        _fill_omnifold_weight(point)
        if verbose > 2:
            _recursive_print_array_stats(point.omnifold_weight, 'omnifold_weight', _get_depth(point.omnifold_weight))
        elif verbose > 1:
            print('omnifold weights shape: ', point.omnifold_weight.shape)


        # Process data
        ##############################


        averages = [point.sets.average, point.percents.average, point.tests.average, point.iterations.average,
                    point.average_bins]
        alt_averages = [point.sets.average, point.percents.average, point.iterations.average, point.average_bins,
                        point.tests.average]


        point.nat_sum_weight = _recursive_process_binned_sum(point.bins, point.nat_data, point.nat_weight, verbose,
                                                             _get_depth(point.nat_weight), use_numpy_histogram)
        if verbose > 2:
            _recursive_print_array_stats(point.nat_sum_weight, 'sum_nat_weight', _get_depth(point.nat_sum_weight))
        elif verbose > 1:
            print('nat sum weight shape: ', point.nat_sum_weight.shape)
        point.sum_omnifold_weight = _recursive_process_binned_sum(point.bins, point.syn_data, point.omnifold_weight, verbose,
                                                                  _get_depth(point.omnifold_weight), use_numpy_histogram)
        if verbose > 2:
            _recursive_print_array_stats(point.sum_omnifold_weight, 'sum_omnifold_weight', _get_depth(point.sum_omnifold_weight))
        elif verbose > 1:
            print('sum omnifold weight shape: ', point.sum_omnifold_weight.shape)

        point.percent_error = _recursive_process_percent_error(point.nat_sum_weight, point.sum_omnifold_weight,
                                                               _get_depth(point.sum_omnifold_weight), use_symmetric_percent_error)
        if verbose > 2:
            _recursive_print_array_stats(point.percent_error, 'percent_error', _get_depth(point.percent_error))
        elif verbose > 1:
            print('percent error shape: ', point.percent_error.shape)

        if calculate_std_dev_using_datapoints:
            point.std_dev = _recursive_process_std_dev_datapoint(point.bins, point.syn_data, point.omnifold_weight,
                                                                 _get_depth(point.omnifold_weight), normalize_std_dev)
        else:
            std_dev_percent_error = np.transpose(point.percent_error, (0, 1, 3, 4, 2))
            point.std_dev = _process_std_dev(std_dev_percent_error, alt_averages, normalize_std_dev)
        if verbose > 2:
            _recursive_print_array_stats(point.std_dev, 'std_dev', _get_depth(point.std_dev))
        elif verbose > 1:
            print('std. deviation shape: ', point.std_dev.shape)

        point.percent_error_avg = _average_data(point.percent_error, averages)
        if verbose > 2:
            _recursive_print_array_stats(point.percent_error_avg, 'percent_error_avg', _get_depth(point.percent_error_avg))
        elif verbose > 1:
            print('percent error avg shape: ', point.percent_error_avg.shape)

        if calculate_std_dev_using_datapoints:
            point.std_dev_avg = _average_data(point.std_dev, averages)
        else:
            point.std_dev_avg = point.std_dev

        if verbose > 2:
            _recursive_print_array_stats(point.std_dev_avg, 'std_dev_avg', _get_depth(point.std_dev_avg))
        elif verbose > 1:
            print('std. deviation avg shape: ', point.std_dev_avg.shape)

        if verbose > 1:
            print('\n')


    # Plot data
    ##############################


    # Make sure points don't have differing depths or average dimensions
    depths = []
    for point in points:
        depths.append(_get_depth(point.percent_error_avg))
    unique_depths = np.unique(depths, axis=0)
    if len(unique_depths) > 1:
        raise ValueError('Error: points have different depths (average flags)')

    # Get x-axis values
    bin_centers = (points[0].bins[:-1] + points[0].bins[1:]) / 2
    bins_dim = Dimension(points[0].average_bins, 'Bins', bin_centers)
    dimensions = [points[0].sets, points[0].percents, points[0].iterations, bins_dim, points[0].tests]
    x_ticks = [points[0].sets.values, points[0].percents.values, points[0].iterations.values, points[0].bins,
               points[0].tests.values]
    x_tick, x_values, x_label = _get_x_axis(dimensions, x_ticks)
    x_values = np.asarray(x_values)
    if len(x_values) < 2:
        raise ValueError('Error: plot.points has averaged too many dimensions (no x-axis values)')
    if verbose > 1:
        print('Plot over ' + x_label + ' ' + str(x_values))

    # Get y-axis limits to allow easier comparison - these will be truncated when normalize_y_axis = False
    y_max, y_min = _get_y_axis_lim(points)

    # Shift points by comparing to x-axis
    _calculate_shifts(shift * (float)(x_values[1] - x_values[0]), points)

    if verbose > 0:
        print('plotting points')
    _recursive_plot_points(points, plot_dir, plot_pat, plot_title_pat, x_tick, x_values, x_label, normalize_y_axis, y_max, y_min,
                           depths[0], [], use_symlog_yscale=use_symlog_yscale)



def plot_defaults(average_sets: bool, average_percents: bool, average_iterations: bool,
                  average_bins: bool, verbose: int):


    # Hack-y variables, don't touch unless you know what you're doing
    ##############################


    # Determines if the program should average over tests. If this is false, then it's almost required to enable
    #   calculate_std_dev_using_datapoints
    d_average_tests = True
    # Determines if the program should use the default numpy histogram() function. This is disabled by default because
    #   the function seemed unstable at high energy bins, returning a bunch of 0s
    d_use_numpy_histogram = False
    # Determines if the program should use symmetric percent error. It could be useful if you're ever dealing with
    #   nat bins that have no data, as this calculation will give you real values
    d_use_symmetric_percent_error = False
    # Determines if the program should calculate standard deviation using datapoints. By default this is false because
    #   calculating std dev over points was unstable, and deemed unnecessary in the long run
    d_calculate_std_dev_using_datapoints = False
    # Determines if the program should normalize wights when calculating standard deviation. This is false by default
    #   because it's not as accurate. Note: this will only have an effect when calculate_std_dev_using_datapoints is True
    d_normalize_std_dev = False
    # Determines if the program should normalize all plots' y-axes to the min and max values. This is set to True by
    #   default because it helps with comparison
    d_normalize_y_axis = True
    # Determines if the program should use symlog to scale on the y-axis. This is generally recommended when
    #   normalize_y_axis is true
    d_use_symlog_yscale = True


    # File vars
    ##############################


    d_data_dir = 'data'
    d_nat_dir = 'mock'
    d_nat_file_name = 'mockdata.nat.Logweighted2.N150000.root'
    d_syn_dir = 'mock'
    d_weight_dir = 'weights'
    d_re_weight_dir = 're_weights'
    d_spline_weight_dir = 'spline_weights'
    d_plot_dir = d_data_dir + '/plots'


    # Omnifold settings
    ##############################


    d_bins_start = 0
    d_bins_end = 100

    d_use_bins_step = False
    d_bins_step = 20
    d_bins_count = 10
    if d_use_bins_step:
        d_bins = np.arange(d_bins_start, d_bins_end + d_bins_step, d_bins_step)
    else:
        d_bins = np.linspace(d_bins_start, d_bins_end, d_bins_count + 1)
    if verbose > 2:
        print('bins:', d_bins)

    d_sets = Dimension(average_sets, 'Set', None, 1, 2, 1)
    d_percents = Dimension(average_percents, 'Percent', None, 1, 5, 1)
    d_tests = Dimension(d_average_tests, 'Test', None, 1, 10, 1)
    d_iterations = Dimension(average_iterations, 'Iteration', None, 1, 5, 1)
    d_datapoints = 150000


    # Points to plot
    ##############################


    d_plot_error_bars = True
    d_points = [
        Point(
            'weighted', '#5dade2', d_plot_error_bars, '#5dade2', 0.0, d_data_dir, d_nat_dir,
            d_nat_file_name, d_syn_dir, d_weight_dir,
            Pattern([Token('mockdata.syn', d_sets.values), Token('.', d_percents.values),
                     Token('Percent.Logweighted2.N' + str(d_datapoints) + '.root')]),
            Pattern([Token('Syn', d_sets.values), Token('_', d_percents.values), Token('Percent_Test', d_tests.values),
                     Token('.npy')]),
            d_bins, d_sets, d_percents, d_tests, d_iterations, d_datapoints, average_bins
        ),
        Point(
            're weighted', '#FB2C36', d_plot_error_bars, '#FB2C36', 0.0, d_data_dir, d_nat_dir,
            d_nat_file_name, d_syn_dir, d_re_weight_dir,
            Pattern([Token('mockdata.syn', d_sets.values), Token('.', d_percents.values),
                     Token('Percent.Logweighted2.N' + str(d_datapoints) + '.root')]),
            Pattern([Token('Syn', d_sets.values), Token('_', d_percents.values), Token('Percent_Test', d_tests.values),
                     Token('.npy')]),
            d_bins, d_sets, d_percents, d_tests, d_iterations, d_datapoints, average_bins
        ),
        Point(
            'spline weighted', '#31C950', d_plot_error_bars, '#31C950', 0.0, d_data_dir, d_nat_dir,
            d_nat_file_name, d_syn_dir, d_spline_weight_dir,
            Pattern([Token('mockdata.syn', d_sets.values), Token('.', d_percents.values),
                     Token('Percent.Logweighted2.N' + str(d_datapoints) + '.root')]),
            Pattern([Token('Syn', d_sets.values), Token('_', d_percents.values), Token('Percent_Test', d_tests.values),
                     Token('.npy')]),
            d_bins, d_sets, d_percents, d_tests, d_iterations, d_datapoints, average_bins
        )
    ]
    # The distance between points on the x-axis, measured by the difference between x-ticks
    d_shift = 0.2


    # Plotting
    ##############################


    path_tokens, title_tokens = get_plot_patterns(d_sets, d_percents, d_tests, d_iterations, average_bins, d_bins)

    d_plot_pat = Pattern(path_tokens)
    d_plot_title_pat = Pattern(title_tokens)
    if verbose > 1:
        print('Plot patterns: ')
    if verbose > 2:
        t_plot_pat = copy.deepcopy(d_plot_pat)
        t_plot_title_pat = Pattern(title_tokens)
        while True:
            print(t_plot_pat.get_pattern())
            print(t_plot_title_pat.get_pattern())
            if not t_plot_pat.increment() or not t_plot_title_pat.increment():
                break
    elif verbose > 1:
        print(d_plot_pat.get_pattern())
        print(d_plot_title_pat.get_pattern())

    plot_manual(d_points, d_shift, d_plot_dir, d_plot_pat, d_plot_title_pat, verbose,
                d_use_numpy_histogram, d_use_symmetric_percent_error, d_calculate_std_dev_using_datapoints,
                d_normalize_std_dev, d_normalize_y_axis, d_use_symlog_yscale)