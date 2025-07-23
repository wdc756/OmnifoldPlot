# This file contains methods to plot omnifold data according to settings from omnifold_plot.py

from collections.abc import Iterable
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from src.util.data import *



########################################################################################################################

# Common functions

########################################################################################################################



def _calculate_shifts(shift: float, points: list[Point]):
    for p in range(len(points)):
        point = points[p]

        # Only shift if the user hasn't set it manually
        if point.shift == 0:
            point.shift = shift * (p - len(points) // 2)

def _get_depth(obj) -> int:
    if isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
        try:
            first = next(iter(obj))
        except StopIteration:
            return 1
        return 1 + _get_depth(first)
    return 0

def _recursive_index(data: np.ndarray, indexes: list[int]):
    if indexes is None or len(indexes) == 0:
        return data
    else:
        if len(indexes) == 1:
            return data[indexes[0]]
        else:
            return _recursive_index(data[indexes[0]], indexes[1:])

def _print_depth(message: str, depth: int):
    message = '    ' + message
    print(message)



def _fill_omnifold_weights(point: Point):
    # Create an empty array so we can just fill it and not worry about .append()
    point.omnifold_weight = np.zeros((len(point.sets.values), len(point.percents.values), len(point.tests.values),
                                      len(point.iterations.values), point.num_datapoints))

    # Omnifold generates 1 weight file for each set, percent, and test, so loop over all values
    for set in point.sets.values:
        for percent in point.percents.values:
            for test in point.tests.values:
                # Put the weights in a temp container because we need to reformat them
                weights = get_omnifold_weights(get_file_path([point.data_dir, point.weight_dir,
                                                              point.weight_pat.get_pattern()]))

                # The raw file stores [iteration][step], but we only need step=1 to plot with, so loop over
                #   iterations.
                for iteration in point.iterations.values:

                    # Throw an error if the datapoints don't match exactly, because if we don't, the program will
                    #   likely just keep going and give us faulty plots
                    if weights.shape[-1] != point.num_datapoints:
                        raise ValueError(
                            'Error: ' + point.name + '.weight_pat.weights does not match point.num_datapoints'
                        )

                    # We multiply our omnifold weights by our original syn weights here to avoid re-indexing
                    #   these points later
                    # point.omnifold_weight[set - 1][percent - 1][test - 1][iteration - 1] = (
                    #         weights[iteration - 1][1] * point.syn_weight[set - 1][percent - 1]
                    # )
                    point.omnifold_weight[set - 1][percent - 1][test - 1][iteration - 1] = (
                            weights[iteration - 1][1]
                    )

                # Throw an error if point runs out of files before we're done filling the arrays, so the user knows
                #   to change the file pattern
                if (not point.weight_pat.increment() and set != point.sets.values[-1] and
                        percent != point.percents.values[-1] and test != point.tests.values[-1]):
                    raise ValueError('Error: ' + point.name + '.weight_pat could not increment')

def _recursive_process_binned_sum(bins: np.linspace, data: np.ndarray, weight: np.ndarray, depth: int,
                                  use_numpy_histogram = False):
    if depth == 1:
        # Base case: weight is 1D array, so we can use data to bin and apply histogram
        try:
            bin_sums = np.zeros(len(bins) - 1)

            if use_numpy_histogram:
                bin_sums, _ = np.histogram(data, bins, weights=weight)
                _print_depth('NP bin sums:' + str(bin_sums), depth)
                _print_depth('NP histogram bins: ' + str(_), depth)
            else:
                for i in range(len(bins) - 1):
                    mask = (data >= bins[i]) & (data < bins[i + 1])
                    bin_sums[i] = np.sum(weight[mask])
                _print_depth("Manual bin sums:" + str(bin_sums), depth)

            return bin_sums

        except ValueError as e:
            print("ValueError in histogram:", e)
            return np.zeros(len(bins) - 1)
    elif _get_depth(data) > 1:
        # Recursive case 1: go deeper into data and weights
        result = []
        for i in range(weight.shape[0]):
            binned = _recursive_process_binned_sum(bins, data[i], weight[i], depth - 1, use_numpy_histogram)
            result.append(binned)
        return np.array(result)
    else:
        # Recursive case: process each slice along the first dimension
        result = []
        for i in range(weight.shape[0]):
            binned = _recursive_process_binned_sum(bins, data, weight[i], depth - 1, use_numpy_histogram)
            result.append(binned)
        return np.array(result)

def _recursive_process_percent_error(nat_weight: np.ndarray, syn_weight: np.ndarray, depth: int,
                                     epsilon: float = 0.000001, symmetric: bool = False):
    if depth == 1:
        # Base case: we're at the bin level, calculate the percent error

        _print_depth("Debugging percent error calculation:", depth)
        _print_depth("Natural weights:" + str(nat_weight), depth)
        _print_depth("Synthetic weights:" + str(syn_weight), depth)
        _print_depth("Difference (syn - nat)" + str(syn_weight - nat_weight), depth)

        if symmetric:
            denominator = (nat_weight + syn_weight) / 2
            # Print denominator before epsilon protection
            _print_depth("Denominator before epsilon protection:" + str(denominator), depth)
            denominator = np.where(np.abs(denominator) < epsilon, epsilon, denominator)
            percent_error = 100 * (nat_weight - syn_weight) / denominator
        else:
            denominator = nat_weight + epsilon
            percent_error = 100 * (syn_weight - nat_weight) / denominator

        # print("Calculated percent error:", percent_error)
        return percent_error
    else:
        # Recursive case: go deeper into nested arrays
        result = []
        for syn_w in syn_weight:
            error = _recursive_process_percent_error(nat_weight, syn_w, depth - 1, epsilon, symmetric)
            result.append(error)
        return np.array(result)


def _recursive_process_std_dev_datapoint(bins: np.linspace, data: np.ndarray, weight: np.ndarray, depth: int):
    if depth == 1:
        # Base case: data and weights are both 1D arrays
        data = np.asarray(data)
        weight = np.asarray(weight)

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
                # Normalize weights within this bin
                normalized_weights = weights_in_bin / (np.sum(weights_in_bin) + np.finfo(float).eps)
                # Calculate standard deviation of normalized weights
                std_devs[bin_idx] = np.std(normalized_weights, ddof=1) * 100  # Convert to percentage
            else:
                std_devs[bin_idx] = 0.0  # Single point or empty bin

            _print_depth(f"Bin {bin_idx} ({bins[bin_idx]}-{bins[bin_idx + 1]}): ", depth)
            _print_depth(f"  Points in bin: {len(weights_in_bin)}", depth)
            _print_depth(f"  Std dev: {std_devs[bin_idx]}", depth)

        return std_devs

    elif _get_depth(data) > 1:
        # Recursive case 1: go deeper into data and weights
        result = []
        for i in range(weight.shape[0]):
            binned = _recursive_process_std_dev_datapoint(bins, data[i], weight[i], depth - 1)
            result.append(binned)
        return np.array(result)
    else:
        # Recursive case 2: go deeper into weight arrays
        result = []
        for w in weight:
            binned = _recursive_process_std_dev_datapoint(bins, data, w, depth - 1)
            result.append(binned)
        return np.array(result)

def _process_std_dev(data: np.ndarray, dimension_averages: list[bool]):
    result = data.copy()
    actual_dims = result.ndim

    # # Normalize the data first
    # if result.size > 0:
    #     # Normalize each bin separately to preserve relative variations
    #     bin_sums = np.sum(result, axis=tuple(range(actual_dims-1)))  # Sum all but last dimension
    #     bin_sums = np.expand_dims(bin_sums, axis=tuple(range(actual_dims-1)))  # Reshape for broadcasting
    #     result = result / (bin_sums + np.finfo(float).eps)  # Add small epsilon to avoid division by zero

    # Process dimensions from innermost to outermost
    for i in range(min(len(dimension_averages), actual_dims) - 1, -1, -1):
        if dimension_averages[i]:
            if result.shape[i] > 1:
                result = np.std(result, axis=i, ddof=1)
            else:
                result = np.squeeze(result, axis=i)

    # Convert to percentage
    result = result * 100

    return result

def _average_data(data: np.ndarray, dimension_averages: list[bool], average_bins: bool = False):
    result = data.copy()

    # Get the actual number of dimensions in our data
    actual_dims = result.ndim

    # Process other dimensions from innermost to outermost (reverse order)
    # Only process dimensions that actually exist in our data
    for i in range(min(len(dimension_averages), actual_dims) - 1, -1, -1):
        if dimension_averages[i]:
            result = np.average(result, axis=i)

    # Handle bins (always the last axis)
    if average_bins and actual_dims > 0:
        result = np.average(result, axis=-1)
        actual_dims -= 1

    return result



def _get_x_axis(dimensions: list[Dimension], average_bins: bool, bins: np.linspace):
    # Loop over all dimensions, outermost to innermost, setting the new x_values label accordingly
    x_values = []
    x_label = ''
    # for dimension in dimensions:
    #     if dimension.average:
    #         x_values = dimension.values
    #         x_label = dimension.name
    for i in range(len(dimensions) - 1):
        if not dimensions[i].average:
            x_values = dimensions[i + 1].values
            x_label = dimensions[i + 1].name

    if not average_bins:
        print('bins:', bins)
        x_values = (bins[:-1] + bins[1:]) / 2
        x_label = 'Bins'

    return x_values, x_label

def _recursive_plot_points(points: list[Point], plot_dir: str, plot_pat: Pattern, x_values: np.ndarray, x_label: str,
                           depth: int, indexes: list[int], _top = True, use_symlog_yscale = False):
    if depth == 1:
        # Base case: we have a 1D array to plot for the current points
        for point in points:
            percent_error = _recursive_index(point.percent_error_avg, indexes)
            print(point.name + ' percent error.shape:', percent_error.shape)
            print(percent_error)
            if len(x_values) != len(percent_error):
                raise IndexError('Error: cannot plot points with unmatched pairs of numbers (x, y)')
            if point.plot_error_bars:
                std_dev = _recursive_index(point.std_dev_avg, indexes)
                print(point.name + ' std.dev.shape:', std_dev.shape)
                print(std_dev)
                plt.errorbar(x_values + point.shift, percent_error, yerr=std_dev, marker='o', linestyle='none',
                             color=point.color, ecolor=point.error_color, label=point.name, capsize=5)
            else:
                plt.plot(x_values + point.shift, percent_error, marker='o', linestyle='none', color=point.color,
                     label=point.name)

        # Plot extras and save
        plt.xlabel(x_label)
        if use_symlog_yscale:
            plt.yscale('symlog', linthresh=1e3)
        plt.ylabel('Percent Error')
        plt.title(plot_pat.get_pattern())
        plt.legend()
        plt.savefig(get_file_path([plot_dir, plot_pat.get_pattern()]))
        plt.show()
        plt.close()

        print('\n')

        # Increment plot pat
        if not plot_pat.increment():
            raise ValueError('Error: plot.plot_pat could not increment')
    else:
        # Recursive case: we have an ND array and need to go deeper
        current = _recursive_index(points[0].percent_error_avg, indexes)
        for i in range(len(current)):
            # Insert try-catch here with _top to not throw any errors after the last plot has been made
            try:
                _recursive_plot_points(points, plot_dir, plot_pat, x_values, x_label, depth - 1, indexes + [i], False, use_symlog_yscale)
            except ValueError as e:
                if _top:
                    return
                raise ValueError(e)



def plot_manual(points: list[Point], shift: float, plot_dir: str, plot_pat: Pattern,
                use_numpy_histogram = False, use_symmetric_percent_error = False,
                calculate_std_dev_using_datapoints = False, use_symlog_yscale = False):

    # Data checks & setup
    ##############################


    if len(points) == 0:
        raise ValueError('Error: plot.points cannot be empty')


    for point in points:
        print('point: ' + point.name)

        # Gather data
        ##############################


        print('gather nat data + weights')
        point.nat_data, point.nat_weight = get_nat_data_and_weights(get_file_path([point.data_dir, point.nat_dir,
                                                                        point.nat_file_name]))
        print('nat data shape: ', point.nat_data.shape)

        print('gather syn data + weights')
        point.syn_data = np.empty((len(point.sets.values), len(point.percents.values), point.num_datapoints))
        point.syn_weight = np.empty((len(point.sets.values), len(point.percents.values), point.num_datapoints))
        for set in point.sets.values:
            for percent in point.percents.values:
                syn_data, syn_weight = get_syn_data_and_weights(get_file_path([point.data_dir,
                                                   point.syn_dir,point.syn_pat.get_pattern()]))
                point.syn_data[set - 1][percent - 1] = syn_data
                point.syn_weight[set - 1][percent - 1] = syn_weight
        print('syn data shape: ', point.syn_data.shape)

        print('gather omnifold weights')
        _fill_omnifold_weights(point)
        print('omnifold weights shape: ', point.omnifold_weight.shape)


        # Process data
        ##############################


        averages = [point.sets.average, point.percents.average, point.tests.average, point.iterations.average]

        print('bin nat data')
        print('sum nat weights')
        point.nat_sum_weight = _recursive_process_binned_sum(point.bins, point.nat_data, point.nat_weight,
                                                        _get_depth(point.nat_weight))
        print('nat sum weight shape: ', point.nat_sum_weight.shape)

        print('bin syn data')
        print('sum syn weights')
        point.sum_omnifold_weight = _recursive_process_binned_sum(point.bins, point.syn_data, point.omnifold_weight,
                                                                  _get_depth(point.omnifold_weight))
        print('sum omnifold weight shape: ', point.sum_omnifold_weight.shape)

        print('calculate % error & std. deviation')
        point.percent_error = _recursive_process_percent_error(point.nat_sum_weight, point.sum_omnifold_weight,
                                                               _get_depth(point.sum_omnifold_weight))
        print('percent error shape: ', point.percent_error.shape)
        if calculate_std_dev_using_datapoints:
            point.std_dev = _recursive_process_std_dev_datapoint(point.bins, point.syn_data, point.omnifold_weight,
                                                   _get_depth(point.omnifold_weight))
        else:
            point.std_dev = _process_std_dev(point.sum_omnifold_weight, averages)
        print('std. deviation shape: ', point.std_dev.shape)

        print('average data')
        point.percent_error_avg = _average_data(point.percent_error, averages, point.average_bins)
        print('percent error avg shape: ', point.percent_error_avg.shape)
        if calculate_std_dev_using_datapoints:
            point.std_dev_avg = _average_data(point.std_dev, averages, point.average_bins)
        else:
            point.std_dev_avg = point.std_dev
        print('std. deviation avg shape: ', point.std_dev_avg.shape)



        print('\n')


    # Plot data
    ##############################


    depths = []
    for point in points:
        depths.append(_get_depth(point.percent_error_avg))
    unique_depths = np.unique(depths, axis=0)
    if len(unique_depths) > 1:
        raise ValueError('Error: plot.points contains points with different depths (average values)')

    dimensions = [points[0].sets, points[0].percents, points[0].tests, points[0].iterations]
    x_values, x_label = _get_x_axis(dimensions, points[0].average_bins, points[0].bins)
    x_values = np.asarray(x_values)
    if len(x_values) == 0:
        raise ValueError('Error: plot.points has averaged too many dimensions (no x-axis values)')
    print('Plot over ' + x_label + ' ' + str(x_values))

    _calculate_shifts(shift * (x_values[1] - x_values[0]), points)

    print('plotting points')
    _recursive_plot_points(points, plot_dir, plot_pat, x_values, x_label, depths[0], [])



def plot_defaults(average_sets: bool, average_percents: bool, average_tests: bool, average_iterations: bool,
                  average_bins: bool):
    d_data_dir = 'data'
    d_nat_dir = 'mock'
    d_nat_file_name = 'mockdata.nat.Logweighted2.N150000.root'
    d_syn_dir = 'mock'
    d_weight_dir = 'weights'
    d_re_weight_dir = 're_weights'
    d_plot_dir = d_data_dir + '/plots/bins'

    d_bins_start = 0
    d_bins_end = 100
    d_bins_step = 20
    d_bins = np.arange(d_bins_start, d_bins_end + d_bins_step, d_bins_step)

    d_sets = Dimension(average_sets, 'Set', None, 1, 1, 1)
    d_percents = Dimension(average_percents, 'Percent', None, 1, 1, 1)
    d_tests = Dimension(average_tests, 'Test', None, 1, 10, 1)
    d_iterations = Dimension(average_iterations, 'Iteration', None, 1, 5, 1)
    d_datapoints = 150000

    d_plot_error_bars = True
    d_points = [
        Point(
            'weighted syn', '#5dade2', d_plot_error_bars, '#5dade2', 0.0, d_data_dir, d_nat_dir,
            d_nat_file_name, d_syn_dir, d_weight_dir,
            Pattern([Token('mockdata.syn', d_sets.values), Token('.', d_percents.values),
                     Token('Percent.Logweighted2.N' + str(d_datapoints) + '.root')]),
            Pattern([Token('Syn', d_sets.values), Token('_', d_percents.values), Token('Percent_Test', d_tests.values),
                     Token('.npy')]),
            d_bins, d_sets, d_percents, d_tests, d_iterations, d_datapoints, average_bins
        ),
        Point(
            're weighted syn', '#45b39d', d_plot_error_bars, '#45b39d', 0.0, d_data_dir, d_nat_dir,
            d_nat_file_name, d_syn_dir, d_re_weight_dir,
            Pattern([Token('mockdata.syn', d_sets.values), Token('.', d_percents.values),
                     Token('Percent.Logweighted2.N' + str(d_datapoints) + '.root')]),
            Pattern([Token('Syn', d_sets.values), Token('_', d_percents.values), Token('Percent_Test', d_tests.values),
                     Token('.npy')]),
            d_bins, d_sets, d_percents, d_tests, d_iterations, d_datapoints, average_bins
        )
    ]
    d_shift = 0.15

    tokens = []
    if not average_sets and not average_percents:
        tokens.append(Token('Set', d_sets.values))
    if not average_percents and not average_tests:
        tokens.append(Token('Percent', d_percents.values))
    if not average_tests and not average_iterations:
        tokens.append(Token('Test', d_tests.values))
    print(average_iterations)
    if not average_iterations and not average_bins:
        tokens.append(Token('Iteration', d_iterations.values))
    if not average_bins and not average_sets and not average_percents and not average_tests and not average_iterations:
        bins_str = []
        for i in range(len(d_bins) - 1):
            bins_str.append(str(d_bins[i]) + '-' + str(d_bins[i + 1]))
        tokens.append(Token('Bin'))
        tokens.append(Token(bins_str))

    tokens.append(Token('.png'))
    d_plot_pat = Pattern(tokens)
    while True:
        print(d_plot_pat.get_pattern())
        if not d_plot_pat.increment():
            break


    # Hack-y variables, don't touch unless you know what you're doing

    d_use_numpy_histogram = False
    d_use_symmetric_percent_error = False
    d_calculate_std_dev_using_datapoints = False
    d_use_symlog_yscale = False

    plot_manual(d_points, d_shift, d_plot_dir, d_plot_pat,
                d_use_numpy_histogram, d_use_symmetric_percent_error, d_calculate_std_dev_using_datapoints,
                d_use_symlog_yscale)