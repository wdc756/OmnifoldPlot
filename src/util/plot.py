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
    if isinstance(obj, Iterable) and not isinstance(obj, (str, bytes, np.ndarray)):
        try:
            first = next(iter(obj))
        except StopIteration:
            return 1
        return 1 + _get_depth(first)
    return 0

def _recursive_index(data: np.ndarray, indexes: list[int]):
    if len(indexes) == 1:
        return data[indexes[0]]
    else:
        return _recursive_index(data[indexes[0]], indexes[1:])



def _get_binned_sum(bins: np.linspace, data, weights, depth: int):
    if depth == 1:
        # base case: data is a 1D array (leaf level), apply histogram
        hist, _ = np.histogram(data, bins=bins, weights=weights)
        return hist
    else:
        # recursive case: go deeper into nested arrays
        result = []
        for d, w in zip(data, weights):
            binned = _get_binned_sum(bins, d, w, depth - 1)
            result.append(binned)
        return np.array(result)

def _get_binned_percent_error(nat_weights, syn_weights, depth: int, epsilon: float = 0.000001, symmetric: bool = False):
    if depth == 1:
        # Base case: we're at the bin level, calculate the percent error
        if symmetric:
            return 100 * (nat_weights - syn_weights) / ((nat_weights + syn_weights) / 2)
        return 100 * (syn_weights - nat_weights) / (nat_weights + epsilon)
    else:
        # Recursive case: go deeper into nested arrays
        result = []
        for syn_w in syn_weights:
            error = _get_binned_percent_error(nat_weights, syn_w, depth - 1, epsilon, symmetric)
            result.append(error)
        return np.array(result)

def _get_binned_std_dev(bins: np.linspace, data, weights, depth: int):
    if depth == 1:
        # Base case: data is 1D array, so we can apply binning and calculate std dev
        bin_map = np.digitize(data, bins) - 1

        # Initialize std dev array for this set of bins
        std_devs = np.zeros(len(bins) - 1)  # bins has n+1 edges, n bins

        # Calculate std dev for each bin
        for bin_idx in range(len(bins) - 1):
            mask = bin_map == bin_idx
            weights_in_bin = weights[mask]
            # Need at least 2 points for std dev
            if len(weights_in_bin) > 1:
                std_devs[bin_idx] = np.std(weights_in_bin)
            else:
                std_devs[bin_idx] = 0

        return std_devs
    else:
        # Recursive case: go deeper into nested arrays
        result = []
        for d, w in zip(data, weights):
            binned = _get_binned_std_dev(bins, d, w, depth - 1)
            result.append(binned)
        return np.array(result)



def _average_data(data: np.ndarray, settings: list[OmnifoldSetting], average_bins: bool, depth: int):
    if depth == 1:
        # Base case: return array based on average flag for current omnifold setting (0)

        # Check if average_bins == True first because it's the lowest layer
        if average_bins:
            return np.average(data, axis=-1)

        if settings[0].average:
            return np.average(data, axis=0)
        return data
    else:
        # Recursive case: go deeper into nested arrays, then apply current average setting
        result = []
        for d in data:
            avg = _average_data(d, settings, average_bins, depth - 1)
            result.append(avg)

        result = np.array(result)

        if settings[depth].average:
            return np.average(result, axis=0)
        return result

def _plot_points(points: list[Point], bins: np.linspace, plot_dir: str, plot_pat: Pattern, depth: int, indexes: list[int] = None):
    if depth == 1:
        # Base case: we have a 1D array to plot for the current points

        # For each point
            # call recursiveIndex() to get the current 1D array
            # plot point
        for point in points:
            percent_error = _recursive_index(point.percent_error_avg, indexes)
            if point.plot_error_bars:
                std_dev = _recursive_index(point.std_error_avg, indexes)
            plt.plot(point.sets.values, percent_error, color=point.color, label=point.name, fmt='o')
    else:
        # Recursive case: we have an ND array and need to go deeper

        # Get current depth size

        # For each i in depth size
            # new_indexes = indexes + [i]
            # _plot_points()



def plot_manual(data_dir: str, nat_dir: str, nat_file_name: str,
                bins_start: int, bins_end: int, bins_step: int, average_bins: bool, shift: float,
                points: list[Point], plot_dir: str, plot_pat: Pattern):

    # Data checks & setup
    ##############################


    if len(points) == 0:
        raise ValueError("Error: plot.points cannot be empty")

    _calculate_shifts(shift, points)


    # Gather data
    ##############################


    print("get nat data + weights")
    nat_data, nat_weights = get_nat_data_and_weights(get_file_path([data_dir, nat_dir, nat_file_name]))

    for point in points:
        print("gather point syn data + weights")
        point.syn_data, point.syn_weight = get_syn_data_and_weights(get_file_path([data_dir, point.syn_dir,
                                                                                   point.syn_pat.get_pattern()]))
        print("gather point omnifold weights")
        point.omnifold_weight = np.zeros((len(point.sets.values), len(point.percents.values),
                                          len(point.iterations.values), len(point.tests.values), point.num_datapoints))

        # Omnifold generates 1 weight file for each set, percent, and test, so loop over all values
        for set in point.sets.values:
            for percent in point.percents.values:
                for test in point.tests.values:
                    # Put the weights in a temp container because we need to reformat them
                    weights = get_omnifold_weights(get_file_path([data_dir, point.weight_dir,
                                                                  point.weight_pat.get_pattern()]))

                    # The raw file stores [iteration][step], but we only need step=1 to plot with, so loop over
                    #   iterations.
                    for iteration in point.iterations.values:

                        # Throw an error if the datapoints don't match exactly, because if we don't, the program will
                        #   likely just keep going and give us faulty plots
                        if weights.shape[iteration - 1][1] != point.num_datapoints:
                            raise ValueError(
                                'Error: ' + point.name + '.weight_pat.weights does not match point.num_datapoints')

                        # We multiply our omnifold weights by our original syn weights here to avoid re-indexing
                        #   these points later
                        point.omnifold_weight[set - 1][percent - 1][iteration - 1][test - 1] = (
                            weights)[iteration - 1][1] * point.syn_weight[set - 1][percent - 1]

                    # Throw an error if point runs out of files before we're done filling the arrays, so the user knows
                    #   to change the file pattern
                    if (not point.weight_pat.increment() and set != point.sets.values[-1] and
                            percent != point.percents.values[-1] and test != point.tests.values[-1]):
                        raise ValueError('Error: ' + point.name + '.weight_pat could not increment')


    # Process data
    ##############################


    bins = np.linspace(bins_start, bins_end, bins_step)

    print('bin nat data')
    print('sum nat weights')
    nat_sum_weights, _ = _get_binned_sum(bins, nat_data, nat_weights, _get_depth(nat_weights))

    for point in points:
        print('bin syn data')
        print('sum syn weights')
        point.sum_omnifold_weight = _get_binned_sum(bins, point.syn_data, point.omnifold_weight,
                                                    _get_depth(point.omnifold_weight))

        print('calculate % error & std. deviation')
        point.percent_error = _get_binned_percent_error(nat_sum_weights, point.sum_omnifold_weight,
                                                        _get_depth(nat_sum_weights))
        point.std_dev = _get_binned_std_dev(bins, point.syn_data, point.omnifold_weight,
                                            _get_depth(point.omnifold_weight))

        print('calculate averages')
        settings = [point.sets, point.percents, point.iterations, point.tests]
        point.percent_error_avg = _average_data(point.percent_error, settings, average_bins,
                                                _get_depth(point.percent_error))
        point.std_error_avg = _average_data(point.std_dev, settings, average_bins, _get_depth(point.std_dev))


    # Plot data
    ##############################


    shapes = []
    for point in points:
        shapes.append(point.percent_error_avg.shape)
    unique_shapes = np.unique(shapes, axis=0)
    if len(unique_shapes) > 1:
        raise ValueError("Error: plot.points contains points with different plot shapes")

    # for set in points[0].sets.values:
    #     for percent in points[0].percents.values:
    #         for iteration in points[0].iterations.values:
    #             for test in points[0].tests.values:
    #                 for point in points:
    #                     if point.plot_error_bars:
    #                         plt.errorbar(bins, point.percent_error_avg[set - 1][percent - 1][iteration - 1][test - 1],
    #                                      yerr=point.std_error_avg[set - 1][percent - 1][iteration - 1][test - 1],
    #                                      color=point.color, label=point.name, fmt='o',
    #                                      ecolor=point.error_color, capsize=3)
    #                     else:
    #                         plt.plot(bins, point.percent_error_avg[set - 1][percent - 1][iteration - 1][test - 1],
    #                                  color=point.color, label=point.name, fmt='o')

    depth = _get_depth(points[0].percent_error_avg)
    for d in range(depth):




def plot(po: PlotOptions):
    # Yes, I know I made the data structs and everything, but those are only for the user. On the backend we only use
    #   raw data. Trust me, it makes everything so much easier to debug
    plot_manual(po.data_dir, po.nat_dir, po.nat_file_name,
                po.bins_start, po.bins_end, po.bins_step, po.shift,
                po.points, po.plot_dir, po.plot_pat)