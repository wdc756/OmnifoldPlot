# This file contains methods to plot omnifold data according to settings from omnifold_plot.py


import numpy as np
import matplotlib.pyplot as plt

import src.util.data as data
from src.util.data import PlotSEOptions, Point



########################################################################################################################

# Common functions

########################################################################################################################



def _get_bin_idx(unbinned_data: np.ndarray, bins: np.linspace, num_bins: int):
    # When binning, we create this bin_idx var. It's an array that maps each nat datapoint to the bin it belongs in.
    #   The first line creates the bin mapping, and the - 1 at the end is to 0-index the map, since np 1-indexes by
    #   default. The 2nd line catches any datapoints on the end edges of the last bin and maps them inside
    bin_idx = np.digitize(unbinned_data, bins) - 1
    bin_idx[bin_idx == num_bins] = num_bins - 1

    return bin_idx

def _get_weighted_nat(op: PlotSEOptions, bins: np.linspace, num_bins: int):
    # These functions pull data from a root file
    nat_data_path = data.get_file_path([op.data_dir, op.nat_dir, op.nat_file_pat.get_pattern()])
    nat_data, nat_weight = data.get_nat_data_and_weights(nat_data_path)

    # Weight data then bin to ease processing later, and let us make other observations
    weighted_nat = nat_data * nat_weight
    bin_idx = _get_bin_idx(weighted_nat, bins, num_bins)

    return weighted_nat, bin_idx

def _get_syn_data(point: Point, data_dir: str):
    # These functions pull data from a root file
    syn_data_path = data.get_file_path([data_dir, point.syn_dir, point.syn_pat.get_pattern()])
    syn_data = data.get_syn_data(syn_data_path)

    if len(syn_data) != point.num_datapoints:
        raise ValueError('syn_data does not match ' + point.name + '.num_datapoints')

    return syn_data

def _get_weighted_syn(op: PlotSEOptions, point: Point):
    # Since each Point can have a different shape (num iterations, test, and datapoints), we need to get individual syn
    #   data for each one
    syn_data = _get_syn_data(point, op.data_dir)

    # Container to hold all the weights at once to avg later
    syn_weight = np.empty((point.num_tests, point.num_iterations, point.num_datapoints), dtype=np.float64)


    # Loop over weight files per-test - because omnifold outputs 1 file per set:percent:iteration:test
    for test in range(point.num_tests):
        # These functions load weights from a np file
        syn_weight_data_path = data.get_file_path([op.data_dir, point.weight_dir, point.weight_pat.get_pattern()])
        syn_weight_data = data.get_syn_weights(syn_weight_data_path)


        if len(syn_weight_data) != point.num_iterations:
            raise ValueError('syn_weight does not match ' + point.name + '.num_iterations')
        if len(syn_weight_data[0]) != 2:
            raise ValueError('syn_weight does not have two sets of weights')
        if len(syn_weight_data[0][0]) != point.num_datapoints:
            raise ValueError('syn_weight does not match ' + point.name + '.num_datapoints')


        # Loop over iterations to get step 2 weights - when omnifold runs, it generates two sets of weights. For
        #   this program, we only need the 2nd set(step)
        for iteration in range(point.num_iterations):
            syn_weight[test][iteration] = syn_weight_data[iteration][1]


        # Increment weight pattern for next file
        if point.weight_pat.increment() is False:
            break


    # Average weights over tests
    syn_weight = syn_weight.mean(axis=0)
    # resulting in the shape [num_iterations, num_datapoints]

    # Reshape syn weights and apply average
    return syn_data[np.newaxis, :] * syn_weight
    # resulting in the shape [num_iterations, num_datapoints]

def _process_weighted_syn(op: PlotSEOptions, point_index: int, bins: np.linspace, num_bins: int, nat_bin_idx: np.ndarray,
                          weighted_nat: np.ndarray, weighted_syn: np.ndarray):
    # When making this function, it was easier to just pass the index of the current point, since the caller didn't
    #   need a point ref, but this function does so we need to get it here
    point = op.points[point_index]


    # This function will return these two arrays at the end
    mean_error = np.empty((num_bins, point.num_iterations), dtype=np.float64)
    std_error = np.empty((num_bins, point.num_iterations), dtype=np.float64)

    # Loop over bins, adding values to mean_error and std_error for each bin
    for b in range(num_bins):

        # Get nat data mask
        nat_mask_b = (nat_bin_idx == b)
        # We need to skip empty bins because if we don't we get infinite % error
        if not nat_mask_b.any():
            mean_error[b, :] = np.nan
            std_error[b, :] = np.nan
            continue

        # Use nat mask to get nat values for this bin
        nat_b = weighted_nat[nat_mask_b]

        # Since syn and nat have different sizes, we need to compare mean values
        mean_nat = np.nanmean(nat_b)


        # Get syn bin map and mask to bin syn data to exclude data from other bins
        syn_bin_idx = _get_bin_idx(weighted_syn, bins, num_bins)
        syn_mask_b = (syn_bin_idx == b)

        # We need to check if syn has any events in bin b, because if not we need to manually set the mean to 0.
        # The reason we don't just continue like earlier is because we actually want this error calculation
        mean_syn = np.zeros(point.num_iterations)
        for i in range(point.num_iterations):
            syn_values = weighted_syn[i, :]
            mask_i = syn_mask_b[i, :]
            values_in_bin = syn_values[mask_i]

            if values_in_bin.size > 0:
                mean_syn[i] = np.nanmean(values_in_bin)
            else:
                mean_syn[i] = 0  # or np.nan, depending on how you want to treat missing iterations
        # mean_syn now has shape [num_iterations]

        # Calculate % error & std. deviation
        num = mean_syn - mean_nat
        denom = mean_nat
        err_per_iter = np.where(denom != 0, num / denom * 100.0, np.nan)
        # Now err_per_iter is shape [i], where each element is the % error for that iteration in bin b

        # Since err_per_iter is already the right shape, just add it onto mean_error, but we need to calculate std_error
        #   for this iteration
        mean_error[b, :] = err_per_iter
        std_error[b] = np.nanstd(err_per_iter)

    return mean_error, std_error

def _calculate_point_shifts(op: PlotSEOptions):
    for p in range(len(op.points)):
        point = op.points[p]

        # Only shift if the user hasn't set it manually
        if point.shift == 0:
            point.shift = op.shift * (p - len(op.points) // 2)



########################################################################################################################

# SEI (Synthetic Error by Iteration) plot functions

########################################################################################################################



############################################################
# Plotting
############################################################



def _sei_plot_weighted_syn(op: PlotSEOptions, point_index: int, mean_error: np.ndarray, std_error: np.ndarray):
    point = op.points[point_index]

    weighted_iters = np.arange(mean_error.size)
    plt.errorbar(weighted_iters + point.shift,
                 mean_error,
                 yerr=std_error,
                 fmt='o',
                 capsize=4,
                 label=point.name,
                 ecolor=point.error_color,
                 color=point.color)

def _sei_plot(op: PlotSEOptions, mean_errors: np.ndarray, std_errors: np.ndarray):
    # Plot each point
    for p in range(len(op.points)):
        _sei_plot_weighted_syn(op, p, mean_errors[p], std_errors[p])

    plt.axhline(0, color='k', linestyle='--', linewidth=1)
    plt.xlabel('Iteration')
    plt.ylabel('Mean % Error ±1σ')
    plt.title(op.plot_file_pat.get_pattern() + ': Syn vs Nat % Error')
    plt.grid(True)
    plt.tight_layout()
    plt.legend()

    plt.savefig(data.get_file_path([op.data_dir, op.plot_dir, op.plot_file_pat.get_pattern()]))
    plt.show()

    op.plot_file_pat.increment()

def _sei_plot_bins(op: PlotSEOptions, bins: np.linspace, num_bins: int, nat_bin_idx: np.ndarray,
                   weighted_nat: np.ndarray, weighted_syn_all: np.ndarray):
    # Get mean & std error for all bins and points, then plot
    for b in range(num_bins):
        for p in range(len(op.points)):
            mean_error, std_error = _process_weighted_syn(op, p, bins, num_bins, nat_bin_idx, weighted_nat,
                                                          weighted_syn_all[p])

            _sei_plot(op, mean_error, std_error)

def _sei_plot_combined(op: PlotSEOptions, bins: np.linspace, num_bins: int, nat_bin_idx: np.ndarray,
                       weighted_nat: np.ndarray, weighted_syn_all: np.ndarray):
    # Since we're plotting the average over all bins, we need to make a large array to temporarily hold all the data
    mean_error_all = []
    std_error_all = []

    # Get mean & std error for all bins and points
    for b in range(num_bins):
        for p in range(len(op.points)):
            mean_error, std_error = _process_weighted_syn(op, p, bins, num_bins, nat_bin_idx, weighted_nat,
                                                          weighted_syn_all[p])
            mean_error_all.append(mean_error)
            std_error_all.append(std_error)
    # This loop results in mean_error_all/std_error_all with shapes [num_points, num_bins, num_iterations]


    # Average all bins for each point
    mean_errors = []
    std_errors = []
    for p in range(len(op.points)):
        mean_error = np.nanmean(mean_error_all[p], axis=0)
        std_error = np.nanmean(std_error_all[p], axis=0)

        mean_errors.append(mean_error)
        std_errors.append(std_error)


    _sei_plot(op, mean_errors, std_errors)



############################################################
# Main - user facing
############################################################



def plot_sei(op: PlotSEOptions):
    # Since the user can enable SEI plotting in 'Main Settings' in omnifold.py but not include any points to plot,
    #   we need to check if there is anything to do, so we don't make a bunch of empty plots
    if len(op.points) == 0:
        print("Plot SEI is true, but no datapoints are set to plot")
        return


    # Since plots can contain any number of points, we want to make sure they don't overlap and block each other
    #   so we use this to shift them on the x-axis a little
    _calculate_point_shifts(op)


    # When plotting, we use bins to segment our data. This is done so we can better analyze performance
    # Basically we can see if omnifold is underperforming in a given data range, and use this knowledge to correct it
    num_bins = int((op.bins_end - op.bins_start) / op.bins_step)
    bins = np.linspace(op.bins_start, op.bins_end, num_bins + 1)


    # Since this function plots % error, we need a frame of reference to compare our omnifold-weighted syn data against
    #   so we need natural (or true) data.
    # Note that weighted_nat can have a different shape (number of datapoints) than your points do, which is why later
    #   on we create other bin_idx map arrays for all other datapoints
    weighted_nat, nat_bin_idx = _get_weighted_nat(op, bins, num_bins)


    # Loop over all syn data files. Set num_syn_datasets and num_percent_deviations to 1 if you only want to compare 1
    #   syn file or set of syn datapoints
    for syn_d in range(op.num_syn_datasets):
        for syn_p in range(op.num_percent_deviations):

            # Loop over all points, weighting their syn data and bin_idx maps
            weighted_syn_all = []
            for point in op.points:
                weighted_syn_data = _get_weighted_syn(op, point)
                weighted_syn_all.append(weighted_syn_data)
            # By the end weighted_syn will be [num_points, num_iterations, num_datapoints]


            if op.plot_combined:
                _sei_plot_combined(op, bins, num_bins, nat_bin_idx, weighted_nat, weighted_syn_all)
            else:
                _sei_plot_bins(op, bins, num_bins, nat_bin_idx, weighted_nat, weighted_syn_all)


            # Increment all point syn file patterns to keep up with the two loops here
            for point in op.points:
                if not point.syn_pat.increment():
                    # We only want to trigger this error before we're done, so if the two iterator values are done, we
                    #   can ignore the error
                    if syn_d < op.num_syn_datasets - 1 or syn_p < op.num_percent_deviations - 1:
                        raise ValueError(point.name + '.syn_pat failed to increment')



########################################################################################################################

# SEB (Synthetic Error by Bin) plot functions

########################################################################################################################



############################################################
# Plotting
############################################################



def _seb_plot_weighted_syn(op, point_index, bin_centers, mean_error, std_error, iteration):
    point = op.points[point_index]

    plt.errorbar(bin_centers + point.shift,
                 mean_error[:, iteration],
                 yerr=std_error[:, iteration],
                 fmt='o',
                 capsize=4,
                 label=point.name,
                 ecolor=point.error_color,
                 color=point.color)

def _seb_plot_extras(op):
    plt.axhline(0, color='k', linestyle='--', linewidth=1)
    plt.xlabel('Bin')
    plt.ylabel('Mean % Error ±1σ')
    plt.title(op.plot_file_pat.get_pattern() + ': Syn vs Nat % Error')
    plt.grid(True)
    plt.tight_layout()
    plt.xlim(op.bins_start, op.bins_end)
    plt.xticks(np.arange(op.bins_start, op.bins_end + 1, op.bins_step))
    plt.legend()

def _seb_plot_iterations(op: PlotSEOptions, bins: np.linspace, num_bins: int, nat_bin_idx: np.ndarray,
                         weighted_nat: np.ndarray, weighted_syn: np.ndarray):
    mean_errors = []
    std_errors = []
    for p in range(len(op.points)):
        weighted_mean_error, weighted_std_error = _process_weighted_syn(op, p, bins, num_bins, nat_bin_idx,
                                                                        weighted_nat, weighted_syn[p])
        mean_errors.append(weighted_mean_error)
        std_errors.append(weighted_std_error)


    # Since we're plotting iterations but with bins on the x-axis, we need a way to map iterations to the x-axis, so we
    #   create this var to help place the iteration points in the center of each bin horizontally
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # This plot doesn't really make sense if the points have different iteration counts, so this loop only goes over
    #   the ones set in the main options. This means if a Point has more iterations, they'll be truncated, and if it
    #   has less it might cause errors, or just missing/skewed points
    for iteration in op.items_to_plot:
        i = iteration - 1

        for point_index in range(len(op.points)):
            _seb_plot_weighted_syn(op, point_index, bin_centers, mean_errors[point_index], std_errors[point_index], i)

        _seb_plot_extras(op)
        plt.savefig(data.get_file_path([op.data_dir, op.plot_dir, op.plot_file_pat.get_pattern()]))
        plt.show()

        op.plot_file_pat.increment()




############################################################
# Main - user facing
############################################################



def plot_seb(op: PlotSEOptions):
    # As a general note, if you're looking for comments on how the code for SEB works, just look at the SEI functions.
    #   Since the two functions are so similar, I decided to just not put any repetitive comments here, though there
    #   are a few unique to SEB


    if len(op.points) == 0:
        print("Plot SEB is true, but no datapoints are set to plot")
        return


    _calculate_point_shifts(op)


    num_bins = int((op.bins_end - op.bins_start) / op.bins_step)
    bins = np.linspace(op.bins_start, op.bins_end, num_bins + 1)


    weighted_nat, nat_bin_idx = _get_weighted_nat(op, bins, num_bins)


    for syn_d in range(op.num_syn_datasets):
        for syn_p in range(op.num_percent_deviations):

            weighted_syn_all = []
            for point in op.points:
                weighted_syn_data = _get_weighted_syn(op, point)
                weighted_syn_all.append(weighted_syn_data)


            _seb_plot_iterations(op, bins, num_bins, nat_bin_idx, weighted_nat, weighted_syn_all)


            for point in op.points:
                if not point.syn_pat.increment():
                    if syn_d < op.num_syn_datasets - 1 or syn_p < op.num_percent_deviations - 1:
                        raise ValueError(point.name + '.syn_pat failed to increment')