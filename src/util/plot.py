# This file contains methods to plot omnifold data according to settings from omnifold_plot.py


import numpy as np
import matplotlib.pyplot as plt

import src.util.data as data
from src.util.data import PlotSEOptions, Point



########################################################################################################################

# Common functions

########################################################################################################################



def _get_bin_idx(unbinned_data: list[float], bins: np.linspace, num_bins: int):
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

def _process_weighted_syn(op, point_index, num_bins, bin_idx, weighted_nat, weighted_syn):
    point = op.points[point_index]

    mean_error = np.empty((num_bins, point.num_iterations), dtype=np.float64)
    std_error = np.empty((num_bins, point.num_iterations), dtype=np.float64)

    for b in range(num_bins):
        mask_b = (bin_idx == b)
        if not mask_b.any():
            mean_error[b, :] = np.nan
            std_error[b, :] = np.nan
            continue

        nat_b = weighted_nat[mask_b]
        weighted_syn_B = weighted_syn[:, mask_b]
        num = weighted_syn_B - nat_b[np.newaxis, :]
        denom = nat_b[np.newaxis, :]
        err_b = np.where(denom != 0, num / denom * 100.0, np.nan)

        mean_error[b, :] = np.nanmean(err_b, axis=1)
        std_error[b, :] = np.nanstd(err_b, axis=1)

    return mean_error, std_error

def _calculate_point_shifts(op):
    for p in range(len(op.points)):
        point = op.points[p]
        point.shift = op.shift * (p - len(op.points) // 2)



########################################################################################################################

# SEI (Synthetic Error by Iteration) plot functions

########################################################################################################################



############################################################
# Plotting
############################################################



def _sei_plot_weighted_syn(op, point_index, mean_error, std_error):
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

def _sei_plot(op, mean_errors, std_errors, data_dir):
    for p in range(len(op.points)):
        _sei_plot_weighted_syn(op, p, mean_errors[p], std_errors[p])

    plt.axhline(0, color='k', linestyle='--', linewidth=1)
    plt.xlabel('Iteration')
    plt.ylabel('Mean % Error ±1σ')
    plt.title(op.plot_file_pat.get_pattern() + ': Syn vs Nat % Error')
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.savefig(data.get_file_path([data_dir, op.plot_dir, op.plot_file_pat.get_pattern()]))
    plt.show()

    op.plot_file_pat.increment()

def _sei_plot_bins(op, num_bins, bin_idx, weighted_nat, datapoints, data_dir):
    for b in range(num_bins):
        mask_b = (bin_idx == b)
        if not np.any(mask_b):
            continue


        nat_b = weighted_nat[mask_b]
        denom = nat_b[np.newaxis, :]


        for p in range(len(op.points)):
            mean_error, std_error = _process_weighted_syn(op, p, num_bins, bin_idx, weighted_nat, datapoints[p])

            _sei_plot(op, mean_error, std_error, data_dir)

def _sei_plot_combined(op: PlotSEOptions, num_bins: int, nat_bin_idx, weighted_nat, datapoints, data_dir):
    # Since we're plotting the average over all bins, we need to make a large array to temporarily hold all the data
    raw_mean_errors = []
    raw_std_errors = []


    for b in range(num_bins):
        # mask_b = (bin_idx == b)
        # if not np.any(mask_b):
        #     continue
        #
        # nat_b = weighted_nat[mask_b]
        # denom = nat_b[np.newaxis, :]
        #
        #
        # for p in range(num_points):
        #     raw_mean_error, raw_std_error = _process_weighted_syn(op, p, num_bins, bin_idx, weighted_nat, datapoints[p])
        #     raw_mean_errors[p, :, :] = raw_mean_error
        #     raw_std_errors[p, :, :] = raw_std_error

        # Get current nat_data for this mask
        nat_mask_b = (nat_bin_idx == b)
        if not np.any(nat_mask_b):
            continue # When we get a bin with no data, we can just skip it, because the % error would be infinite
        nat_b = weighted_nat[nat_mask_b]

        for p in range(len(op.points)):


    # Average all bins for each point
    mean_errors = []
    std_errors = []
    for p in range(num_points):
        mean_error = np.nanmean(raw_mean_errors[p], axis=0)
        std_error = np.nanstd(raw_mean_errors[p], axis=0)

        mean_errors.append(mean_error)
        std_errors.append(std_error)


    _sei_plot(op, mean_errors, std_errors, data_dir)



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
            weighted_syn = []
            bin_idx = []
            for point in op.points:
                weighted_syn_data, syn_bin_idx = _get_weighted_syn(op, point)
                weighted_syn.append(weighted_syn_data)
                bin_idx.append(syn_bin_idx)
            # By the end weighted_syn will be [num_points, num_iterations, num_datapoints], and
            #   bin_idx will be [num_points, num_datapoints]


            if op.plot_combined:
                _sei_plot_combined(op, num_bins, bin_idx, weighted_nat, weighted_syn, data_dir)
            else:
                _sei_plot_bins(op, num_bins, bin_idx, weighted_nat, weighted_syn, data_dir)


            # Increment all point syn file patterns to keep up with the two loops here
            for point in op.points:
                if not point.syn_pat.increment():
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

def _seb_plot_iterations(op, num_bins, bin_idx, bins, weighted_nat, datapoints, data_dir):
    mean_errors = []
    std_errors = []
    for point_index in range(len(op.points)):
        weighted_mean_error, weighted_std_error = _process_weighted_syn(op, point_index, num_bins, bin_idx, weighted_nat,
                                                                        datapoints[point_index])
        mean_errors.append(weighted_mean_error)
        std_errors.append(weighted_std_error)


    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    for iteration in op.iterations_to_plot:
        iteration = iteration - 1

        for point_index in range(len(op.points)):
            _seb_plot_weighted_syn(op, point_index, bin_centers, mean_errors[point_index], std_errors[point_index], iteration)

        _seb_plot_extras(op)
        plt.savefig(data.get_file_path([data_dir, op.plot_dir, op.plot_file_pat.get_pattern()]))
        plt.show()

        op.plot_file_pat.increment()



############################################################
# Main - user facing
############################################################



def plot_seb(op: PlotSEOptions, data_dir: str):
    # Check execution bools first
    if len(op.points) == 0:
        print("Plot SEB is true, but no datapoints are set to plot")
        return


    # Create bins
    num_bins = int((op.bins_end - op.bins_start) / op.bins_step)
    bins = np.linspace(op.bins_start, op.bins_end, num_bins + 1)


    _calculate_point_shifts(op)


    # Get nat data and binning array
    weighted_nat, bin_idx = _get_weighted_nat(op, bins, num_bins, data_dir)


    # Loop over all syn data files
    for syn_d in range(op.num_syn_datasets):
        for syn_p in range(op.num_percent_deviations):
            # Get syn data
            syn_data = _get_syn_data(op, data_dir)


            # Get weighted syn data from all points
            datapoints = []
            for point_index in range(len(op.points)):
                datapoints.append(_get_weighted_syn(op, point_index, data_dir, syn_data))


            # Plot all iterations for this syn file
            _seb_plot_iterations(op, num_bins, bin_idx, bins, weighted_nat, datapoints, data_dir)


            # Increment syn_file_pat to use next file on loop continue
            if op.syn_file_pat.increment() is False:
                print("Done with syn files")
                break