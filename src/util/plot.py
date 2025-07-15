# This file contains methods to plot omnifold data according to settings from omnifold_plot.py


import numpy as np
import matplotlib.pyplot as plt

import src.util.data as data



########################################################################################################################

# Common functions

########################################################################################################################



def _get_weighted_nat(op, bins, num_bins, data_dir):
    # Get nat data and weight
    nat_data_path = data.get_file_path([data_dir, op.nat_dir, op.nat_file_pat.get_pattern()])
    nat_data, nat_weight = data.get_nat_data_and_weights(nat_data_path)

    # Check shape
    if nat_data.shape[0] != op.num_datapoints:
        raise Exception("Nat data has wrong number of datapoints: " + nat_data.shape[0] + "!=" + op.num_datapoints)
    if nat_weight.shape[0] != op.num_datapoints:
        raise Exception("Nat weight has wrong number of datapoints: " + nat_weight.shape[0] + "!=" + op.num_datapoints)

    # Apply nat weights
    weighted_nat = nat_data * nat_weight

    # Bin nat data
    bin_idx = np.digitize(nat_data, bins) - 1
    bin_idx[bin_idx == num_bins] = num_bins - 1

    return weighted_nat, bin_idx

def _get_syn_data(op, data_dir):
    syn_data_path = data.get_file_path([data_dir, op.syn_dir, op.syn_file_pat.get_pattern()])
    syn_data = data.get_syn_data(syn_data_path)

    if syn_data.shape[0] != op.num_datapoints:
        raise Exception("Syn data has wrong number of datapoints: " + syn_data.shape[0] + "!=" + op.num_datapoints)

    return syn_data

def _get_weighted_syn(op, point_index, data_dir, syn_data):
    if point_index >= len(op.points):
        raise Exception("Point index out of range: " + point_index + " >= " + len(op.points))

    point = op.points[point_index]

    # Container to hold all the weights at once to avg later
    syn_weight = np.empty((point.num_tests, point.num_iterations, op.num_datapoints), dtype=np.float64)

    # Loop over weight files
    for test in range(point.num_tests):
        # Load file
        syn_weight_data_path = data.get_file_path([data_dir, point.dir, point.file_pat.get_pattern()])
        syn_weight_data = data.get_syn_weights(syn_weight_data_path)
        if syn_weight_data.shape != (point.num_iterations, 2, op.num_datapoints):
            raise Exception("Syn weight data has wrong shape: " + syn_weight_data.shape + "!=" +
                            str((point.num_iterations, 2, op.num_datapoints)))

        # Loop over iterations to get step 2 weights
        for iteration in range(point.num_iterations):
            syn_weight[test][iteration] = syn_weight_data[iteration][1]

        # Increment weight file pat to use the next one
        if point.file_pat.increment() is False:
            break

    # Average weights over tests
    syn_weight = syn_weight.mean(axis=0)

    # Reshape syn weights to apply avg to
    return syn_data[np.newaxis, :] * syn_weight

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

def _sei_plot_combined(op, num_bins, bin_idx, weighted_nat, datapoints, data_dir):
    num_points = len(op.points)
    raw_mean_errors = np.empty((num_points, num_bins, op.num_iterations), dtype=np.float64)
    raw_std_errors = np.empty((num_points, num_bins, op.num_iterations), dtype=np.float64)


    for b in range(num_bins):
        mask_b = (bin_idx == b)
        if not np.any(mask_b):
            continue

        nat_b = weighted_nat[mask_b]
        denom = nat_b[np.newaxis, :]


        for p in range(num_points):
            raw_mean_error, raw_std_error = _process_weighted_syn(op, p, num_bins, bin_idx, weighted_nat, datapoints[p])
            raw_mean_errors[p, :, :] = raw_mean_error
            raw_std_errors[p, :, :] = raw_std_error

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



def plot_sei(op, data_dir):
    # Check execution bools first
    if len(op.points) == 0:
        print("Plot SEI is true, but no datapoints are set to plot")
        return


    # Create bins
    num_bins = int((op.bins_end - op.bins_start) / op.bins_step)
    bins = np.linspace(op.bins_start, op.bins_end, num_bins + 1)


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


            if op.plot_combined:
                _sei_plot_combined(op, num_bins, bin_idx, weighted_nat, datapoints, data_dir)
            else:
                _sei_plot_bins(op, num_bins, bin_idx, weighted_nat, datapoints, data_dir)


            # Increment syn_file_pat to use next file on loop continue
            if op.syn_file_pat.increment() is False:
                print("Done with syn files")
                break



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



def plot_seb(op: data.PlotSEBOptions, data_dir: str):
    # Check execution bools first
    if len(op.points) == 0:
        print("Plot SEB is true, but no datapoints are set to plot")
        return


    # Create bins
    num_bins = int((op.bins_end - op.bins_start) / op.bins_step)
    bins = np.linspace(op.bins_start, op.bins_end, num_bins + 1)


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