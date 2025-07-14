# This file contains methods to plot omnifold data according to settings from omnifold_plot.py


import numpy as np
import matplotlib.pyplot as plt

import src.util.data as data



########################################################################################################################

# SEI (Synthetic Error by Iteration) plot functions

########################################################################################################################



############################################################
# Data Processing
############################################################



def _sei_get_weighted_nat(op, bins, num_bins, data_dir):
    # Get nat data and weight
    nat_data_path = data.get_file_path([data_dir, op.nat_data_dir, op.nat_file_pat.get_pattern()])
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

def _sei_get_weighted_syn(op, syn_data, data_dir):
    if op.plot_weighted_syn_error:
        # Container to hold all the weights at once to avg later
        syn_weight = np.empty((op.num_tests, op.num_iterations, op.num_datapoints), dtype=np.float64)

        # Loop over weight files
        for test in range(op.num_tests):
            # Load file
            syn_weight_data_path = data.get_file_path([data_dir, op.weight_dir, op.weight_file_pat.get_pattern()])
            syn_weight_data = data.get_syn_weights(syn_weight_data_path)
            if syn_weight_data.shape != (op.num_iterations, 2, op.num_datapoints):
                raise Exception("Syn weight data has wrong shape: " + syn_weight_data.shape + "!=" +
                                str((op.num_iterations, 2, op.num_datapoints)))

            # Loop over iterations to get step 2 weights
            for iteration in range(op.num_iterations):
                syn_weight[test][iteration] = syn_weight_data[iteration][1]

            # Increment weight file pat to use next one
            if op.weight_file_pat.increment() is False:
                break

        # Average weights over tests
        syn_weight = syn_weight.mean(axis=0)

        # Reshape syn weights to apply avg to
        return syn_data[np.newaxis, :] * syn_weight
    return None

def _sei_get_re_weighted_syn(op, syn_data, data_dir):
    if op.plot_re_weighted_syn_error:
        syn_re_weight = np.empty((op.num_tests, op.num_iterations, op.num_datapoints), dtype=np.float64)

        for test in range(op.num_tests):
            syn_re_weight_data_path = data.get_file_path(
                [data_dir, op.re_weight_dir, op.re_weight_file_pat.get_pattern()])
            syn_re_weight_data = data.get_syn_weights(syn_re_weight_data_path)
            if syn_re_weight_data.shape != (op.num_iterations, 2, op.num_datapoints):
                raise Exception("Syn re_weight data has wrong shape: " + syn_re_weight_data.shape + "!=" +
                                str((op.num_iterations, 2, op.num_datapoints)))

            for iteration in range(op.num_iterations):
                syn_re_weight[test][iteration] = syn_re_weight_data[iteration][1]

            if op.re_weight_file_pat.increment() is False:
                break

        syn_re_weight = syn_re_weight.mean(axis=0)

        return syn_data[np.newaxis, :] * syn_re_weight
    return None


def _sei_process_syn(op, syn, mask_b, nat_b, denom):
    if op.plot_syn_error:
        syn_b = syn[:, mask_b]
        num = syn_b - nat_b[np.newaxis, :]

        err_b = np.where(denom != 0, num / denom * 100.0, np.nan)
        mean_error = np.nanmean(err_b, axis=1)

        std_error = 0
        if op.plot_error_bars:
            std_error = np.nanstd(err_b, axis=1)

        return mean_error, std_error
    return None, None

def _sei_process_weighted_syn(op, weighted_syn, mask_b, nat_b, denom):
    if op.plot_weighted_syn_error:
        weighted_syn_b = weighted_syn[:, mask_b]
        weighted_num = weighted_syn_b - nat_b[np.newaxis, :]

        weighted_err_b = np.where(denom != 0, weighted_num / denom * 100.0, np.nan)
        weighted_mean_error = np.nanmean(weighted_err_b, axis=1)

        weighted_std_error = 0
        if op.plot_error_bars:
            weighted_std_error = np.nanstd(weighted_err_b, axis=1)

        return weighted_mean_error, weighted_std_error
    return None, None

def _sei_process_re_weighted_syn(op, re_weighted_syn, mask_b, nat_b, denom):
    if op.plot_re_weighted_syn_error:
        re_weighted_syn_b = re_weighted_syn[:, mask_b]
        re_weighted_num = re_weighted_syn_b - nat_b[np.newaxis, :]

        re_weighted_err_b = np.where(denom != 0, re_weighted_num / denom * 100.0, np.nan)
        re_weighted_mean_error = np.nanmean(re_weighted_err_b, axis=1)

        re_weighted_std_error = 0
        if op.plot_error_bars:
            re_weighted_std_error = np.nanstd(re_weighted_err_b, axis=1)

        return re_weighted_mean_error, re_weighted_std_error
    return None, None



############################################################
# Plotting
############################################################



def _sei_plot_syn(op, mean_error, std_error):
    if op.plot_syn_error:
        shift = -0.5 * op.shift_distance
        if op.plot_weighted_syn_error != op.plot_re_weighted_syn_error:
            shift = -0.25 * op.shift_distance
        elif not (op.plot_weighted_syn_error and op.plot_re_weighted_syn_error):
            shift = 0.0


        iters = np.arange(mean_error.size)
        plt.errorbar(iters + shift,
                     mean_error,
                     yerr=std_error,
                     fmt='o',
                     capsize=4,
                     label='% Error',
                     ecolor=op.syn_color,
                     color=op.syn_color)

def _sei_plot_weighted_syn(op, weighted_mean_error, weighted_std_error):
    if op.plot_weighted_syn_error:
        weighted_shift = 0
        if op.plot_syn_error != op.plot_re_weighted_syn_error:
            if op.plot_syn_error:
                weighted_shift = 0.25 * op.shift_distance
            else:
                weighted_shift = -0.25 * op.shift_distance


        weighted_iters = np.arange(weighted_mean_error.size)
        plt.errorbar(weighted_iters + weighted_shift,
                     weighted_mean_error,
                     yerr=weighted_std_error,
                     fmt='o',
                     capsize=4,
                     label='% Error',
                     ecolor=op.weighted_color,
                     color=op.weighted_color)

def _sei_plot_re_weighted_syn(op, re_weighted_mean_error, re_weighted_std_error):
    if op.plot_re_weighted_syn_error:
        re_weighted_shift = 0.5 * op.shift_distance
        if op.plot_syn_error != op.plot_weighted_syn_error:
            re_weighted_shift = 0.25 * op.shift_distance
        elif not (op.plot_syn_error and op.plot_weighted_syn_error):
            re_weighted_shift = 0.0


        re_weighted_iters = np.arange(re_weighted_mean_error.size)
        plt.errorbar(re_weighted_iters + re_weighted_shift,
                     re_weighted_mean_error,
                     yerr=re_weighted_std_error,
                     fmt='o',
                     capsize=4,
                     label='% Error',
                     ecolor=op.re_weighted_color,
                     color=op.re_weighted_color)


def _sei_plot(op, syn_error, syn_error_std, weighted_syn_error, weighted_syn_error_std, re_weighted_syn_error, re_weighted_syn_error_std, data_dir):
    _sei_plot_syn(op, syn_error, syn_error_std)
    _sei_plot_weighted_syn(op, weighted_syn_error, weighted_syn_error_std)
    _sei_plot_re_weighted_syn(op, re_weighted_syn_error, re_weighted_syn_error_std)

    plt.axhline(0, color='k', linestyle='--', linewidth=1)
    plt.xlabel('Iteration')
    plt.ylabel('Mean % Error ±1σ')
    plt.title(op.error_plot_file_pat.get_pattern() + ': Syn vs Nat % Error')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(data.get_file_path([data_dir, op.plot_dir, op.error_plot_file_pat.get_pattern()]))
    plt.show()

    op.error_plot_file_pat.increment()


def _sei_plot_bins(op, num_bins, bin_idx, weighted_nat, syn, weighted_syn, re_weighted_syn, data_dir):
    for b in range(num_bins):
        mask_b = (bin_idx == b)
        if not np.any(mask_b):
            continue


        nat_b = weighted_nat[mask_b]
        denom = nat_b[np.newaxis, :]


        syn_error, syn_error_std = _sei_process_syn(op, syn, mask_b, nat_b, denom)
        weighted_syn_error, weighted_syn_error_std = _sei_process_weighted_syn(op, weighted_syn, mask_b, nat_b, denom)
        re_weighted_syn_error, re_weighted_syn_error_std = _sei_process_re_weighted_syn(op, re_weighted_syn, mask_b, nat_b, denom)


        _sei_plot(op, syn_error, syn_error_std, weighted_syn_error, weighted_syn_error_std, re_weighted_syn_error, re_weighted_syn_error_std, data_dir)

def _sei_plot_combined(op, num_bins, bin_idx, weighted_nat, syn, weighted_syn, re_weighted_syn, data_dir):
    syn_error = np.empty((num_bins, op.num_iterations), dtype=np.float64)
    syn_error_std = np.empty((num_bins, op.num_iterations), dtype=np.float64)
    weighted_syn_error = np.empty((num_bins, op.num_iterations), dtype=np.float64)
    weighted_syn_error_std = np.empty((num_bins, op.num_iterations), dtype=np.float64)
    re_weighted_syn_error = np.empty((num_bins, op.num_iterations), dtype=np.float64)
    re_weighted_syn_error_std = np.empty((num_bins, op.num_iterations), dtype=np.float64)

    for b in range(num_bins):
        mask_b = (bin_idx == b)
        if not np.any(mask_b):
            continue

        nat_b = weighted_nat[mask_b]
        denom = nat_b[np.newaxis, :]

        if op.plot_syn_error:
            syn_error[b, :], syn_error_std[b, :] = _sei_process_syn(op, syn, mask_b, nat_b, denom)
        if op.plot_weighted_syn_error:
            weighted_syn_error[b, :], weighted_syn_error_std[b, :] = _sei_process_weighted_syn(op, weighted_syn, mask_b, nat_b, denom)
        if op.plot_re_weighted_syn_error:
            re_weighted_syn_error[b, :], re_weighted_syn_error_std[b, :] = _sei_process_re_weighted_syn(op, re_weighted_syn, mask_b, nat_b, denom)

    if op.plot_syn_error:
        syn_error = np.nanmean(syn_error, axis=0)
        syn_error_std = np.nanstd(syn_error, axis=0)
    if op.plot_weighted_syn_error:
        weighted_syn_error = np.nanmean(weighted_syn_error, axis=0)
        weighted_syn_error_std = np.nanstd(weighted_syn_error, axis=0)
    if op.plot_re_weighted_syn_error:
        re_weighted_syn_error = np.nanmean(re_weighted_syn_error, axis=0)
        re_weighted_syn_error_std = np.nanstd(re_weighted_syn_error, axis=0)

    _sei_plot(op, syn_error, syn_error_std, weighted_syn_error, weighted_syn_error_std, re_weighted_syn_error, re_weighted_syn_error_std, data_dir)



############################################################
# Main - user facing
############################################################



def plot_sei(op: data.PlotSEIOptions, data_dir: str):
    # Check execution bools first
    if not op.plot_syn_error and not op.plot_weighted_syn_error and not op.plot_re_weighted_syn_error:
        print("Plot SEI is true, but no datapoints are set to plot")
        return


    # Create bins
    num_bins = int((op.bins_end - op.bins_start) / op.bins_step)
    bins = np.linspace(op.bins_start, op.bins_end, num_bins + 1)


    weighted_nat, bin_idx = _sei_get_weighted_nat(op, bins, num_bins, data_dir)


    # Loop over all syn data files
    for syn_d in range(op.num_syn_datasets):
        for syn_p in range(op.num_percent_deviations):
            # Get syn data
            syn_data_path = data.get_file_path([data_dir, op.syn_data_dir, op.syn_file_pat.get_pattern()])
            syn_data = data.get_syn_data(syn_data_path)
            if syn_data.shape[0] != op.num_datapoints:
                raise Exception("Syn data has wrong number of datapoints: " + syn_data.shape[0] + "!=" + op.num_datapoints)


            syn = syn_data[np.newaxis, :]
            weighted_syn = _sei_get_weighted_syn(op, syn_data, data_dir)
            re_weighted_syn = _sei_get_re_weighted_syn(op, syn_data, data_dir)


            if op.plot_combined:
                _sei_plot_combined(op, num_bins, bin_idx, weighted_nat, syn, weighted_syn, re_weighted_syn, data_dir)
            else:
                _sei_plot_bins(op, num_bins, bin_idx, weighted_nat, syn, weighted_syn, re_weighted_syn, data_dir)


            # Increment syn_file_pat to use next file on loop continue
            if op.syn_file_pat.increment() is False:
                print("Done with syn files")
                break



########################################################################################################################

# SEB (Synthetic Error by Bin) plot functions

########################################################################################################################



############################################################
# Data Processing
############################################################



def _seb_get_weighted_nat(op, bins, num_bins, data_dir):
    # Get nat data and weight
    nat_data_path = data.get_file_path([data_dir, op.nat_data_dir, op.nat_file_pat.get_pattern()])
    nat_data, nat_weight = data.get_nat_data_and_weights(nat_data_path)
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

def _seb_get_weighted_syn(op, syn_data, data_dir):
    if op.plot_weighted_syn_error:
        # Container to hold all the weights at once to avg later
        syn_weight = np.empty((op.num_tests, op.num_iterations, op.num_datapoints), dtype=np.float64)

        # Loop over weight files
        for test in range(op.num_tests):
            # Load file
            syn_weight_data_path = data.get_file_path([data_dir, op.weight_dir, op.weight_file_pat.get_pattern()])
            syn_weight_data = data.get_syn_weights(syn_weight_data_path)
            if syn_weight_data.shape != (op.num_iterations, 2, op.num_datapoints):
                raise Exception("Syn weight data has wrong shape: " + syn_weight_data.shape + "!=" +
                                str((op.num_iterations, 2, op.num_datapoints)))

            # Loop over iterations to get step 2 weights
            for iteration in range(op.num_iterations):
                syn_weight[test][iteration] = syn_weight_data[iteration][1]

            # Increment weight file pat to use next one
            if op.weight_file_pat.increment() is False:
                break

        # Average weights over tests
        syn_weight = syn_weight.mean(axis=0)

        # Reshape syn weights to apply avg to
        return syn_data[np.newaxis, :] * syn_weight
    return None

def _seb_get_re_weighted_syn(op, syn_data, data_dir):
    if op.plot_re_weighted_syn_error:
        # Container to hold all the weights at once to avg later
        syn_re_weight = np.empty((op.num_tests, op.num_iterations, op.num_datapoints), dtype=np.float64)

        # Loop over weight files
        for test in range(op.num_tests):
            # Load file
            syn_re_weight_data_path = data.get_file_path(
                [data_dir, op.re_weight_dir, op.re_weight_file_pat.get_pattern()])
            syn_re_weight_data = data.get_syn_weights(syn_re_weight_data_path)
            if syn_re_weight_data.shape != (op.num_iterations, 2, op.num_datapoints):
                raise Exception("Syn re_weight data has wrong shape: " + syn_re_weight_data.shape + "!=" +
                                str((op.num_iterations, 2, op.num_datapoints)))

            # Loop over iterations to get step 2 weights
            for iteration in range(op.num_iterations):
                syn_re_weight[test][iteration] = syn_re_weight_data[iteration][1]

            # Increment weight file pat to use next one
            if op.re_weight_file_pat.increment() is False:
                break

        # Average weights over tests
        syn_re_weight = syn_re_weight.mean(axis=0)

        # Reshape syn weights to apply avg to
        return syn_data[np.newaxis, :] * syn_re_weight
    return None


def _seb_process_weighted_syn(op, num_bins, bin_idx, weighted_nat, weighted_syn):
    mean_error = np.empty((num_bins, op.num_iterations), dtype=np.float64)
    std_error = np.empty((num_bins, op.num_iterations), dtype=np.float64)

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

def _seb_process_re_weighted_syn(op, num_bins, bin_idx, weighted_nat, re_weighted_syn):
    mean_error = np.empty((num_bins, op.num_iterations), dtype=np.float64)
    std_error = np.empty((num_bins, op.num_iterations), dtype=np.float64)

    for b in range(num_bins):
        mask_b = (bin_idx == b)
        if not mask_b.any():
            mean_error[b, :] = np.nan
            std_error[b, :] = np.nan
            continue

        nat_b = weighted_nat[mask_b]
        re_weighted_syn_B = re_weighted_syn[:, mask_b]
        num = re_weighted_syn_B - nat_b[np.newaxis, :]
        denom = nat_b[np.newaxis, :]
        err_b = np.where(denom != 0, num / denom * 100.0, np.nan)

        mean_error[b, :] = np.nanmean(err_b, axis=1)
        std_error[b, :] = np.nanstd(err_b, axis=1)

    return mean_error, std_error



############################################################
# Plotting
############################################################



def _seb_plot_weighted_syn(op, bin_centers, weighted_mean_error, weighted_std_error, iteration):
    weighted_shift = 0
    if op.plot_re_weighted_syn_error:
        weighted_shift = -0.5 * op.shift_distance

    plt.errorbar(bin_centers + weighted_shift,
                 weighted_mean_error[:, iteration],
                 yerr=weighted_std_error[:, iteration],
                 fmt='o',
                 capsize=4,
                 label=f'Iter {iteration}',
                 ecolor=op.weighted_color,
                 color=op.weighted_color)

def _seb_plot_re_weighted_syn(op, bin_centers, re_weighted_mean_error, re_weighted_std_error, iteration):
    re_weighted_shift = 0
    if op.plot_weighted_syn_error:
        re_weighted_shift = 0.5 * op.shift_distance

    plt.errorbar(bin_centers + re_weighted_shift,
                 re_weighted_mean_error[:, iteration],
                 yerr=re_weighted_std_error[:, iteration],
                 fmt='o',
                 capsize=4,
                 label=f'Iter {iteration}',
                 ecolor=op.re_weighted_color,
                 color=op.re_weighted_color)

def _seb_plot_iterations(op, num_bins, bin_idx, bins, weighted_nat, weighted_syn, re_weighted_syn, data_dir):
    weighted_mean_error, weighted_std_error = _seb_process_weighted_syn(op, num_bins, bin_idx, weighted_nat, weighted_syn)
    re_weighted_mean_error, re_weighted_std_error = _seb_process_re_weighted_syn(op, num_bins, bin_idx, weighted_nat, re_weighted_syn)

    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    for iteration in op.iterations_to_plot:
        iteration = iteration - 1
        _seb_plot_weighted_syn(op, bin_centers, weighted_mean_error, weighted_std_error, iteration)
        _seb_plot_re_weighted_syn(op, bin_centers, re_weighted_mean_error, re_weighted_std_error, iteration)


        plt.axhline(0, color='k', linestyle='--', linewidth=1)
        plt.xlabel('Bin')
        plt.ylabel('Mean % Error ±1σ')
        plt.title(op.error_plot_file_pat.get_pattern() + ': Syn vs Nat % Error')
        plt.grid(True)
        plt.tight_layout()
        plt.xlim(op.bins_start, op.bins_end)
        plt.xticks(np.arange(op.bins_start, op.bins_end + 1, op.bins_step))
        plt.savefig(data.get_file_path([data_dir, op.plot_dir, op.error_plot_file_pat.get_pattern()]))
        plt.show()

        op.error_plot_file_pat.increment()



############################################################
# Main - user facing
############################################################



def plot_seb(op: data.PlotSEBOptions, data_dir: str):
    # Check execution bools first
    if not op.plot_weighted_syn_error and not op.plot_re_weighted_syn_error:
        return


    # Create bins
    num_bins = int((op.bins_end - op.bins_start) / op.bins_step)
    bins = np.linspace(op.bins_start, op.bins_end, num_bins + 1)


    # Get nat data and binning array
    weighted_nat, bin_idx = _sei_get_weighted_nat(op, bins, num_bins, data_dir)


    # Loop over all syn data files
    for syn_d in range(op.num_syn_datasets):
        for syn_p in range(op.num_percent_deviations):
            # Get syn data
            syn_data_path = data.get_file_path([data_dir, op.syn_data_dir, op.syn_file_pat.get_pattern()])
            syn_data = data.get_syn_data(syn_data_path)
            if syn_data.shape[0] != op.num_datapoints:
                raise Exception(
                    "Syn data has wrong number of datapoints: " + syn_data.shape[0] + "!=" + op.num_datapoints)


            # Get and process data
            syn = syn_data[np.newaxis, :]
            weighted_syn = _seb_get_weighted_syn(op, syn_data, data_dir)
            re_weighted_syn = _seb_get_re_weighted_syn(op, syn_data, data_dir)


            # Plot all iterations for this syn file
            _seb_plot_iterations(op, num_bins, bin_idx, bins, weighted_nat, weighted_syn, re_weighted_syn, data_dir)


            # Increment syn_file_pat to use next file on loop continue
            if op.syn_file_pat.increment() is False:
                print("Done with syn files")
                break