# This file contains methods to plot large amounts of data

# Dictionary
"""



"""
import numpy as np
import matplotlib.pyplot as plt

import src.util.data as data



def _handle_nat_syn_error_data(op, bins, num_bins, data_dir):
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

def _handle_weighted_syn_error_data(op, syn_data, data_dir):
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

        # Average weights
        syn_weight = syn_weight.mean(axis=0)

        # Reshape syn weights to apply avg to
        return syn_data[np.newaxis, :] * syn_weight
    return None

def _handle_re_weighted_syn_error_data(op, syn_data, data_dir):
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

        # Average weights
        syn_re_weight = syn_re_weight.mean(axis=0)

        # Reshape syn weights to apply avg to
        return syn_data[np.newaxis, :] * syn_re_weight
    return None

def _plot_syn_error_bins(op, num_bins, bin_idx, weighted_nat, syn, weighted_syn, re_weighted_syn, data_dir):
    # Loop over bins and plot each one
    for b in range(num_bins):
        # mask for events in bin b
        mask_b = (bin_idx == b)
        if not np.any(mask_b):
            continue  # skip empty bins

        nat_b = weighted_nat[mask_b]
        denom = nat_b[np.newaxis, :]

        # Syn data processing
        if op.plot_syn_error:
            syn_b = syn[:, mask_b]
            num = syn_b - nat_b[np.newaxis, :]
            err_b = np.where(denom != 0, num / denom * 100.0, np.nan)
            mean_error = np.nanmean(err_b, axis=1)
            std_error = 0
            if op.plot_error_bars:
                std_error = np.nanstd(err_b, axis=1)
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

        # Weighted syn data processing
        if op.plot_weighted_syn_error:
            weighted_syn_b = weighted_syn[:, mask_b]
            weighted_num = weighted_syn_b - nat_b[np.newaxis, :]
            weighted_err_b = np.where(denom != 0, weighted_num / denom * 100.0, np.nan)
            weighted_mean_error = np.nanmean(weighted_err_b, axis=1)
            weighted_std_error = 0
            if op.plot_error_bars:
                weighted_std_error = np.nanstd(weighted_err_b, axis=1)
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

        # Re-weighted syn data processing
        if op.plot_re_weighted_syn_error:
            re_weighted_syn_b = re_weighted_syn[:, mask_b]
            re_weighted_num = re_weighted_syn_b - nat_b[np.newaxis, :]
            re_weighted_err_b = np.where(denom != 0, re_weighted_num / denom * 100.0, np.nan)
            re_weighted_mean_error = np.nanmean(re_weighted_err_b, axis=1)
            re_weighted_std_error = 0
            if op.plot_error_bars:
                re_weighted_std_error = np.nanstd(re_weighted_err_b, axis=1)
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

        # Plot finishing touches
        plt.axhline(0, color='k', linestyle='--', linewidth=1)
        plt.xlabel('Iteration')
        plt.ylabel('Mean % Error ±1σ')
        plt.title(op.error_plot_file_pat.get_pattern() + ': Syn vs Nat % Error')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(data.get_file_path([data_dir, op.plot_dir, op.error_plot_file_pat.get_pattern()]))
        plt.show()

        # Increment error plot file pattern
        op.error_plot_file_pat.increment()


def plot_sei(op: data.PlotSEIOptions, data_dir: str):
    # Check execution bools first
    if not op.plot_syn_error and not op.plot_weighted_syn_error and not op.plot_re_weighted_syn_error:
        return


    # Create bins
    num_bins = int((op.bins_end - op.bins_start) / op.bins_step)
    bins = np.linspace(op.bins_start, op.bins_end, num_bins + 1)


    weighted_nat, bin_idx = _handle_nat_syn_error_data(op, bins, num_bins, data_dir)


    # Loop over all syn data files
    for syn_d in range(op.num_syn_datasets):
        for syn_p in range(op.num_percent_deviations):
            # Get syn data
            syn_data_path = data.get_file_path([data_dir, op.syn_data_dir, op.syn_file_pat.get_pattern()])
            syn_data = data.get_syn_data(syn_data_path)
            if syn_data.shape[0] != op.num_datapoints:
                raise Exception("Syn data has wrong number of datapoints: " + syn_data.shape[0] + "!=" + op.num_datapoints)


            syn = syn_data[np.newaxis, :]
            weighted_syn = _handle_weighted_syn_error_data(op, syn_data, data_dir)
            re_weighted_syn = _handle_re_weighted_syn_error_data(op, syn_data, data_dir)


            _plot_syn_error_bins(op, num_bins, bin_idx, weighted_nat, syn, weighted_syn, re_weighted_syn, data_dir)


            # Increment syn_file_pat to use next file on loop continue
            if op.syn_file_pat.increment() is False:
                print("Done with syn files")
                break