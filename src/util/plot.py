# This file contains methods to plot large amounts of data

# Dictionary
"""



"""

# Used for plotting options/settings
import src.util.options as op

# Used for dynamic file names
from src.util.file_patterns import TokenType, Token, FilePattern, get_file_path

# Used to pull data
from src.util.data import get_nat_150000_data, get_syn_150000_data, get_syn_150000_weights

# Used for data management
import numpy as np

# Used for various plotting functions
import matplotlib.pyplot as plt



# Plots the average error% of all tests for each iteration
def plot_tests_for_iteration_error_over_energy():
    # Get nat data from file
    nat_data, nat_weights = get_nat_150000_data(get_file_path(op.nat_data_file_pattern, op.nat_data_dir))
    weighted_nat = nat_data * nat_weights

    # Bin nat data
    n_bins = 5
    bins = np.linspace(0, 100, n_bins + 1)
    bin_idx = np.digitize(nat_data, bins) - 1
    bin_idx[bin_idx == n_bins] = n_bins - 1

    # Loop over syn data
    plots_name_pattern = FilePattern([
        Token(TokenType.INCREMENTAL, 'syn', 1, (1, 2, 1)),
        Token(TokenType.INCREMENTAL, ' ', 1, (1, 5, 1)),
        Token(TokenType.STATIC, 'Percent')
    ])
    while True:
        # Get syn data
        syn_data = get_syn_150000_data(get_file_path(op.syn_data_file_pattern, op.syn_data_dir))

        # Loop over syn weights iterations
        syn_weight = np.empty((10, 5, 150000), dtype=np.float64)
        re_weight = np.empty((10, 5, 150000), dtype=np.float64)
        for test in range(10):
            for iteration in range(5):
                # Get step 2 weights - for this plot we only care about step 2 of the omnifold process
                syn_weight[test][iteration] = (
                    get_syn_150000_weights(get_file_path(op.weights_file_pattern, op.weights_dir))
                )[iteration, 1]

                re_weight[test][iteration] = (
                    get_syn_150000_weights(get_file_path(op.re_weight_file_pattern, op.re_weights_dir))
                )[iteration, 1]

            # Increment weights file pattern
            if not op.weights_file_pattern.increment() and iteration != 4 and test != 9:
                raise Exception("Error: weights file pattern ran out of increments too early")
            if not op.re_weight_file_pattern.increment() and iteration != 4 and test != 9:
                raise Exception("Error: re_weight file pattern ran out of increments too early")

        # Average syn weights to get [iteration, weight] - so avg over tests for iteration
        syn_avg_weight = syn_weight.mean(axis=0)
        re_weight_avg = re_weight.mean(axis=0)

        # Reshape syn data to apply weights
        syn_data = syn_data[np.newaxis, :]

        # Apply weights
        weighted_syn = syn_data * syn_avg_weight
        re_weighted_syn = syn_data * re_weight_avg

        # Loop over bins
        for b in range(n_bins):
            # mask for events in bin b
            mask_b = (bin_idx == b)
            if not np.any(mask_b):
                continue  # skip empty bins

            # slice natural andght! synthetic to those events
            nat_b = weighted_nat[mask_b]
            syn_b = weighted_syn[:, mask_b]
            re_syn_b = re_weighted_syn[:, mask_b]

            # percent-error per iteration × event
            num = syn_b - nat_b[np.newaxis, :]
            denom = nat_b[np.newaxis, :]
            err_b = np.where(denom != 0,
                             num / denom * 100.0,
                             np.nan)
            re_num = re_syn_b - nat_b[np.newaxis, :]
            re_err_b = np.where(denom != 0,
                                re_num / denom * 100.0,
                                np.nan)

            # stats over events
            mean_error = np.nanmean(err_b, axis=1)
            std_error = np.nanstd(err_b, axis=1)
            re_mean_error = np.nanmean(re_err_b, axis=1)
            re_std_error = np.nanstd(re_err_b, axis=1)


            # Graph data
            iters = np.arange(mean_error.size)
            plt.errorbar(iters - 0.1,
                         mean_error,
                         yerr=std_error,
                         fmt='o',
                         capsize=4,
                         label='% Error',
                         ecolor='b')
            re_iters = np.arange(re_mean_error.size)
            plt.errorbar(re_iters + 0.1,
                         re_mean_error,
                         yerr=re_std_error,
                         fmt='o',
                         capsize=4,
                         label='% Error',
                         ecolor='r')
            plt.axhline(0, color='k', linestyle='--', linewidth=1)
            plt.xlabel('Iteration')
            plt.ylabel('Mean % Error ±1σ')
            plt.title(str(bins[b]) + '-' + str(bins[b+1]) + ' ' + plots_name_pattern.get_full_pattern() + ': Syn vs Nat % Error')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(get_file_path(op.plots_file_pattern, op.plots_dir))
            plt.show()

            # Increment plot file path and name
            op.plots_file_pattern.increment()


        plots_name_pattern.increment()



        # Increment syn_data
        if not op.syn_data_file_pattern.increment():
            break
