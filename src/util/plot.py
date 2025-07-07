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

    # Loop over syn data
    percent = 1
    set = 1
    while True:
        # Get syn data
        syn_data = get_syn_150000_data(get_file_path(op.syn_data_file_pattern, op.syn_data_dir))

        # Loop over syn weights iterations
        syn_weight = np.empty((10, 5, 150000), dtype=np.float64)
        for test in range(10):
            for iteration in range(5):
                # Get step 2 weights - for this plot we only care about step 2 of the omnifold process
                syn_weight[test][iteration] = (
                    get_syn_150000_weights(get_file_path(op.weights_file_pattern, op.weights_dir)))[iteration, 1]

            # Increment weights file pattern
            if not op.weights_file_pattern.increment():
                raise Exception("Error: weights file pattern did not increment before tests was done")

        # Average syn weights to get [iteration, weight] - so avg over tests for iteration
        syn_avg_weight = syn_weight.mean(axis=0)

        # Reshape syn data to apply weights
        syn_data = syn_data[np.newaxis, :]    # Shape (1, data)

        # Apply weights
        weighted_syn = syn_data * syn_avg_weight    # Shape (iteration, weights)

        # Calculate error%
        num = weighted_syn - weighted_nat[np.newaxis, :]
        denom = weighted_nat[np.newaxis, :]
        percent_error = np.where(
            denom != 0,
            num / denom * 100.0,
            np.nan
        )  # Shape (5,150000)

        # Mean % error per iteration (skip NaNs)
        mean_error = np.nanmean(percent_error, axis=1)  # Shape (iteration)

        # Std. deviation per iteration
        std_error = np.nanstd (percent_error, axis=1)  # Shape (iteration)



        # Graph data
        iters = np.arange(mean_error.size)  # Get simple x-values [0,1,2,3,4]

        plt.errorbar(iters,
                     mean_error,
                     yerr=std_error,
                     fmt='o',
                     capsize=4,
                     label='% Error')
        plt.axhline(0, color='k', linestyle='--', linewidth=1)
        plt.xlabel('Iteration')
        plt.ylabel('Mean % Error ±1σ')
        plt.title('Set ' + str(set) + ' ' + str(percent) + 'Percent: Syn vs Nat % Error')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(get_file_path(op.plots_file_pattern, op.plots_dir))
        plt.show()



        # Increment syn_data and plot files
        op.plots_file_pattern.increment()
        if not op.syn_data_file_pattern.increment():
            break

        if percent >= 5:
            percent = 1
            set += 1
        else:
            percent += 1