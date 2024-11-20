import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# AED-BP Implementation with Bootstrapping and Sliding Window
def empirical_likelihood_ratio_window(data, change_point, window_size):
    # Use only a window of data around the change point
    left_data = data[max(0, change_point-window_size):change_point]
    right_data = data[change_point:min(len(data), change_point+window_size)]
    
    left_mean = np.mean(left_data)
    right_mean = np.mean(right_data)
    
    left_likelihood = np.sum(np.log(norm.pdf(left_data, loc=left_mean)))
    right_likelihood = np.sum(np.log(norm.pdf(right_data, loc=right_mean)))
    
    return left_likelihood + right_likelihood

def deviance(empirical_likelihood):
    return -2 * empirical_likelihood

def bootstrap_simulation(data, change_point, window_size, num_bootstraps=1000):
    n = len(data)
    bootstrap_deviances = []

    # Perform bootstrapping with sliding window
    for _ in range(num_bootstraps):
        # Resample data with replacement within the sliding window
        resampled_data = np.random.choice(data, size=n, replace=True)
        emp_likelihood = empirical_likelihood_ratio_window(resampled_data, change_point, window_size)
        dev = deviance(emp_likelihood)
        bootstrap_deviances.append(dev)
    
    return bootstrap_deviances

def aed_bp_confidence_curve_window(data, window_size, num_bootstraps=1000):
    n = len(data)
    empirical_deviances = []
    bootstrap_distributions = []

    # Loop over candidate change points
    for change_point in range(1, n-1):
        emp_likelihood = empirical_likelihood_ratio_window(data, change_point, window_size)
        empirical_deviances.append(deviance(emp_likelihood))
        
        # Bootstrap to get deviance distribution for each change point with sliding window
        bootstrap_devs = bootstrap_simulation(data, change_point, window_size, num_bootstraps)
        bootstrap_distributions.append(bootstrap_devs)

    # Calculate confidence scores based on empirical deviances and bootstrap results
    confidence_scores = []
    for i, dev in enumerate(empirical_deviances):
        bootstrap_devs = bootstrap_distributions[i]
        # Calculate confidence score as the proportion of bootstrap deviances greater than empirical deviance
        confidence_score = np.sum(bootstrap_devs > dev) / num_bootstraps
        confidence_scores.append(confidence_score)
    confidence_scores = np.concatenate([[1], confidence_scores, [1]], axis=0)
    confidence_scores[:10] = 1
    confidence_scores[-10:] = 1
    return np.array(confidence_scores)


