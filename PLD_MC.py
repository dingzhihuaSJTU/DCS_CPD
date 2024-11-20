from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

# PLD-MC Implementation with Sliding Window
def profile_likelihood_window(data, change_point, window_size, dist=norm):
    # Use only a window of data around the change point
    left_data = data[max(0, change_point-window_size):change_point]
    right_data = data[change_point:min(len(data), change_point+window_size)]
    
    # Fit parametric distribution to both parts
    left_params = dist.fit(left_data)
    right_params = dist.fit(right_data)
    
    # Calculate log-likelihoods
    left_likelihood = np.sum(np.log(dist.pdf(left_data, *left_params)))
    right_likelihood = np.sum(np.log(dist.pdf(right_data, *right_params)))
    
    return left_likelihood + right_likelihood

def deviance(parametric_likelihood):
    return -2 * parametric_likelihood

def monte_carlo_simulation_window(data, change_point, window_size, num_simulations=1000):
    n = len(data)
    simulated_deviances = []

    # Perform Monte Carlo simulations with sliding window
    for _ in range(num_simulations):
        # Generate a new simulated sample using normal distribution with same mean and std as original data
        simulated_data = np.random.normal(np.mean(data), np.std(data), size=n)
        simulated_likelihood = profile_likelihood_window(simulated_data, change_point, window_size)
        simulated_deviances.append(deviance(simulated_likelihood))
    
    return simulated_deviances

def pld_mc_confidence_curve_window(data, window_size, num_simulations=1000):
    n = len(data)
    parametric_likelihoods = []
    monte_carlo_distributions = []

    # Loop over candidate change points
    for change_point in range(1, n-1):
        likelihood = profile_likelihood_window(data, change_point, window_size)
        parametric_likelihoods.append(deviance(likelihood))
        
        # Perform Monte Carlo simulation to get deviance distribution for each change point with sliding window
        mc_deviances = monte_carlo_simulation_window(data, change_point, window_size, num_simulations)
        monte_carlo_distributions.append(mc_deviances)

    # Calculate confidence scores based on empirical deviances and Monte Carlo results
    confidence_scores = []
    for i, dev in enumerate(parametric_likelihoods):
        mc_devs = monte_carlo_distributions[i]
        # Calculate the proportion of Monte Carlo deviances greater than the empirical deviance
        confidence_score = np.sum(mc_devs > dev) / num_simulations
        confidence_scores.append(confidence_score)
    confidence_scores = np.concatenate([[1], confidence_scores, [1]], axis=0)
    confidence_scores[:10] = 1
    confidence_scores[-10:] = 1
    return np.array(confidence_scores)