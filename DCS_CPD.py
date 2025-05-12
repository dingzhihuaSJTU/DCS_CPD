# discrete cosine similrity
from DCS import DCS
import numpy as np
import matplotlib.pyplot as plt
from Tools import Mask2Index, PltSettings
from CraftedData import CraftedData, CraftedChangePointMask
from scipy.stats import norm
from scipy.special import gammaln

def log_comb(n, k):
    """ln C(n, k)"""
    if n < 0 or k < 0 or k > n:
        raise ValueError("Invalid values: n and k must be non-negative with k <= n.")
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)

def combination(n, k):
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result

def getThr_rho(alpha_t: float, 
               snr: float, 
               period: float, 
               ):

    mu_x = snr
    mu_y = snr+1+1/period
    # sigma_x2 = 1/tau+2*b2/(tau*sigma_xi2)
    sigma_x2 = 1/period
    # sigma_y2 = 2/(tau-1)+4*b2/(tau*sigma_xi2)
    sigma_y2 = 2/(period-1)
    mu_1 = mu_x/mu_y
    sigma_1 = sigma_x2/mu_x**2 + sigma_y2/mu_y**2

    y_t = mu_1*np.power(alpha_t/(1-alpha_t), np.sqrt(sigma_1)/1.7)

    return y_t

def P_C_A(n, tau, beta, delta, alpha):
    # print(np.log(2*tau*alpha))
    # print(0.5*np.log(tau*beta*(1-beta)*(1+2*delta))+tau/(1+2*delta)+1.266)
    # print(np.log(p/(1-p)))


    mu = 2*tau*beta
    sigma = np.sqrt(2*tau*beta*(1-beta)*(1+2*delta))
    # sigma = 2*period*beta*(1-beta)*(1+2*delta)
    p_2= norm.pdf(n, loc=mu, scale=sigma)
    p_3 = combination(2*tau, n)*alpha**n*(1-alpha)**(2*tau-n)

    return p_3/p_2

def sim_P_C_A1(n, tau, beta, delta, alpha):
    C = 0.5*np.log(tau*beta*(1-beta)*(1+2*delta))+tau*beta/((1+2*delta)*(1-beta))+1.266 + 2*tau*np.log(1-alpha)
    # c_2tau_n = (np.log(2*tau))*n - n*np.log(n)
    # tmp = -(np.log((1-alpha)/alpha)+1/((1-beta)*(1+2*delta)))*n + C
    tmp = -(np.log((1-alpha)/alpha))*n + C

    # tmp = -1/(np.log(2*tau*alpha))*(0.5*np.log(tau*beta*(1-beta)*(1+2*delta))+tau/(1+2*delta)+1.266-np.log(p/(1-p)))

    return tmp
def sim_P_C_A2(n, tau, beta, delta, alpha):
    C = 0.5*np.log(tau*beta*(1-beta)*(1+2*delta))+tau*beta/((1+2*delta)*(1-beta))+1.266 + 2*tau*np.log(1-alpha)
    # tmp = -(np.log((1-alpha)/alpha)+1/((1-beta)*(1+2*delta)))*n + C + np.log(combination(2*tau, tau))+(2*tau)/(2*beta*(1-beta)*(1+2*delta))
    tmp = -(np.log((1-alpha)/alpha))*n + C + log_comb(2*tau, tau) + (tau)/(beta*(1-beta)*(1+2*delta))
    # tmp = np.log(2*tau*alpha)*n-n+0.5*np.log(tau*beta*(1-beta)*(1+2*delta))+tau/(1+2*delta)+1.266
    # tmp = -1/(np.log(2*tau*alpha))*(0.5*np.log(tau*beta*(1-beta)*(1+2*delta))+tau/(1+2*delta)+1.266-np.log(p/(1-p)))

    return tmp


def getConCurve(data_np, snr_np, p, beta, alpha_t, period, delta, theta):
    # p_A = alpha_t+2*period*(l/(2*period)-alpha_t)*p 
    # p_B = (1-p)**(2*period)
    dcs_np = DCS(data_np, period)
    length = len(data_np)
    
    y_T = []
    for snr in snr_np:
        y_T.append(getThr_rho(alpha_t, snr, period=period))

    _dcs = dcs_np - np.array(y_T)

    p_x_list = []
    for i in range(0, length-(2*period-1)):
        n = np.sum(_dcs[i: i+2*period-1] <= 0)
        # p_x = 1/(P_C_A(n, period, beta, delta, alpha_t)*(1-p)/p+1)
        p_x = 1/(np.exp(theta*sim_P_C_A1(n, period, beta, delta, alpha_t)+(1-theta)*sim_P_C_A2(n, period, beta, delta, alpha_t))*(1-p)/p+1)

        p_x_list.append(p_x)
    p_x_list = np.concatenate([np.array(p_x_list), [1e-10]*(2*period-1)], axis=0)
    return p_x_list



def getminindex(values):
    max_value = np.max(values)
    max_indices = np.where(values == max_value)[0]
    max_index_of_min = np.mean(max_indices)
    return max_index_of_min

def DCSCPD(p_x_list, tau, thr=0.5, min_size=None):
    if min_size is None:
        min_size = 4*tau
    i = 0
    ans = []
    while i < len(p_x_list): 
        if p_x_list[i] >= thr:
            # print(i)
            p_i = i+getminindex(p_x_list[i:i+min_size])
            ans.append(p_i)
            # i+=1
            i = p_i+min_size
        else:
            i+=1
        i = int(i)
    ans.append(len(p_x_list))
    return ans

if __name__ == "__main__":
    snr = 2
    length = 2000
    period = 32
    changerange = -2
    changemask = CraftedChangePointMask(length, period)
    changeindextrue = Mask2Index(changemask)
    print(changeindextrue)
    p = (len(changeindextrue)-1)/length
    beta = 0.6
    alpha = 0.005
    theta = 0.7
    delta = period - period/(3*beta)
    change = "mean"
    data, snr_np, signal = CraftedData(length=length, period=period, snr=snr, b=1, waveform='sine', change=change, changerange=changerange)
    dcs_np = DCS(data, period)
    p_x_list = getConCurve(data, snr_np, p, beta, alpha, period, delta, theta)
    changeindex = DCSCPD(p_x_list, period)

    PltSettings((16, 16))
    plt.subplot(3, 1, 1)
    plt.plot(data)
    plt.plot(signal)
    plt.subplot(3, 1, 2)
    plt.plot(dcs_np)
    plt.plot([getThr_rho(snr, period, period) for snr in snr_np])
    plt.subplot(3, 1, 3)
    plt.plot(p_x_list)
    for index in changeindex:
        plt.axvline(index, color='r')
    for index in changeindextrue:
        plt.axvline(index, color='b')
    plt.show()
