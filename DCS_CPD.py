# discrete cosine similrity
from DCS import DCS

import numpy as np
import matplotlib.pyplot as plt

from Tools import Mask2Index, PltSettings

from CraftedData import CraftedData, CraftedChangePointMask

# Calculate the DCS threshold y_t based on the given probability threshold p_t
def getThr_rho(p_t: float, 
               snr: float, 
               period: float, 
               p: float, 
               beta: float = 0.6):

    alpha_t = 2*period*beta*p*p_t/((1-p)**(2*period)-p_t+2*period*p*p_t)

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


def getConCurve(data_np, snr_np, p, beta, pt, period, plot=False):

    dcs_np = DCS(data_np, period)
    length = len(data_np)
    l = 2*period*beta

    alpha_t = l*p*pt/((1-p)**(2*period)-pt+2*period*p*pt)
    p_A = alpha_t+2*period*(l/(2*period)-alpha_t)*p 
    p_B = (1-p)**(2*period)

    p_x_list = []
    for i in range(2*period-1, length):
        n = np.sum(dcs_np[i-2*period+1: i+1] <= getThr_rho(pt, snr_np[i], period, p))
        pt_ = (p_B - pt*p_A) / (1-p_A)

        p_x = 1 - (1-(1-pt)/(2*period))**n*(1-(1-pt_)/(2*period))**(2*period-n)
        

        p_x_list.append(p_x)
    p_x_list = np.concatenate([1-np.array(p_x_list), [1]*(2*period-1)], axis=0)
        
    # print(p_x_list)
    if plot:
        PltSettings((16, 12))
        plt.subplot(3, 1, 1)
        plt.plot(data_np)
        plt.subplot(3, 1, 2)
        plt.plot(dcs_np)
        plt.plot([getThr_rho(pt, snr, period, p) for snr in snr_np], label=r"$p_t < $"+f"{round(pt*100, 2)}%")
        plt.legend(loc=4)
        plt.subplot(3, 1, 3)
        plt.plot(p_x_list)

        plt.xlim((-50, length+50))
        plt.ylim((-0.1, 1.1))
        plt.axvline(length//2)
        plt.suptitle(f"SNR={snr_np[0]}, T={period}")
        plt.show()

    return p_x_list

def getminindex(values):
    # 找到数组中的最小值
    min_value = np.min(values)
    # 找到所有最小值的索引
    min_indices = np.where(values == min_value)[0]
    # 如果有多个最小值，返回索引中的最大值
    max_index_of_min = np.max(min_indices)
    return max_index_of_min
def DCSCPD(p_x_list, T, thr=0.9, min_size=None):
    if min_size is None:
        min_size = 4*T
    i = 0
    ans = []
    while i < len(p_x_list):
        if p_x_list[i] <= thr:
            # print(i)
            p_i = i+getminindex(p_x_list[i:i+min_size])
            ans.append(p_i)
            # i+=1
            i = p_i+min_size
        else:
            i+=1
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
    pt=1e-3
    change = "mean"
    data, snr_np, signal = CraftedData(length=length, period=period, snr=snr, b=1, waveform='sine', change=change, changerange=changerange)
    dcs_np = DCS(data, period)
    p_x_list = getConCurve(data, snr_np, p, beta, pt, period, False)
    changeindex = DCSCPD(p_x_list, period)

    PltSettings((16, 16))
    plt.subplot(3, 1, 1)
    plt.plot(data)
    plt.plot(signal)
    plt.subplot(3, 1, 2)
    plt.plot(dcs_np)
    plt.subplot(3, 1, 3)
    plt.plot(p_x_list)
    for index in changeindex:
        plt.axvline(index, color='r')
    for index in changeindextrue:
        plt.axvline(index, color='b')
    plt.show()