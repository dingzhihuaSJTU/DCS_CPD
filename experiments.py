from CPDAlgorithm import Pelt, BinSeg, Window, BottomUp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from Tools import Mask2Index, PltSettings, F1Score, slimness
import time

# discrete cosine similrity
from DCS import DCS

from CraftedData import CraftedData, CraftedChangePointMask

from DCS_CPD import DCSCPD, getConCurve
from AED_BP import aed_bp_confidence_curve_window
from PLD_MC import pld_mc_confidence_curve_window

def TraditionalAlgorithm(data, tau):
    import time
    a = time.time()
    bkps_binseg = BinSeg(data, 4*tau)
    b = time.time()
    bkps_pelt = Pelt(data, 4*tau)
    c = time.time()
    bkps_window = Window(data, 4*tau)
    d = time.time()
    bkps_botup = BottomUp(data, 4*tau)
    e = time.time()

    return bkps_binseg, bkps_pelt, bkps_window, bkps_botup, b-a, c-b, d-c, e-d

def exp(length, period, waveform, b, snr, change, changerange, beta, pt):
    changemask = CraftedChangePointMask(length, period)
    changeindextrue = Mask2Index(changemask)
    data, snr_np, signal = CraftedData(length=length, period=period, snr=snr, b=b, waveform=waveform, change=change, changerange=changerange)
    a = time.time()
    p = (len(changeindextrue)-1)/length
    p_x_list = getConCurve(data, snr_np, p, beta, pt, period, False)
    changeindex = DCSCPD(p_x_list, period)
    print("Algroithm: {}, Time: {}, F1-score: {}, Len: {}".format("DCS-CPD", time.time()-a, F1Score(changeindextrue, changeindex, period/4), len(changeindex)))
    
    print()
    bkps_binseg, bkps_pelt, bkps_window, bkps_botup, time_binseg, time_pelt, time_win, time_bot = TraditionalAlgorithm(data, period)
    print("Algroithm: {}, Time: {}, F1-score: {}, Len: {}".format("BinSeg", time_binseg, F1Score(changeindex, bkps_binseg, period/4), len(bkps_binseg)))
    print("Algroithm: {}, Time: {}, F1-score: {}, Len: {}".format("Pelt", time_pelt, F1Score(changeindex, bkps_pelt, period/4), len(bkps_pelt)))
    print("Algroithm: {}, Time: {}, F1-score: {}, Len: {}".format("Window", time_win, F1Score(changeindex, bkps_window, period/4), len(bkps_window)))
    print("Algroithm: {}, Time: {}, F1-score: {}, Len: {}".format("BotUp", time_bot, F1Score(changeindex, bkps_botup, period/4), len(bkps_botup)))
    
    print()
    # Compute confidence curve
    b = time.time()
    confidence_scores_aed = aed_bp_confidence_curve_window(data, period, 100)
    print("Algroithm: {}, Time: {}".format("AED-BP", time.time()-b))
    c = time.time()
    confidence_scores_pld = pld_mc_confidence_curve_window(data, period, 100)
    print("Algroithm: {}, Time: {}".format("PLD-MC", time.time()-c))
    return p_x_list, bkps_binseg, bkps_pelt, bkps_window, bkps_botup, confidence_scores_aed, confidence_scores_pld

if __name__ == "__main__":
    length = 2000
    period = 32
    changerange = -2
    beta = 0.6
    pt=1e-3
    waveform='sine'
    b = 1
    snr = 1
    change = 'mean'
    p_x_list, bkps_binseg, bkps_pelt, bkps_window, bkps_botup, confidence_scores_aed, confidence_scores_pld = exp(length, period, waveform, b, snr, change, changerange, beta, pt)