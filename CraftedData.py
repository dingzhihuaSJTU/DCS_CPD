import argparse
import numpy as np

# basic curve function
from Tools import sine, square, triangle
# plot settings
from Tools import PltSettings

waveform_dict = {'sine':sine, 
              'square': square, 
              'triangle': triangle}

def ArgParse():
    parser = argparse.ArgumentParser(description='The basic settings for Synthetic Datasets. ')
    # basic config
    parser.add_argument('--length', type=int, required=False, default=5000, help='the length of series')
    parser.add_argument('--period', type=int, required=False, default=16, help='the period of series')
    parser.add_argument('--SNR', type=float, required=False, default=2, help='the signal-to-noise ratio')
    parser.add_argument('--waveform', type=str, required=False, default='sine', help='the waveform of series[sine, square, triangle, ...]')
    parser.add_argument('--change', type=str, required=False, default='mean', help='the types of statistical property changes[mean, var, fre]')
    parser.add_argument('--changerange', type=float, required=False, default=1, help='the magnitude of the change')
    args = parser.parse_args()
    return args

def CraftedChangePointMask(length: int, 
                            period: int, 
                            ) -> np.ndarray:
    """
    ## craft the change point. 
    args：
        input:
            length: the total length of curve; 
            period;
        Output:
            change point mask;
    
    """
    index = 0
    cycle = 0
    changepointmask = np.zeros((length, ))
    while index < length:
        begin = index
        index += int(period*8)
        end = min(length, index)
        changepointmask[begin: end] = cycle % 2
        cycle += 1
    return changepointmask

def CraftedData(length: int, 
                period: int, 
                b: float = 0, 
                snr: float=2, 
                waveform: str='sine', 
                change: str='mean', 
                changerange: float = 1, 
                seed: int = 2024, 
                )->tuple[np.ndarray, np.ndarray]:
    """
    args:
        Input: 
            length: the total length of curve; 
            period;
            snr: float, SNR; 
            waveform: str, ['sine', 'square', 'triangle', ...], the waveform of series;
            change: str, ['mean', 'var', 'fre'], the types of statistical property changes;
            ...
    """
    rawdata = waveform_dict[waveform](length, period)+b
    var_signal = 0.5+b**2
    var_xi = var_signal / snr
    changepointmask = CraftedChangePointMask(length, period)
    if change == "var":
        changedata = changerange*waveform_dict[waveform](length, period)+b
        snr2 = (0.5*changerange**2+b**2)/var_xi
    elif change == "fre":
        changedata = waveform_dict[waveform](length, period*changerange)+b
        snr2 = (0.5+b**2)/var_xi
    else:
        changedata = changerange+waveform_dict[waveform](length, period)+b
        snr2 = (0.5+(changerange+b)**2)/var_xi
    signal = rawdata*changepointmask + changedata*(np.ones((length, ))-changepointmask)
    
    snr1np = snr*np.ones((length, ))
    snr2np = snr2*np.ones((length, ))
    snr_np = snr1np*changepointmask + snr2np*(np.ones((length, ))-changepointmask)
    np.random.seed(seed)
    noise = np.random.normal(loc=0, scale=np.sqrt(var_xi), size=(length, ))
    
    return signal+noise, snr_np, signal

def PltCurve(data: np.ndarray, save: bool = False, savedic: str = "./data.png", picsize: tuple[int, int] = (12, 6)):
    import matplotlib.pyplot as plt
    PltSettings(picsize)
    plt.plot(data)
    if save:
        plt.save(savedic)
    else:
        plt.show()
    return 

def DataSave(data: np.ndarray, 
            length: int, 
            period: int, 
            snr: float=2, 
            waveform: str='sine', 
            change: str='mean', 
            changerange: float = 1, 
            filedict: str = "./Datasets/SyntheticDatasets/"):
    """
    Args:
        Input: 
            data: np.ndarray;
            dataname: str;
    """
    filename = "length_{}_period_{}_snr_{}_wave_{}_change_{}_changerange_{}.npy".format(str(length), str(period), str(snr), waveform, change, str(changerange))
    np.save(filedict+filename, data)
    return
