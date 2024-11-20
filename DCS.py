import numpy as np

def covxy(x:np.ndarray, y:np.ndarray)->float:
    """
    ## calculate the covariance
    Args:
        Input: 
            x, y: the series to calculate the covariance
    """
    covxy = np.mean((x - x.mean()) * (y - y.mean()))+np.mean(x)*np.mean(y)
    return covxy

def CosineSimilrity(x:np.ndarray, y:np.ndarray)->float:
    cov_xy = covxy(x, y)
    return cov_xy/np.sqrt(np.mean(x**2)*np.mean(y**2))



def DCS(x: np.ndarray, windowsize: int)->np.ndarray:
    """
    ## Do the DCA for series

    Args:
        Input: 
            x: the series to calculate the DCS
            windowsize: the size of sliding windows
            cov: default for False. If True, output is the covariance series. 
        Output:
            Dynamic correlate series
    """
    n = len(x)
    x = np.array(x)

    corr_list = [1]*(2*windowsize-1)
    for i in range(n-2*windowsize+1):
        x1 = x[i: i+windowsize]
        x2 = x[i+windowsize: i+2*windowsize]
        corr_list.append(CosineSimilrity(x1, x2))
    return np.array(corr_list)
