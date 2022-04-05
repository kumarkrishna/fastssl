import numpy as np
from sklearn.linear_model import LinearRegression


class PowerLaw:
    """
    Fits a power law to batch of input samples over a defined window. 
    Args:
        variant : str (optional) 
            # supports ['standard', 'robust', 'stringer']
    """
    def __init__(self, variant='standard'):
        self.variant = variant

    def fit(self, data, trange):
        """
        Args:
            data : List[int]
            trange : tuple(int, int)
        Returns:
            alpha: float
            y_pred: np.array
        """
        if self.variant == 'standard':
            return self.fit_powerlaw(data, trange)
        elif self.variant == 'robust':
            return self.robust_fit_powerlaw(data, trange)
        elif self.variant == 'stringer':
            return self.stringer_fit_powerlaw(data, trange)
    
    def fit_powerlaw(self, data, trange):
        start, end = trange
        x_range = np.arange(start, end+1).astype(int)

        # because the first eigenvalue is at index 0, 
        # so eigenval_{start} is at index (start-1)
        y_range = data[x_range-1]	

        # build regressor
        regression = LinearRegression()
        reg = regression.fit(
            np.log(x_range).reshape(-1,1),
            np.log(y_range).reshape(-1,1))
        
        # get predictions in standard scale
        y_pred = np.exp(reg.coef_ * np.log(x_range).reshape(-1,1) + reg.intercept_)

        alpha = -reg.coef_[0][0]
        return alpha, y_pred

    def robust_fit_powerlaw(self, data, trange, verbose=False):
        start, end = trange
        window = int((end-start)/10)
        slope_arr = np.array(
            [fit_powerlaw(data, idx, idx+window)[0] for idx in range(start,end+1)])
        alpha = np.median(slope_arr)

        xrange = np.arange(start, end+1).astype(int)
        y_pred = np.exp(-robust_slope*np.log(x_range_plt) + np.log(
            data[start-1])+robust_slope*np.log(start))

        if verbose:
            # plot the powerlaw fit
            print("slope of fit to {}".format(alpha))
            import matplotlib.pyplot as plt
            plt.loglog(np.arange(1, 1+data.shape[0]), data); 
            plt.loglog(xrange, y_pred); 
            plt.show()
        return alpha, y_pred

    def stringer_fit_powerlaw(self, data, trange):
        # COPIED FROM Stringer+Pachitariu 2018b github repo! (https://github.com/MouseLand/stringer-pachitariu-et-al-2018b/blob/master/python/utils.py)
        """fit exponent to variance curve"""
        start, end = trange
        logss = np.log(np.abs(data))
        trange = np.arange(start, end)
        y = logss[trange][:,np.newaxis]
        trange += 1
        nt = trange.size
        x = np.concatenate((-np.log(trange)[:,np.newaxis], np.ones((nt,1))), axis=1)
        w = 1.0 / trange.astype(np.float32)[:,np.newaxis]
        b = np.linalg.solve(x.T @ (x * w), (w * x).T @ y).flatten()
        
        allrange = np.arange(0, data.size).astype(int) + 1
        x = np.concatenate((-np.log(allrange)[:,np.newaxis], np.ones((data.size,1))), axis=1)
        ypred = np.exp((x * b).sum(axis=1))
        alpha = b[0]
        return alpha, ypred