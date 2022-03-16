import numpy as np
from sklearn.linear_model import LinearRegression

def fit_powerlaw(arr,start,end):
	x_range = np.arange(start,end+1).astype(int)
	y_range = arr[x_range-1]	# because the first eigenvalue is at index 0, so eigenval_{start} is at index (start-1)
	reg = LinearRegression().fit(np.log(x_range).reshape(-1,1),np.log(y_range).reshape(-1,1))
	y_pred = np.exp(reg.coef_*np.log(x_range).reshape(-1,1)+reg.intercept_)
	return -reg.coef_[0][0], x_range, y_pred

def robust_fit_powerlaw(arr,start,end,verbose=False):
	window = int((end-start)/10)
	slope_arr = np.array([fit_powerlaw(arr,idx,idx+window)[0] for idx in range(start,end+1)])
	robust_slope = np.median(slope_arr)
	if verbose:
		print(robust_slope)
		import matplotlib.pyplot as plt
		x_range_plt = np.arange(start,end+1).astype(int)
		y_pred_full = np.exp(-robust_slope*np.log(x_range_plt)+np.log(arr[start-1])+robust_slope*np.log(start))
		plt.loglog(np.arange(1,1+arr.shape[0]),arr); 
		plt.loglog(x_range_plt,y_pred_full); 
		plt.show()
	return robust_slope

def stringer_get_powerlaw(ss, trange):
	# COPIED FROM Stringer+Pachitariu 2018b github repo! (https://github.com/MouseLand/stringer-pachitariu-et-al-2018b/blob/master/python/utils.py)
    ''' fit exponent to variance curve'''
    logss = np.log(np.abs(ss))
    y = logss[trange][:,np.newaxis]
    trange += 1
    nt = trange.size
    x = np.concatenate((-np.log(trange)[:,np.newaxis], np.ones((nt,1))), axis=1)
    w = 1.0 / trange.astype(np.float32)[:,np.newaxis]
    b = np.linalg.solve(x.T @ (x * w), (w * x).T @ y).flatten()
    
    allrange = np.arange(0, ss.size).astype(int) + 1
    x = np.concatenate((-np.log(allrange)[:,np.newaxis], np.ones((ss.size,1))), axis=1)
    ypred = np.exp((x * b).sum(axis=1))
    alpha = b[0]
    return alpha,ypred