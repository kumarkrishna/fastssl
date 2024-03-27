import numpy as np
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
import torch

from threadpoolctl import threadpool_limits

def rankme(eigen):
    """ Compute rankme score given a covariance eigenspectrum
    """
    l1 = np.sum(np.abs(eigen))
    eps = 1e-7
    scores = eigen / l1 + eps
    entropy = - np.sum(scores * np.log(scores))
    return np.exp(entropy)


def fit_powerlaw(arr, start, end):
    x_range = np.arange(start, end + 1).astype(int)
    y_range = arr[x_range - 1]  # because the first eigenvalue is at index 0, so eigenval_{start} is at index (start-1)
    reg = LinearRegression().fit(np.log(x_range).reshape(-1, 1), np.log(y_range).reshape(-1, 1))
    y_pred = np.exp(reg.coef_ * np.log(x_range).reshape(-1, 1) + reg.intercept_)
    return -reg.coef_[0][0], x_range, y_pred


def robust_fit_powerlaw(arr, start, end, verbose=False):
    window = int((end - start) / 10)
    slope_arr = np.array([fit_powerlaw(arr, idx, idx + window)[0] for idx in range(start, end + 1)])
    robust_slope = np.median(slope_arr)
    if verbose:
        print(robust_slope)
        import matplotlib.pyplot as plt
        x_range_plt = np.arange(start, end + 1).astype(int)
        y_pred_full = np.exp(
            -robust_slope * np.log(x_range_plt) + np.log(arr[start - 1]) + robust_slope * np.log(start))
        plt.loglog(np.arange(1, 1 + arr.shape[0]), arr);
        plt.loglog(x_range_plt, y_pred_full);
        plt.show()
    return robust_slope

def stringer_get_powerlaw(ss, trange):
    # COPIED FROM Stringer+Pachitariu 2018b github repo! (https://github.com/MouseLand/stringer-pachitariu-et-al-2018b/blob/master/python/utils.py)
    ''' fit exponent to variance curve'''
    logss = np.log(np.abs(ss))
    y = logss[trange][:, np.newaxis]
    trange += 1
    nt = trange.size
    x = np.concatenate((-np.log(trange)[:, np.newaxis], np.ones((nt, 1))), axis=1)
    w = 1.0 / trange.astype(np.float32)[:, np.newaxis]
    b = np.linalg.solve(x.T @ (x * w), (w * x).T @ y).flatten()

    allrange = np.arange(0, ss.size).astype(int) + 1
    x = np.concatenate((-np.log(allrange)[:, np.newaxis], np.ones((ss.size, 1))), axis=1)
    ypred = np.exp((x * b).sum(axis=1))
    alpha = b[0]
    max_range = 500 if len(ss) >= 512 else len(
        ss) - 10  # subtracting 10 here arbitrarily because we want to avoid the last tail!
    fit_R2 = r2_score(y_true=logss[trange[0]:max_range], y_pred=np.log(np.abs(ypred))[trange[0]:max_range])
    try:
        fit_R2_100 = r2_score(y_true=logss[trange[0]:100], y_pred=np.log(np.abs(ypred))[trange[0]:100])
    except:
        fit_R2_100 = None
    return alpha, ypred, fit_R2, fit_R2_100

def generate_activations_prelayer(net,layer,data_loader,use_cuda=False,dim_thresh=10000,test_run=False):
    pool_size = 1
    activations = []
    def hook_fn(m,i,o):
        activations.append(i[0].cpu().numpy())
    handle = layer.register_forward_hook(hook_fn)
    
    if use_cuda:
        net = net.cuda()    
    net.eval()
    for i, inp in enumerate(tqdm(data_loader)):
        (images, labels) = (inp[0], inp[1]) # discarding any extra item from the batch
        if use_cuda:
            images = images.cuda()
        with torch.no_grad():
            output = net(images)
        if i==10 and test_run:
            break
    handle.remove()
    activations_np = np.vstack(activations)     # assuming first dimension is num_examples: batches x batch_size x <feat_dims> --> num_examples x <feat_dims>
    return activations_np

def generate_activations_prelayer_torch(net,layer,data_loader,use_cuda=False,dim_thresh=10000,test_run=False):
    pool_size = 1
    activations = []
    def hook_fn(m,i,o):
        activations.append(i[0])
    handle = layer.register_forward_hook(hook_fn)

    if use_cuda:
        net = net.cuda()
    net.eval()
    for i, inp in enumerate(tqdm(data_loader)):
        (images, labels) = (inp[0], inp[1]) # discarding any extra item from the batch
        if isinstance(images, (list, tuple)):
            images = images[0]
        if use_cuda:
            images = images.cuda()
        with torch.no_grad():
            output = net(images)
        if i==10 and test_run:
            break
    handle.remove()
    activations_torch = torch.vstack(activations)     # assuming first dimension is num_examples: batches x batch_size x <feat_dims> --> num_examples x <feat_dims>
    del activations
    return activations_torch

def get_eigenspectrum_torch(activations,max_eigenvals=2048):
    feats = activations.reshape(activations.shape[0],-1)
    feats_center = feats - feats.mean(dim=0)
    covariance = feats_center.T @ feats_center / activations.shape[0]
    eigenspectrum = torch.linalg.svdvals(covariance).cpu().numpy()
    return eigenspectrum

def generate_activations_prelayer_batch(net,layer,images,pool_transform=None):
    activations = []
    def hook_fn(m, i, o):
        activations.append(i[0])
    handle = layer.register_forward_hook(hook_fn)
    if pool_transform is not None:
        images = pool_transform(images)
    output = net(images)
    handle.remove()
    return activations[0]

def get_eigenspectrum(activations_np,max_eigenvals=2048):
    with threadpool_limits(limits=1):
        feats = activations_np.reshape(activations_np.shape[0],-1)
        feats_center = feats - feats.mean(axis=0)
        pca = PCA(n_components=min(max_eigenvals, feats_center.shape[0], feats_center.shape[1]), svd_solver='full')
        pca.fit(feats_center)
        eigenspectrum = pca.explained_variance_ratio_
    return eigenspectrum

def stringer_get_powerlaw_batch(net, layer, data_loader, trange, use_cuda=False, test_run=False):
    alpha_arr = []
    R2_arr = []
    R2_100_arr = []
    if use_cuda:
        net = net.cuda()
    net.eval()
    for i, inp in enumerate(tqdm(data_loader)):
        (images, labels) = (inp[0], inp[1]) # discarding any extra item from the batch
        # ignore last batch if incomplete
        # print(i,len(data_loader),len(images),data_loader.batch_size)
        if i == len(data_loader) - 1 and len(images) < data_loader.batch_size:
            break
        trange_orig = trange.copy()
        if use_cuda:
            images = images.cuda()
        with torch.no_grad():
            feats = generate_activations_prelayer_batch(net=net, layer=layer, images=images)
        feats_np = feats.cpu().numpy()
        feats_eig = get_eigenspectrum(feats_np)
        # print(trange.min(),trange.max(),feats_eig.shape)
        alpha_batch, _, R2_batch, R2_100_batch = stringer_get_powerlaw(ss=feats_eig, trange=trange_orig)
        alpha_arr.append(alpha_batch)
        R2_arr.append(R2_batch)
        R2_100_arr.append(R2_100_batch)
        if i == 2 and test_run:
            break
    return np.array(alpha_arr), np.array(R2_arr), np.array(R2_100_arr)
