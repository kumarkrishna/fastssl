import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import os, glob
from linclab_utils import plot_utils
from tqdm import tqdm
from sklearn.metrics import r2_score
import matplotlib as mpl
from matplotlib import cm
from typing import Mapping
import argparse

plot_utils.linclab_plt_defaults(
    font="Arial", fontdir=os.path.expanduser("~") + "/Projects/fonts"
)  # Run locally, not from cluster


parser = argparse.ArgumentParser(
    description="Arguments for plotting alpha tracking results"
)
parser.add_argument("--flag_debug", action="store_true")
parser.add_argument("--use_old_alpha", action="store_true")
parser.add_argument("--R2_thresh", type=float, default=0.95)
parser.add_argument("--ssl_alg", type=str, default="barlowtwins")
parser.add_argument("--dataset_ssl", type=str, default="cifar10")
parser.add_argument("--hparam_filter_val", type=float, default=-1)
parser.add_argument("--plot_control", action="store_true")
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()

flag_debug = args.flag_debug  # False
use_old_alpha = args.use_old_alpha  # True
plot_control = args.plot_control  # False
verbose = args.verbose  # False
R2_thresh = args.R2_thresh  # 0.95
ssl_alg = args.ssl_alg  # 'simclr'
assert ssl_alg in ["barlowtwins", "simclr"], "Unrecognized ssl algorithm ({})".format(
    ssl_alg
)
hparam_filter_val = args.hparam_filter_val
model_arch = "resnet50"
dataset_ssl = args.dataset_ssl  # 'cifar10'
assert dataset_ssl in ["cifar10", "stl10"], "Unrecognized ssl dataset ({})".format(
    dataset_ssl
)
ckpt_top_dir = "adaptiveSSL_ckpt"
ckpt_main_dir = "{}checkpoints_{}adaptiveLamda_arch_hparams_track_alpha_{}".format(
    "simclr_" if ssl_alg == "simclr" else "",
    "no" if plot_control else "",
    dataset_ssl,
)
track_ckpt_dir = os.path.join(ckpt_top_dir, ckpt_main_dir, model_arch)


def stringer_get_powerlaw(ss, trange):
    # COPIED FROM Stringer+Pachitariu 2018b github repo! (https://github.com/MouseLand/stringer-pachitariu-et-al-2018b/blob/master/python/utils.py)
    """fit exponent to variance curve"""
    logss = np.log(np.abs(ss))
    y = logss[trange][:, np.newaxis]
    trange += 1
    nt = trange.size
    x = np.concatenate((-np.log(trange)[:, np.newaxis], np.ones((nt, 1))), axis=1)
    w = 1.0 / trange.astype(np.float32)[:, np.newaxis]
    b = np.linalg.solve(x.T @ (x * w), (w * x).T @ y).flatten()

    allrange = np.arange(0, ss.size).astype(int) + 1
    x = np.concatenate(
        (-np.log(allrange)[:, np.newaxis], np.ones((ss.size, 1))), axis=1
    )
    ypred = np.exp((x * b).sum(axis=1))
    alpha = b[0]
    max_range = (
        500 if len(ss) >= 512 else len(ss) - 10
    )  # subtracting 10 here arbitrarily because we want to avoid the last tail!
    fit_R2 = r2_score(
        y_true=logss[trange[0] : max_range],
        y_pred=np.log(np.abs(ypred))[trange[0] : max_range],
    )
    try:
        fit_R2_range = r2_score(
            y_true=logss[trange], y_pred=np.log(np.abs(ypred))[trange]
        )
    except:
        fit_R2_range = None
    return alpha, ypred, fit_R2, fit_R2_range


def plot_alpha_fit(
    data_dict: Mapping,
    trange: np.ndarray,
    hyperparam_dict: Mapping,
    epoch_val: int = None,
):
    eigen = data_dict["eigenspectrum"]
    al, ypred, r2, r2_range = stringer_get_powerlaw(eigen, trange)
    print("{:.4f}, {:.4f}, {:.4f}".format(al, r2, r2_range))
    plt.figure(figsize=(10, 6))
    plt.loglog(np.arange(1, 1 + 1200), eigen[:1200])
    plt.loglog(np.arange(1, 1 + 1200), ypred[:1200])
    hyperparam_str = ""
    for key, val in hyperparam_dict.items():
        hyperparam_str += "{} = ".format(key)
        if isinstance(val, int) or float(val).is_integer():
            hyperparam_str += "{}, ".format(int(val))
        else:
            hyperparam_str += "{:.2e}, ".format(val)

    plt.title(
        "{}{}alpha={:.3f}".format(
            hyperparam_str,
            " epoch = {}, ".format(epoch_val) if epoch_val is not None else "",
            al,
        )
    )
    plt.show()


def plot_colorplot(
    pdim_arr,
    lamda_arr,
    attribute_dict,
    attribute_label,
    vmin=None,
    vmax=None,
    cmap="seismic",
):
    plotting_arr = np.empty((len(pdim_arr), len(lamda_arr)))
    plotting_arr[:] = np.nan
    for pidx, pdim in enumerate(pdim_arr):
        for lidx, lamda in enumerate(lamda_arr):
            try:
                if type(attribute_dict[pdim][lamda]) is list:
                    plotting_arr[pidx, lidx] = np.mean(
                        np.array(attribute_dict[pdim][lamda])
                    )
                else:
                    plotting_arr[pidx, lidx] = attribute_dict[pdim][lamda]
            except:
                pass
    plt.figure()
    plt.imshow(plotting_arr, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
    plt.colorbar()
    plt.title(attribute_label)
    plt.yticks(np.arange(len(pdim_arr)), pdim_arr)
    plt.xticks(
        np.arange(0, len(lamda_arr), 2), lamda_arr[np.arange(0, len(lamda_arr), 2)]
    )
    plt.gca().set_xticklabels(
        FormatStrFormatter("%.2e").format_ticks(
            lamda_arr[np.arange(0, len(lamda_arr), 2)]
        )
    )
    plt.ylabel("Projector Dimensionality")
    plt.xlabel(r"$\lambda$")


def plot_scatterplot(
    attribute1_dict,
    attribute1_label,
    attribute2_dict,
    attribute2_label,
    filter_dict=None,
    vmin=None,
    vmax=None,
    hmin=None,
    hmax=None,
):
    plt.figure()
    markers_arr = ["o", "^", "x", "s", "d", "v", "*"]
    for pidx, pdim in enumerate(attribute1_dict.keys()):
        attr1_arr = []
        attr1_err_arr = []
        attr2_arr = []
        attr2_err_arr = []
        for lidx, lamda in enumerate(attribute1_dict[pdim].keys()):
            if (
                filter_dict is not None
                and np.nanmean(np.array(filter_dict[pdim][lamda])) < R2_thresh
            ):
                continue
            if type(attribute1_dict[pdim][lamda]) is list:
                attr1_arr.append(np.mean(np.array(attribute1_dict[pdim][lamda])))
                attr1_err_arr.append(np.std(np.array(attribute1_dict[pdim][lamda])))
            else:
                attr1_arr.append(attribute1_dict[pdim][lamda])
                attr1_err_arr.append(0)
            if type(attribute2_dict[pdim][lamda]) is list:
                attr2_arr.append(np.mean(np.array(attribute2_dict[pdim][lamda])))
                attr2_err_arr.append(np.std(np.array(attribute2_dict[pdim][lamda])))
            else:
                attr2_arr.append(attribute2_dict[pdim][lamda])
                attr2_err_arr.append(0)

        plt.errorbar(
            x=attr1_arr,
            y=attr2_arr,
            xerr=attr1_err_arr,
            yerr=attr2_err_arr,
            marker=markers_arr[pidx % len(markers_arr)],
            ls="",
            label="pdim={}".format(int(pdim)),
        )
    plt.title("{} vs {}".format(attribute1_label, attribute2_label))
    plt.xlabel("{}".format(attribute1_label))
    plt.ylabel("{}".format(attribute2_label))
    plt.legend()
    plt.ylim([vmin, vmax])
    plt.xlim([hmin, hmax])


track_alpha_files = glob.glob(os.path.join(track_ckpt_dir, "*"))
if ssl_alg == "simclr":
    sorting_order = [
        i[0]
        for i in sorted(
            enumerate(track_alpha_files),
            key=lambda x: float(os.path.basename(x[1]).split("pdim_")[-1].split("_")[0])
            + float(os.path.basename(x[1]).split("temp_")[-1].split("_")[0])
            + 1000 * float(os.path.basename(x[1]).split("bsz_")[-1].split("_")[0]),
        )
    ]
else:
    sorting_order = [
        i[0]
        for i in sorted(
            enumerate(track_alpha_files),
            key=lambda x: float(os.path.basename(x[1]).split("pdim_")[-1].split("_")[0])
            + float(os.path.basename(x[1]).split("lambd_")[-1].split("_")[0]),
        )
    ]
files_sorted = [track_alpha_files[idx] for idx in sorting_order]

res_all_files = []
alpha_correction_due_to_minibatches = 0.0  # 0.1
min_accuracy = 100
max_accuracy = -100
for fidx, file in enumerate(tqdm(files_sorted)):
    try:
        if ssl_alg == "simclr":
            temp_val = float(os.path.basename(file).split("temp_")[-1].split("_")[0])
            bsz_val = float(os.path.basename(file).split("bsz_")[-1].split("_")[0])
            pdim_val = float(os.path.basename(file).split("pdim_")[-1].split("_")[0])
            hyperparam_vals = {"temp": temp_val, "batchSize": bsz_val, "pdim": pdim_val}
            hparam_filter_condition = bsz_val
        else:
            # breakpoint()
            lamda_val = float(os.path.basename(file).split("lambd_")[-1].split("_")[0])
            pdim_val = float(os.path.basename(file).split("pdim_")[-1].split("_")[0])
            hyperparam_vals = {"lamda": lamda_val, "pdim": pdim_val}
            hparam_filter_condition = pdim_val
        SSL_fnames = os.path.join(
            file,
            "results_{}_alpha_{}_100*.npy".format(
                dataset_ssl, "SimCLR" if ssl_alg == "simclr" else "ssl"
            ),
        )
        SSL_files = glob.glob(SSL_fnames)
        epoch_alpha_dict = {}
        epoch_lamda_dict = {}
        for SSL_fname in SSL_files:
            # SSL_fname = SSL_files[0]
            if os.path.exists(SSL_fname):
                # add init lamda_val in epoch_lamda_dict
                if 0 not in epoch_lamda_dict.keys():
                    epoch_lamda_dict[0] = [lamda_val]
                else:
                    epoch_lamda_dict[0].append(lamda_val)
                SSL_file = np.load(SSL_fname, allow_pickle=True).item()
                # breakpoint()
                eigenspectrum_series = SSL_file["eigenspectrum"]
                R2_100_series = SSL_file["R2_100"]
                alpha_series = SSL_file["alpha"]
                if "lambda" in SSL_file.keys():
                    lamda_series = SSL_file["lambda"]
                else:
                    lamda_series = None
                for idx, (epoch, eigen) in enumerate(eigenspectrum_series):
                    assert (
                        epoch == R2_100_series[idx][0]
                    ), "Epochs for R2 series don't match {} vs {}".format(
                        epoch, R2_100_series[idx][0]
                    )
                    # if (not use_old_alpha and R2_100_series[idx][1]<R2_thresh):
                    try:
                        # try to compute new alpha
                        # unless explicitly told not to :)
                        assert not use_old_alpha
                        # range_init = 5
                        # range_init = 3
                        range_init = 11
                        range_end = 50
                        # range_end = 30
                        alpha, ypred, R2, r2_range = stringer_get_powerlaw(
                            eigen, np.arange(range_init, range_end)
                        )
                        while r2_range < R2_thresh:
                            range_end = int(range_end // 2)
                            if range_end <= range_init:
                                print(
                                    "No good powerlaw fit for hyperparams:",
                                    hyperparam_vals,
                                    "epoch = {}".format(epoch),
                                )
                                r2_range = None
                                break
                            alpha, ypred, R2, r2_range = stringer_get_powerlaw(
                                eigen, np.arange(range_init, range_end)
                            )
                        if r2_range:
                            R2_100 = r2_range
                        else:
                            # no powerlaw fit
                            continue
                    except:
                        assert (
                            epoch == alpha_series[idx][0]
                        ), "Epochs for alpha series don't match {} vs {}".format(
                            epoch, alpha_series[idx][0]
                        )
                        alpha = alpha_series[idx][1]
                    if epoch not in epoch_alpha_dict.keys():
                        epoch_alpha_dict[epoch] = [alpha]
                    else:
                        epoch_alpha_dict[epoch].append(alpha)
                    if epoch > 0 and lamda_series is not None:
                        if epoch not in epoch_lamda_dict.keys():
                            epoch_lamda_dict[epoch] = [lamda_series[idx - 1][1]]
                        else:
                            epoch_lamda_dict[epoch].append(lamda_series[idx - 1][1])
                    if flag_debug:
                        plot_alpha_fit(
                            {"eigenspectrum": eigenspectrum_series[idx][1]},
                            trange=np.arange(11, 50),
                            hyperparam_dict=hyperparam_vals,
                            epoch_val=epoch,
                        )

            else:
                continue
        linear_files = glob.glob(
            os.path.join(file, "results_{}_alpha_linear_200*".format(dataset_ssl))
        )
        accuracy_arr = []
        for linear_fname in linear_files:
            linear_dict = np.load(linear_fname, allow_pickle=True).item()
            test_acc_arr = np.array(linear_dict["test_acc_1"])
            final_test_acc = test_acc_arr[-1]
            accuracy_arr.append(final_test_acc)
        accuracy_arr = np.array(accuracy_arr)
        # if np.mean(accuracy_arr) < 60:
        # 	continue
        if np.mean(accuracy_arr) > max_accuracy:
            max_accuracy = np.mean(accuracy_arr)
        if np.mean(accuracy_arr) < min_accuracy:
            min_accuracy = np.mean(accuracy_arr)

        if hparam_filter_val > 0:
            if hparam_filter_condition != hparam_filter_val:
                continue
        res_all_files.append((epoch_alpha_dict, epoch_lamda_dict, accuracy_arr))
        if verbose:
            tqdm.write(
                "lamda = {}, alpha = {}, accuracy = {}".format(
                    lamda_val,
                    np.mean(list(epoch_alpha_dict.values())[-1]),
                    np.mean(accuracy_arr),
                )
            )
        if flag_debug:
            plot_alpha_fit(
                linear_dict,
                trange=np.arange(11, 50),
                hyperparam_dict=hyperparam_vals,
            )
            breakpoint()

    except:
        breakpoint()
        pass

N_colors = 20
min_acc_cbar = 77 if ssl_alg == "simclr" else 70
cscale = cm.get_cmap("coolwarm", N_colors)

# all_fin_accuracy_vals = [res[2] for res in res_all_files]
# all_fin_alpha_vals = [res[0][100] for res in res_all_files]
# all_init_lambda_vals = [res[1][0] for res in res_all_files]

plt.figure(r"Scatter plot: $\alpha$ vs accuracy")
plt.ylabel("Accuracy")
plt.xlabel(r"$\alpha$")
plt.grid("on")
plt.ylim([min_acc_cbar, max_accuracy + 2])

plt.figure(r"Scatter plot: $\lambda$ vs accuracy")
plt.xlabel(r"$\lambda$")
plt.xscale("log")
plt.ylabel("Accuracy")
plt.grid("on")
plt.ylim([min_acc_cbar, max_accuracy + 2])

accuracy_range = max_accuracy - min_acc_cbar
for epoch_alpha, epoch_lamda, final_accuracy in res_all_files:
    color_idx = int(
        np.round(N_colors * (np.mean(final_accuracy) - min_acc_cbar) / accuracy_range)
    )
    plt.figure(r"Scatter plot: $\alpha$ vs accuracy")
    plt.scatter(epoch_alpha[100], final_accuracy, color=cscale(color_idx))

    plt.figure(r"Scatter plot: $\lambda$ vs accuracy")
    if len(final_accuracy) == 1:
        plt.scatter(epoch_lamda[0], final_accuracy[0], color=cscale(color_idx))
    else:
        plt.errorbar(
            x=np.mean(epoch_lamda[0]),
            y=np.mean(final_accuracy),
            xerr=np.std(epoch_lamda[0]),
            yerr=np.std(final_accuracy),
            color=cscale(color_idx),
            marker="o",
        )

    plt.figure(r"Evolution of $\alpha$ across SSL pretraining")
    if len(list(epoch_alpha.values())[0]) == 1:
        plt.plot(
            list(epoch_alpha.keys()),
            [np.mean(a) for a in list(epoch_alpha.values())],
            c=cscale(color_idx),
            marker="o",
            lw=1,
            alpha=np.mean(final_accuracy) / 100,
            # alpha=1-np.mean(final_accuracy)/100
        )
    else:
        plt.errorbar(
            x=list(epoch_alpha.keys()),
            y=[np.mean(a) for a in list(epoch_alpha.values())],
            yerr=[np.std(a) for a in list(epoch_alpha.values())],
            c=cscale(color_idx),
            marker="o",
            lw=1,
            alpha=np.mean(final_accuracy) / 100,
            # alpha=1-np.mean(final_accuracy)/100
        )
    if len(epoch_lamda.keys()) > 1:
        plt.figure(r"Evolution of $\lambda$ across SSL pretraining")
        if len(list(epoch_lamda.values())[0]) == 1:
            plt.plot(
                list(epoch_lamda.keys()),
                [np.mean(l) for l in list(epoch_lamda.values())],
                c=cscale(color_idx),
                marker="o",
                lw=1,
                alpha=np.mean(final_accuracy) / 100,
                # alpha=1-np.mean(final_accuracy)/100
            )
        else:
            plt.errorbar(
                x=list(epoch_lamda.keys()),
                y=[np.mean(l) for l in list(epoch_lamda.values())],
                yerr=[np.std(l) for l in list(epoch_lamda.values())],
                c=cscale(color_idx),
                marker="o",
                lw=1,
                alpha=np.mean(final_accuracy) / 100,
                # alpha=1-np.mean(final_accuracy)/100
            )
        plt.yscale("log")

    # plt.figure(r'Change in $\alpha$ across SSL pretraining')
    # epoch_alpha_list = list(epoch_alpha.keys())
    # epoch_alpha_vals = list(epoch_alpha.values())
    # del_alpha_vals = np.diff(epoch_alpha_vals)
    # plt.plot(epoch_alpha_list[1:],
    # 		np.abs(del_alpha_vals),
    # 		c=cscale(color_idx),
    # 		marker='o',lw=1,
    # 		alpha=np.mean(final_accuracy)/100,
    # 		# alpha=1-final_accuracy/100
    # 		)
plt.figure(r"Evolution of $\alpha$ across SSL pretraining")
if hparam_filter_val > 0:
    plt.title(
        "{} = {}".format(
            "pdim" if ssl_alg == "barlowtwins" else "temp", hparam_filter_val
        )
    )
plt.grid("on")
if verbose:
    print(
        "min accuracy: {:3f}, max accuracy: {:.3f}".format(min_accuracy, max_accuracy)
    )
norm = mpl.colors.Normalize(vmin=0, vmax=2)
sm = cm.ScalarMappable(cmap=cscale, norm=norm)
sm.set_array([])
cbar = plt.colorbar(
    sm, ticks=np.linspace(0, 2, 10), boundaries=np.arange(-0.05, 2.1, 0.1)
)
cbar.ax.set_yticklabels(
    ["{:.2f}".format(i) for i in np.linspace(min_acc_cbar, max_accuracy, 10)]
)
plt.ylabel(r"$\alpha$")
plt.ylim([0.7, 3.5])
plt.xlabel("SSL pretraining Epochs")

# plt.figure(r'Change in $\alpha$ across SSL pretraining')
# if hparam_filter_val > 0:
# 	plt.title('{} = {}'.format(
# 		'pdim' if ssl_alg=='barlowtwins' else 'temp',
# 		hparam_filter_val))
# plt.grid('on')
# norm = mpl.colors.Normalize(vmin=0,vmax=2)
# sm = cm.ScalarMappable(cmap=cscale, norm=norm)
# sm.set_array([])
# cbar = plt.colorbar(sm,
# 					ticks=np.linspace(0,2,10),
# 					boundaries=np.arange(-0.05,2.1,.1))
# cbar.ax.set_yticklabels(
# 	['{:.2f}'.format(i) for i in np.linspace(min_acc_cbar,max_accuracy,10)]
# 	)
# plt.ylabel(r'$\Delta\alpha$')
# plt.xlabel('SSL pretraining Epochs')
# plt.yscale('log')

plt.show()
