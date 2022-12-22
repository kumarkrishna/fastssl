import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import os
import glob
from linclab_utils import plot_utils
from tqdm import tqdm
from sklearn.metrics import r2_score

plot_utils.linclab_plt_defaults(font="Arial", fontdir=os.path.expanduser(
    '~')+"/Projects/fonts") 	# Run locally, not from cluster

plot_abs = False
flag_debug = False
calc_new_alpha = True
plot_colorplots = False
R2_thresh = 0.95
dataset_ssl = 'cifar10'
dataset_classifier = 'cifar10'
ckpt_dir = 'simclr_checkpoints_arch_hparams_{}'.format(dataset_ssl)
arch_arr = ['shallowConv_2', 'shallowConv_4',
            'shallowConv_6', 'shallowConv_8', 'resnet50']
# arch_arr = ['shallowConv_2','shallowConv_4']
colors_arr = ['violet', 'darkorange', 'darkgreen', 'gold', 'brown']
hyperparam_order = {'temp': 0, 'pdim': 1, 'bsz': 2, 'lr': 3, 'wd': 4}
default_hyperparam_values = {'temp': 0.01,
                             'pdim': 512, 'bsz': 512, 'lr': 0.001, 'wd': 1e-6}
hyperparam_sweep = ['temp', 'bsz']


def stringer_get_powerlaw(ss, trange):
    # COPIED FROM Stringer+Pachitariu 2018b github repo! (https://github.com/MouseLand/stringer-pachitariu-et-al-2018b/blob/master/python/utils.py)
    ''' fit exponent to variance curve'''
    logss = np.log(np.abs(ss))
    y = logss[trange][:, np.newaxis]
    trange += 1
    nt = trange.size
    x = np.concatenate(
        (-np.log(trange)[:, np.newaxis], np.ones((nt, 1))), axis=1)
    w = 1.0 / trange.astype(np.float32)[:, np.newaxis]
    b = np.linalg.solve(x.T @ (x * w), (w * x).T @ y).flatten()

    allrange = np.arange(0, ss.size).astype(int) + 1
    x = np.concatenate(
        (-np.log(allrange)[:, np.newaxis], np.ones((ss.size, 1))), axis=1)
    ypred = np.exp((x * b).sum(axis=1))
    alpha = b[0]
    # subtracting 10 here arbitrarily because we want to avoid the last tail!
    max_range = 500 if len(ss) >= 512 else len(ss) - 10
    fit_R2 = r2_score(y_true=logss[trange[0]:max_range], y_pred=np.log(
        np.abs(ypred))[trange[0]:max_range])
    try:
        fit_R2_range = r2_score(
            y_true=logss[trange], y_pred=np.log(np.abs(ypred))[trange])
    except:
        fit_R2_range = None
    return alpha, ypred, fit_R2, fit_R2_range


def plot_alpha_fit(data_dict, trange, hyperparam_configs):
    eigen = data_dict['eigenspectrum']
    al, ypred, r2, r2_range = stringer_get_powerlaw(eigen, trange)
    print(r2, r2_range)
    plt.loglog(np.arange(1, 1+200), eigen[:200])
    plt.loglog(np.arange(1, 1+200), ypred[:200])
    plt.title("Hyperparams = {:.2e}, alpha={:.3f}".format(
        hyperparam_configs, al))
    plt.show()


def plot_colorplot(attribute_dict, attribute_label, vmin=None, vmax=None, cmap='seismic'):
    keys = list(attribute_dict.keys())
    assert hyperparam_sweep[0] in hyperparam_order.keys(
    ), "{} not a valid hyperparam".format(hyperparam_sweep[0])
    hparam1_arr = np.array(
        sorted(list(set([k[hyperparam_order[hyperparam_sweep[0]]] for k in keys]))))
    assert hyperparam_sweep[1] in hyperparam_order.keys(
    ), "{} not a valid hyperparam".format(hyperparam_sweep[1])
    hparam2_arr = np.array(
        sorted(list(set([k[hyperparam_order[hyperparam_sweep[1]]] for k in keys]))))
    plotting_arr = np.empty((len(hparam1_arr), len(hparam2_arr)))
    plotting_arr[:] = np.nan
    for idx1, hparam1_val in enumerate(hparam1_arr):
        for idx2, hparam2_val in enumerate(hparam2_arr):
            dict_key = ()
            for hparam, val in default_hyperparam_values.items():
                if hparam == hyperparam_sweep[0]:
                    dict_key += (hparam1_val,)
                elif hparam == hyperparam_sweep[1]:
                    dict_key += (hparam2_val,)
                else:
                    dict_key += (val,)
            try:
                if type(attribute_dict[dict_key]) is list:
                    plotting_arr[idx1, idx2] = np.mean(
                        np.array(attribute_dict[dict_key]))
                else:
                    plotting_arr[idx1, idx2] = attribute_dict[dict_key]
            except:
                pass
    plt.figure()
    plt.imshow(plotting_arr, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
    plt.colorbar()
    plt.title(attribute_label)
    plt.yticks(np.arange(len(hparam1_arr)), hparam1_arr)
    plt.xticks(np.arange(0, len(hparam2_arr), 2),
               hparam2_arr[np.arange(0, len(hparam2_arr), 2)])
    plt.gca().set_xticklabels(FormatStrFormatter('%.2e').format_ticks(
        hparam2_arr[np.arange(0, len(hparam2_arr), 2)]))
    plt.ylabel(hyperparam_sweep[0])
    plt.xlabel(hyperparam_sweep[1])


def plot_scatterplot(
    attribute1_dict,
    attribute1_label,
    attribute2_dict,
    attribute2_label,
    group_by_hyperparam='pdim',
    color=None,
    filter_dict=None,
    vmin=None, vmax=None,
    hmin=None, hmax=None
):
    assert group_by_hyperparam in hyperparam_order.keys(
    ), "{} is not a valid hyperparam".format(group_by_hyperparam)
    plt.figure('scatterplot')
    markers_arr = ['o', '^', 'x', 's', 'd', 'v', '*', 'P']
    attr1_arr = {}
    attr1_err_arr = {}
    attr2_arr = {}
    attr2_err_arr = {}
    for cidx, hyperparam_config in enumerate(attribute1_dict.keys()):
        gr_hparam = hyperparam_config[hyperparam_order[group_by_hyperparam]]
        if gr_hparam not in attr1_arr.keys():
            attr1_arr[gr_hparam] = []
            attr1_err_arr[gr_hparam] = []
            attr2_arr[gr_hparam] = []
            attr2_err_arr[gr_hparam] = []
        if filter_dict is not None and np.nanmean(np.array(filter_dict[hyperparam_config])) < R2_thresh:
            continue
        if type(attribute1_dict[hyperparam_config]) is list:
            attr1_arr[gr_hparam].append(
                np.mean(np.array(attribute1_dict[hyperparam_config])))
            attr1_err_arr[gr_hparam].append(
                np.std(np.array(attribute1_dict[hyperparam_config])))
        else:
            attr1_arr[gr_hparam].append(attribute1_dict[hyperparam_config])
            attr1_err_arr[gr_hparam].append(0)
        if type(attribute2_dict[hyperparam_config]) is list:
            attr2_arr[gr_hparam].append(
                np.mean(np.array(attribute2_dict[hyperparam_config])))
            attr2_err_arr[gr_hparam].append(
                np.std(np.array(attribute2_dict[hyperparam_config])))
        else:
            attr2_arr[gr_hparam].append(attribute2_dict[hyperparam_config])
            attr2_err_arr[gr_hparam].append(0)

    gr_hparam_arr = list(attr1_arr.keys())
    for gidx, g_hparam in enumerate(gr_hparam_arr):
        # plt.scatter(attr1_arr,attr2_arr,marker=markers_arr[pidx%len(markers_arr)],label='pdim={}'.format(int(pdim)))
        if color is None:
            color = 'blue'
        plt.errorbar(x=attr1_arr[g_hparam],
                     y=attr2_arr[g_hparam],
                     xerr=attr1_err_arr[g_hparam],
                     yerr=attr2_err_arr[g_hparam],
                     marker=markers_arr[gidx % len(markers_arr)],
                     color=color, ls='',
                     label='{}={}'.format(group_by_hyperparam, g_hparam))
    plt.title('SimCLR {} vs {}'.format(attribute1_label, attribute2_label))
    plt.xlabel('{}'.format(attribute1_label))
    plt.ylabel('{}'.format(attribute2_label))
    plt.legend()
    plt.ylim([vmin, vmax])
    plt.xlim([hmin, hmax])


def get_best_hparam(acc_dict):
    acc_keys = list(acc_dict.keys())
    acc_values = [np.array(v).mean() for v in list(acc_dict.values())]
    max_acc = max(acc_values)
    hparam_setting = acc_keys[acc_values.index(max_acc)]
    return hparam_setting, max_acc


plt.figure('scatterplot')
for aidx, arch in enumerate(arch_arr):
    ckpt_dir_arch = os.path.join(ckpt_dir, arch)
    files = glob.glob(os.path.join(ckpt_dir_arch, '*'))

    def extract_hparams_from_fname(x):
        temp = float(os.path.basename(x).split('temp_')[-1].split('_')[0])
        pdim = int(os.path.basename(x).split('pdim_')[-1].split('_')[0])
        bsz = int(os.path.basename(x).split('bsz_')[-1].split('_')[0])
        lr = float(os.path.basename(x).split('lr_')[-1].split('_')[0])
        wd = float(os.path.basename(x).split('wd_')[-1].split('_')[0])
        return (temp, pdim, bsz, lr, wd)
    sorting_order = [i[0] for i in sorted(enumerate(files),
                                          key=lambda x: extract_hparams_from_fname(x[1]))]
    files_sorted = [files[idx] for idx in sorting_order]
    if len(files_sorted) == 0:
        continue

    accuracy_dict = {}
    best_accuracy_dict = {}
    alpha_dict = {}
    SSL_loss_dict = {}
    R2_dict = {}
    R2_100_dict = {}
    alpha_correction_due_to_minibatches = 0.0  # 0.1
    for fidx, file in enumerate(tqdm(files_sorted)):
        try:
            hparam_settings = extract_hparams_from_fname(file)
            SSL_fname = os.path.join(
                file, 'results_{}_alpha_SimCLR_100.npy'.format(dataset_ssl))
            if os.path.exists(SSL_fname):
                SSL_file = np.load(SSL_fname, allow_pickle=True).item()
                # np.log(SSL_file['train_loss'][-1])/pdim_val
                SSL_loss_dict[hparam_settings] = SSL_file['train_loss'][-1]
            else:
                SSL_loss_dict[hparam_settings] = np.nan
            linear_files = glob.glob(os.path.join(
                file, 'results_{}_alpha_linear_200*'.format(dataset_classifier)))
            for linear_fname in linear_files:
                # linear_fname = os.path.join(file,'results_{}_early_alpha_linear_200.npy'.format(dataset))
                linear_dict = np.load(linear_fname, allow_pickle=True).item()
                if hparam_settings not in alpha_dict.keys():
                    alpha_dict[hparam_settings] = []
                    R2_dict[hparam_settings] = []
                    R2_100_dict[hparam_settings] = []
                    accuracy_dict[hparam_settings] = []
                    best_accuracy_dict[hparam_settings] = []

                if (calc_new_alpha and linear_dict['R2_100'] < R2_thresh) or dataset_classifier != dataset_ssl:
                    eigen = linear_dict['eigenspectrum']
                    if dataset_ssl != dataset_classifier:
                        range_init = 11
                    else:
                        range_init = 5
                    alpha, ypred, R2, r2_range = stringer_get_powerlaw(
                        eigen, np.arange(range_init, 50))
                    if r2_range < R2_thresh:
                        alpha, ypred, R2, r2_range = stringer_get_powerlaw(
                            eigen, np.arange(range_init, 20))
                    R2_100 = r2_range
                else:
                    alpha = linear_dict['alpha']
                    R2 = linear_dict['R2']
                    R2_100 = linear_dict['R2_100']
                if plot_abs:
                    alpha_dict[hparam_settings].append(np.abs(1-alpha))
                else:
                    alpha_dict[hparam_settings].append(alpha)
                R2_dict[hparam_settings].append(R2)
                R2_100_dict[hparam_settings].append(R2_100)
                test_acc_arr = np.array(linear_dict['test_acc_1'])
                # breakpoint()

                final_test_acc = test_acc_arr[-1]
                best_test_acc = test_acc_arr.max()
                accuracy_dict[hparam_settings].append(final_test_acc)
                best_accuracy_dict[hparam_settings].append(best_test_acc)
                if flag_debug:
                    plot_alpha_fit(linear_dict, trange=np.arange(
                        3, 100), hyperparam_configs=hparam_settings)
                    breakpoint()
        except:
            breakpoint()
            pass

    print(arch, get_best_hparam(accuracy_dict))
    if plot_colorplots:
        plot_colorplot(accuracy_dict, "Final accuracy", cmap='coolwarm', vmin=50,
                       vmax=85 if dataset_ssl != dataset_classifier in dataset_classifier else 90)
        plt.gca().set_title(plt.gca().get_title()+" ("+arch+")")
        plot_colorplot(alpha_dict, r"$|1-\alpha|$" if plot_abs else r"$\alpha$",
                       cmap='coolwarm_r', vmax=1.2 if plot_abs else 2.2)
        plt.gca().set_title(plt.gca().get_title()+" ("+arch+")")
        plot_colorplot(R2_100_dict, r"$R^2$ (top 100)",
                       cmap='coolwarm', vmin=0.90)
        plt.gca().set_title(plt.gca().get_title()+" ("+arch+")")
        plot_colorplot(SSL_loss_dict, "SSL loss/pdim",
                       cmap='coolwarm_r')  # ,vmax=6000)
        plt.gca().set_title(plt.gca().get_title()+" ("+arch+")")

    plot_scatterplot(alpha_dict, r"$|1-\alpha|$" if plot_abs else r"$\alpha$",
                     accuracy_dict, "Final accuracy",
                     group_by_hyperparam='bsz', color=colors_arr[aidx],
                     filter_dict=R2_100_dict,
                     vmin=50, vmax=85 if dataset_ssl != dataset_classifier in dataset_classifier else 90,
                     hmax=1.6 if 'stl' in dataset_classifier else 2)
if plot_abs:
    plt.axvline(x=0.0, color='k', ls='--')
    lim_min, lim_max = plt.xlim()
    if lim_min > -0.05:
        lim_min = -0.05
else:
    plt.axvline(x=1.0, color='k', ls='--')
    if dataset_ssl == dataset_classifier:
        plt.axvline(x=0.8, color='k', ls=':')
    lim_min, lim_max = plt.xlim()
    if lim_min > 0.95:
        lim_min = 0.95
plt.xlim([lim_min, lim_max])

# Delete duplicate labels
hand, labl = plt.gca().get_legend_handles_labels()
handout = []
lablout = []
for h, l in zip(hand, labl):
    if l not in lablout:
        # breakpoint()
        lablout.append(l)
        handout.append(h)
legend = plt.legend(handout, lablout, frameon=False)
plt.gca().set_title(plt.gca().get_title() +
                    r" : {} $\rightarrow$ {}".format(dataset_ssl, dataset_classifier))
plt.show()
