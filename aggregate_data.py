# -*- coding: utf-8 -*-

import json
import numpy as np
import os
import sys
import logging

plots_path = "./plots"
name = 'ssl'

global missing_runs
missing_runs = []

try:
    root_dir = os.path.join(
        os.environ['SAVE_DIR'],
        name
    )
except KeyError:
    print("Please run setup_env first!")
    sys.exit(1)


def parse_stats(fname, stats, metric, epoch):
    try:
        metric_vals = stats[str(metric)]
    except KeyError:
        logger.info(f"Metric {metric} not found in {fname}")
        missing_runs.append(fname)
        return 0

    if metric not in figure1_conf["performance_metrics"] + ["train_loss"]:
        keys = [ i[0] for i in metric_vals ]
        vals = [ i[1] for i in metric_vals ]
        try:
            key = keys.index(epoch)
            val = vals[key]
        except ValueError:
            logger.info(f"Epoch {epoch} missing for metric {metric} in {fname}")
            missing_runs.append(fname)
            val = 0
    else:
        val = metric_vals[-1]
    return val


""" Figure 1

    SSL pretraining: alpha, feature jacobian, effective rank
    
    algorithms: barlow_twins, byol, simclr, vicreg
    models: resnet18
    datasets: cifar10, stl10
    epoch: 100 (barlow_twins), 300 (byol), 100 (simclr), 100 (vicreg)
    
    train_configs: ssl, linear
    widths
    seeds
    noise configs
    metrics
"""

figure1_conf = {
    "algorithms": ["barlow_twins", "byol", "simclr", "vicreg"],
    "base_models": ["resnet18"],
    "widths": {
        "barlow_twins": {
            "cifar10": list(range(1,65)),
            "stl10": list(range(1,65)),
        },
        "byol" : {
            "cifar10": list(range(1,65)),
            "stl10": list(range(1,65)),
        },
        "simclr": {
            "cifar10": list(range(1,65)),
            "stl10": list(range(1,65)),
        },
        "vicreg" : {
            "cifar10": list(range(1,65)),
            "stl10": list(range(1,65)),
        },
    },
    "seeds" : [0, 1, 2],
    "epochs": {
        "barlow_twins": 100,
        "byol": 300,
        "simclr": 100,
        "vicreg": 100,
        "linear": 200,
    },
    "noise_configs": [0, 5, 10, 15, 20, 40, 60, 80, 100],
    "datasets": ["cifar10", "stl10"],
    "performance_metrics": ["train_acc_1", "train_acc_1_clean", "train_acc_1_corrupted", "train_acc_1_restored", "test_acc_1"],
    "metrics": ["train_loss", "alpha", "feature_input_jacobian", "effective_rank"],
}

figure1_conf.update({
    "filenames": {
        "barlow_twins": {
            dataset: {
                width: [ f"{root_dir}_barlow_twins-{dataset}/resnet18/width{width}/2_augs/lambd_0.005000_pdim_{32 * width}_lr_0.001_wd_1e-05/results_{dataset}_alpha_ssl_100_seed_{seed}.npy"  for seed in figure1_conf["seeds"] ] for width in figure1_conf["widths"]["barlow_twins"][dataset]
            } for dataset in figure1_conf["datasets"]
        },
        "byol": {
            dataset: {
                width: [ f"{root_dir}_byol-{dataset}/resnet18/width{width}/2_augs/lambd_0.007812_pdim_{32 * width}_no_autocast_lr_0.001_wd_1e-06/results_{dataset}_alpha_ssl_300_seed_{seed}.npy"  for seed in figure1_conf["seeds"] ] for width in figure1_conf["widths"]["byol"][dataset]
            } for dataset in figure1_conf["datasets"]
        },
        "simclr": {
            "cifar10": {
                width: [ f"{root_dir}_simclr-cifar10/resnet18/width{width}/2_augs/temp_0.100_pdim_{32 * width}_no_autocast_bsz_512_lr_0.001_wd_1e-05/results_cifar10_alpha_SimCLR_100_seed_{seed}.npy"  for seed in figure1_conf["seeds"] ] for width in figure1_conf["widths"]["simclr"]["cifar10"]
            },
            "stl10": { 
                width: [ f"{root_dir}_simclr-stl10/resnet18/width{width}/2_augs/temp_0.100_pdim_{32 * width}_no_autocast_bsz_256_lr_0.001_wd_1e-05/results_stl10_alpha_SimCLR_100_seed_{seed}.npy"  for seed in figure1_conf["seeds"] ] for width in figure1_conf["widths"]["simclr"]["stl10"]
            },
        },
        "vicreg": {
            "cifar10": {
                width: [ f"{root_dir}_vicreg-cifar10/resnet18/width{width}/2_augs/lambd_25.000_mu_25.000_pdim_{32 * width}_bsz_512_lr_0.001_wd_1e-05/results_cifar10_alpha_VICReg_100_seed_{seed}.npy"  for seed in figure1_conf["seeds"] ] for width in figure1_conf["widths"]["vicreg"]["cifar10"]
            },
            "stl10": {
                width: [ f"{root_dir}_vicreg-stl10/resnet18/width{width}/2_augs/lambd_25.000_mu_25.000_pdim_{32 * width}_bsz_256_lr_0.001_wd_1e-05/results_stl10_alpha_VICReg_100_seed_{seed}.npy"  for seed in figure1_conf["seeds"] ] for width in figure1_conf["widths"]["vicreg"]["stl10"]
            }
        },
    },
})

figure1_conf.update({
    "performance_filenames": {
        "barlow_twins": {
            dataset: {
                0: {
                    width: [ f"{root_dir}_barlow_twins-{dataset}/resnet18/width{width}/2_augs/lambd_0.005000_pdim_{32 * width}_lr_0.001_wd_1e-06/1_augs_eval/results_{dataset}_alpha_linear_200_seed_{seed}.npy"  for seed in figure1_conf["seeds"] ] for width in figure1_conf["widths"]["barlow_twins"][dataset]
                },
                **{noise: {
                    width: [ f"{root_dir}_barlow_twins_noise{noise}-{dataset}/resnet18/width{width}/2_augs/lambd_0.005000_pdim_{32 * width}_lr_0.001_wd_1e-06/1_augs_eval/results_{dataset}_alpha_linear_200_seed_{seed}.npy"  for seed in figure1_conf["seeds"] ] for width in figure1_conf["widths"]["barlow_twins"][dataset]
                } for noise in figure1_conf["noise_configs"][1:]}
            } for dataset in figure1_conf["datasets"]
        },
        "byol": {
            dataset: {
                0 : {
                    width: [ f"{root_dir}_byol-{dataset}/resnet18/width{width}/2_augs/lambd_0.007812_pdim_{width * 32}_lr_0.001_wd_1e-06/1_augs_eval/results_{dataset}_alpha_linear_200_seed_{seed}.npy"  for seed in figure1_conf["seeds"] ] for width in figure1_conf["widths"]["byol"][dataset]
                },
                **{noise: {
                    width: [ f"{root_dir}_byol_noise{noise}-{dataset}/resnet18/width{width}/2_augs/lambd_0.007812_pdim_{width * 32}_lr_0.001_wd_1e-06/1_augs_eval/results_{dataset}_alpha_linear_200_seed_{seed}.npy"  for seed in figure1_conf["seeds"] ] for width in figure1_conf["widths"]["byol"][dataset]
                } for noise in figure1_conf["noise_configs"][1:]}
            } for dataset in figure1_conf["datasets"]
        },
        "simclr": {
            dataset: {
                0 : {
                    width: [ f"{root_dir}_simclr-{dataset}/resnet18/width{width}/2_augs/lambd_0.007812_pdim_{32 * width}_lr_0.001_wd_1e-06/1_augs_eval/results_{dataset}_alpha_linear_200_seed_{seed}.npy"  for seed in figure1_conf["seeds"] ] for width in figure1_conf["widths"]["simclr"][dataset]
                },
            } for dataset in figure1_conf["datasets"]
        },
        "vicreg": {
            dataset: {
                0 : {
                    width: [ f"{root_dir}_vicreg-{dataset}/resnet18/width{width}/2_augs/lambd_25.000000_pdim_{32 * width}_lr_0.001_wd_1e-06/1_augs_eval/results_{dataset}_alpha_linear_200_seed_{seed}.npy"  for seed in figure1_conf["seeds"] ] for width in figure1_conf["widths"]["vicreg"][dataset]
                },
            } for dataset in figure1_conf["datasets"]
        },
    },
})



missing_runs = []
def aggregate_data(destdir=plots_path):
    """ Aggregate results for figure 1
    """
    nseeds = len(figure1_conf["seeds"])
    nmetrics = len(figure1_conf["metrics"])
    nnoise = len(figure1_conf["noise_configs"])
    figure1_data = figure1_conf

    plot_data = {}
    for algorithm in figure1_conf["algorithms"]:
        plot_data[algorithm] = {}
        for dataset in figure1_conf["datasets"]:
            plot_data[algorithm][dataset] = {}
            nwidths = len(figure1_conf["widths"][algorithm][dataset])
            for base_model in figure1_conf["base_models"]:
                plot_data[algorithm][dataset][base_model] = np.zeros((nwidths, nmetrics, nseeds))
                for w_id, width in enumerate(figure1_conf["widths"][algorithm][dataset]):
                    for s_id in range(nseeds):
                        fname = figure1_conf["filenames"][algorithm][dataset][width][s_id]
                        logger.info(f"Loading {fname}")
                        try:
                            stats = np.load(
                                fname,
                                allow_pickle=True
                            ).tolist()
                            for m_id, metric in enumerate(figure1_conf["metrics"]):
                                epoch = figure1_conf["epochs"][algorithm]
                                logger.info(f"Parsing {base_model}_{width} epoch {epoch} {metric} seed {s_id}")
                                plot_data[algorithm][dataset][base_model][w_id, m_id, s_id] = parse_stats(fname, stats, metric, epoch)
                                
                        except FileNotFoundError:
                            logger.info(f"File not found: {fname}")
                            missing_runs.append(fname)
                            
                plot_data[algorithm][dataset][base_model] = \
                    plot_data[algorithm][dataset][base_model].tolist()
        
    figure1_data.pop("filenames")
    figure1_data["plot_data"] = plot_data
    
    nmetrics = len(figure1_conf["performance_metrics"])
    performance_data = {}
    for algorithm in figure1_conf["algorithms"]:
        performance_data[algorithm] = {}
        epoch = figure1_conf["epochs"]["linear"]
        for dataset in figure1_conf["datasets"]:
            performance_data[algorithm][dataset] = {}
            nwidths = len(figure1_conf["widths"][algorithm][dataset])
            for base_model in figure1_conf["base_models"]:
                performance_data[algorithm][dataset][base_model] = np.zeros((nnoise, nwidths, nmetrics, nseeds))
                for n_id, noise in enumerate(figure1_conf["noise_configs"]):
                    if noise > 0 and algorithm in ["simclr", "vicreg"]: continue
                    for w_id, width in enumerate(figure1_conf["widths"][algorithm][dataset]):
                        for s_id in range(nseeds):
                            fname = figure1_conf["performance_filenames"][algorithm][dataset][noise][width][s_id]
                            logger.info(f"Loading {fname}")
                            try:
                                stats = np.load(
                                    fname,
                                    allow_pickle=True
                                ).tolist()
                                for m_id, metric in enumerate(figure1_conf["performance_metrics"]):
                                    if noise == 0 and metric in ["train_acc_1_clean", "train_acc_1_corrupted", "train_acc_1_restored"]: continue
                                    logger.info(f"Parsing {base_model}_{width} epoch {epoch} {metric} seed {s_id}")
                                    performance_data[algorithm][dataset][base_model][n_id, w_id, m_id, s_id] = parse_stats(fname, stats, metric, epoch)
                                
                            except FileNotFoundError:
                                logger.info(f"File not found: {fname}")
                                missing_runs.append(fname)
                            
                performance_data[algorithm][dataset][base_model] = \
                    performance_data[algorithm][dataset][base_model].tolist()
    
    figure1_data.pop("performance_filenames")
    figure1_data["performance_data"] = performance_data

    if len(missing_runs) > 0:
        logger.info("The following files were missing:")
        for f in missing_runs:
            logger.info(f)

    filename = os.path.join(destdir, 'plot_data.json')
    logger.info(f"Saving plot data to {filename}")
    with open(filename, 'w') as fp:
        json.dump(figure1_data, fp, allow_nan=False)



""" Figure 2

    SSL pretraining: alpha, feature jacobian, rankme, 
    
    algorithms: 
    models: resnet18
    datasets: cifar10, stl10
    epoch: 
    
    train_configs: ssl, linear
    widths
    seeds
    noise configs
    metrics
"""

figure2_conf = {
    "algorithms": ["barlow_twins", "byol", "vicreg"],
    "base_models": ["resnet18"],
    "seeds" : [0, 1, 2, 3, 4],
    "epochs": [50, 100, 200],
    "noise_configs": [40,],
    "datasets": ["cifar10", "stl10"],
    "performance_metrics": ["train_acc_1", "train_acc_1_clean", "train_acc_1_corrupted", "train_acc_1_restored", "test_acc_1"],
    "metrics": ["train_loss", "alpha", "feature_input_jacobian", "effective_rank", "rankme"],
    "hparams": {
        "barlow_twins": ["0.000500", "0.001000", "0.005000", "0.010000"],
        "byol": ["0.007812"],
        "vicreg": ["5.000", "15.000", "25.000", "35.000"]
    },
}


figure2_conf.update({
    "filenames": {
        "barlow_twins": {
            dataset: {
                hparam: [ f"{root_dir}_sweep_barlow_twins-{dataset}/resnet18/width64/2_augs/lambd_{hparam}_pdim_2048_lr_0.001_wd_1e-05/results_{dataset}_alpha_ssl_200_seed_{seed}.npy"  for seed in figure2_conf["seeds"] ] for hparam in figure2_conf["hparams"]["barlow_twins"]
            } for dataset in figure2_conf["datasets"]
        },
        "byol": {
            dataset: {
                hparam: [ f"{root_dir}_sweep_byol-{dataset}/resnet18/width64/2_augs/lambd_{hparam}_pdim_2048_lr_0.001_wd_1e-05/results_{dataset}_alpha_ssl_200_seed_{seed}.npy"  for seed in figure2_conf["seeds"] ] for hparam in figure2_conf["hparams"]["byol"]
            } for dataset in figure2_conf["datasets"]
        },
        "vicreg": {
            "cifar10": {
                hparam: [ f"{root_dir}_sweep_VICReg-cifar10/resnet18/width64/2_augs/lambd_{hparam}_mu_25.000_pdim_2048_bsz_512_lr_0.001_wd_1e-05/results_cifar10_alpha_VICReg_200_seed_{seed}.npy"  for seed in figure2_conf["seeds"] ] for hparam in figure2_conf["hparams"]["vicreg"]
            },
            "stl10": {
                hparam: [ f"{root_dir}_sweep_VICReg-stl10/resnet18/width64/2_augs/lambd_{hparam}_mu_25.000_pdim_2048_bsz_256_lr_0.001_wd_1e-05/results_stl10_alpha_VICReg_200_seed_{seed}.npy"  for seed in figure2_conf["seeds"] ] for hparam in figure2_conf["hparams"]["vicreg"]
            },
        },
    },
})

figure2_conf.update({
    "performance_filenames": {
        "barlow_twins": {
            dataset: {
                40: {
                    epoch: {
                        hparam: [ f"{root_dir}_sweep_barlow_twins_noise40-{dataset}/resnet18/width64/2_augs/lambd_{hparam}_pdim_2048_lr_0.001_wd_1e-06/1_augs_eval_epoch_{epoch}/results_{dataset}_alpha_linear_200_seed_{seed}.npy"  for seed in figure2_conf["seeds"] ] for hparam in figure2_conf["hparams"]["barlow_twins"]
                    } for epoch in figure2_conf["epochs"]
                }
            } for dataset in figure2_conf["datasets"]
        },
        "byol": {
            dataset: {
                40 : {
                    epoch: {
                        hparam: [ f"{root_dir}_sweep_byol_noise40-{dataset}/resnet18/width64/2_augs/lambd_{hparam}_pdim_2048_lr_0.001_wd_1e-06/1_augs_eval_epoch_{epoch}/results_{dataset}_alpha_linear_200_seed_{seed}.npy" for seed in figure2_conf["seeds"] ] for hparam in figure2_conf["hparams"]["byol"]
                    } for epoch in figure2_conf["epochs"]
                },
            } for dataset in figure2_conf["datasets"]
        },       
        "vicreg": {
            dataset: {
                40 : {
                    epoch: {
                        hparam: [ f"{root_dir}_sweep_VICReg_noise40-{dataset}/resnet18/width64/2_augs/lambd_{hparam}000_pdim_2048_lr_0.001_wd_1e-06/1_augs_eval_epoch_{epoch}/results_{dataset}_alpha_linear_200_seed_{seed}.npy"  for seed in figure2_conf["seeds"] ] for hparam in figure2_conf["hparams"]["vicreg"]
                    } for epoch in figure2_conf["epochs"]
                },
            } for dataset in figure2_conf["datasets"]
        },
    },
})

missing_runs = []
def aggregate_data(destdir=plots_path):
    """ Aggregate results for figure 2
    """
    nseeds = len(figure2_conf["seeds"])
    nmetrics = len(figure2_conf["metrics"])
    nnoise = len(figure2_conf["noise_configs"])
    nepochs = len(figure2_conf["epochs"])
    noise = figure2_conf["noise_configs"][0]
    
    figure2_data = figure2_conf

    plot_data = {}
    for dataset in figure2_conf["datasets"]:
        plot_data[dataset] = {}
        for algorithm in figure2_conf["algorithms"]:
            nconfigs = len(figure2_conf["hparams"][algorithm])
            plot_data[dataset][algorithm] = np.zeros((nmetrics, nconfigs, nepochs, nseeds))
            for h_id, hparam in enumerate(figure2_conf["hparams"][algorithm]):
                for s_id in range(nseeds):
                    fname = figure2_conf["filenames"][algorithm][dataset][hparam][s_id]
                    logger.info(f"Loading {fname}")
                    try:
                        stats = np.load(
                            fname,
                            allow_pickle=True
                        ).tolist()
                        for e_id, epoch in enumerate(figure2_conf["epochs"]):
                            for m_id, metric in enumerate(figure2_conf["metrics"]):
                                logger.info(f"Parsing hparam {hparam} epoch {epoch} {metric} seed {s_id}")
                                plot_data[dataset][algorithm][m_id, h_id, e_id, s_id] = parse_stats(fname, stats, metric, epoch)
                                
                    except FileNotFoundError:
                        logger.info(f"File not found: {fname}")
                        missing_runs.append(fname)
                        
            plot_data[dataset][algorithm] = \
                plot_data[dataset][algorithm].tolist()
        
    figure2_data.pop("filenames")
    figure2_data["plot_data"] = plot_data
    
    nmetrics = len(figure2_conf["performance_metrics"])
    performance_data = {}
    for dataset in figure2_conf["datasets"]:
        performance_data[dataset] = {}
        for algorithm in figure2_conf["algorithms"]:
            nconfigs = len(figure2_conf["hparams"][algorithm])
            performance_data[dataset][algorithm] = np.zeros((nmetrics, nconfigs, nepochs, nseeds))
            for e_id, epoch in enumerate(figure2_conf["epochs"]):
                for h_id, hparam in enumerate(figure2_conf["hparams"][algorithm]):
                    for s_id in range(nseeds):
                        fname = figure2_conf["performance_filenames"][algorithm][dataset][noise][epoch][hparam][s_id]
                        logger.info(f"Loading {fname}")
                        try:
                            stats = np.load(
                                fname,
                                allow_pickle=True
                            ).tolist()
                            for m_id, metric in enumerate(figure2_conf["performance_metrics"]):
                                if noise == 0 and metric in ["train_acc_1_clean", "train_acc_1_corrupted", "train_acc_1_restored"]: continue
                                logger.info(f"Parsing hparam {hparam} epoch {epoch} {metric} seed {s_id}")
                                performance_data[dataset][algorithm][m_id, h_id, e_id, s_id] = parse_stats(fname, stats, metric, epoch)
                            
                        except FileNotFoundError:
                            logger.info(f"File not found: {fname}")
                            missing_runs.append(fname)
                        
            performance_data[dataset][algorithm] = \
                    performance_data[dataset][algorithm].tolist()
    
    figure2_data.pop("performance_filenames")
    figure2_data["performance_data"] = performance_data

    if len(missing_runs) > 0:
        logger.info("The following files were missing:")
        for f in missing_runs:
            logger.info(f)

    filename = os.path.join(destdir, 'plot_data.json')
    logger.info(f"Saving plot data to {filename}")
    with open(filename, 'w') as fp:
        json.dump(figure2_data, fp, allow_nan=False)


""" Figure 3

    SSL pretraining: alpha, feature jacobian, rankme,
    for Barlow Twins on Tiny CIFAR-10
    
    algorithms: Barlow Twins
    models: resnet18
    datasets: cifar10 subsampled
    epoch: 
    
    train_configs: ssl, linear
    widths
    seeds
    noise configs
    metrics
"""

figure3_conf = {
    "algorithms": ["barlow_twins"],
    "base_models": ["resnet18"],
    "widths": list(range(1,65)),
    "seeds" : [0, 1, 2],
    "epochs": [10000,],
    "noise_configs": [0,],
    "datasets": ["cifar10",],
    "metrics": ["train_loss", "train_loss_on_diag", "train_loss_off_diag", "alpha", "feature_input_jacobian", "effective_rank"],
    "performance_metrics": ["test_loss", "test_loss_on_diag", "test_loss_off_diag",],
}


figure3_conf.update({
    "filenames": {
        "barlow_twins": {
            dataset: {
                width: [ f"{root_dir}_barlow_twins_subsample2-{dataset}/resnet18/width{width}/2_augs/lambd_0.005000_pdim_{32 * width}_lr_0.0001_wd_1e-06/results_{dataset}_alpha_ssl_10000_seed_{seed}.npy"  for seed in figure3_conf["seeds"] ] for width in figure3_conf["widths"]
            } for dataset in figure3_conf["datasets"]
        },
    },
})

figure3_conf.update({
    "performance_filenames": {
        "barlow_twins": {
            dataset: {
                0: {
                    width: [ f"{root_dir}_barlow_twins_subsample2-{dataset}/resnet18/width{width}/2_augs/lambd_0.005000_pdim_{32 * width}_lr_0.0001_wd_1e-06/results_{dataset}_ssl_eval_ssl_10000_seed_{seed}.npy"  for seed in figure3_conf["seeds"] ] for width in figure3_conf["widths"]
                },
            } for dataset in figure3_conf["datasets"]
        },
    },
})


missing_runs = []
def aggregate_data(destdir=plots_path):
    """ Aggregate results for figure 3
    """
    nseeds = len(figure3_conf["seeds"])
    nmetrics = len(figure3_conf["metrics"])
    nnoise = len(figure3_conf["noise_configs"])
    nwidths = len(figure3_conf["widths"])
    figure3_data = figure3_conf

    plot_data = {}
    for algorithm in figure3_conf["algorithms"]:
        plot_data[algorithm] = {}
        for dataset in figure3_conf["datasets"]:
            plot_data[algorithm][dataset] = {}
            for base_model in figure3_conf["base_models"]:
                plot_data[algorithm][dataset][base_model] = np.zeros((nwidths, nmetrics, nseeds))
                for w_id, width in enumerate(figure3_conf["widths"]):
                    for s_id in range(nseeds):
                        fname = figure3_conf["filenames"][algorithm][dataset][width][s_id]
                        logger.info(f"Loading {fname}")
                        try:
                            stats = np.load(
                                fname,
                                allow_pickle=True
                            ).tolist()
                            for m_id, metric in enumerate(figure3_conf["metrics"]):
                                epoch = figure3_conf["epochs"][algorithm]
                                logger.info(f"Parsing {base_model}_{width} epoch {epoch} {metric} seed {s_id}")
                                plot_data[algorithm][dataset][base_model][w_id, m_id, s_id] = parse_stats(fname, stats, metric, epoch)
                                
                        except FileNotFoundError:
                            logger.info(f"File not found: {fname}")
                            missing_runs.append(fname)
                            
                plot_data[algorithm][dataset][base_model] = \
                    plot_data[algorithm][dataset][base_model].tolist()
        
    figure3_data.pop("filenames")
    figure3_data["plot_data"] = plot_data
    
    nmetrics = len(figure3_conf["performance_metrics"])
    performance_data = {}
    for algorithm in figure3_conf["algorithms"]:
        performance_data[algorithm] = {}
        epoch = figure3_conf["epochs"]
        for dataset in figure3_conf["datasets"]:
            performance_data[algorithm][dataset] = {}
            for base_model in figure3_conf["base_models"]:
                performance_data[algorithm][dataset][base_model] = np.zeros((nnoise, nwidths, nmetrics, nseeds))
                for n_id, noise in enumerate(figure3_conf["noise_configs"]):
                    for w_id, width in enumerate(figure3_conf["widths"]):
                        for s_id in range(nseeds):
                            fname = figure3_conf["performance_filenames"][algorithm][dataset][noise][width][s_id]
                            logger.info(f"Loading {fname}")
                            try:
                                stats = np.load(
                                    fname,
                                    allow_pickle=True
                                ).tolist()
                                for m_id, metric in enumerate(figure3_conf["performance_metrics"]):
                                    if noise == 0 and metric in ["train_acc_1_clean", "train_acc_1_corrupted", "train_acc_1_restored"]: continue
                                    logger.info(f"Parsing {base_model}_{width} epoch {epoch} {metric} seed {s_id}")
                                    performance_data[algorithm][dataset][base_model][n_id, w_id, m_id, s_id] = parse_stats(fname, stats, metric, epoch)
                                
                            except FileNotFoundError:
                                logger.info(f"File not found: {fname}")
                                missing_runs.append(fname)
                            
                performance_data[algorithm][dataset][base_model] = \
                    performance_data[algorithm][dataset][base_model].tolist()
    
    figure3_data.pop("performance_filenames")
    figure3_data["performance_data"] = performance_data

    if len(missing_runs) > 0:
        logger.info("The following files were missing:")
        for f in missing_runs:
            logger.info(f)

    filename = os.path.join(destdir, 'plot_data.json')
    logger.info(f"Saving plot data to {filename}")
    with open(filename, 'w') as fp:
        json.dump(figure3_data, fp, allow_nan=False)



"""Main
"""

def aggregate_stats(fig_id):
    logger.info(f"Aggregating data for figure {fig_id}")
    
    if fig_id == 1:
        destdir = os.path.join(plots_path, 'figure1')
        if not os.path.exists(destdir):
            os.makedirs(destdir)
        aggregate_data(destdir)
    elif fig_id == 2:
        destdir = os.path.join(plots_path, 'figure2')
        if not os.path.exists(destdir):
            os.makedirs(destdir)
        aggregate_data(destdir)
    elif fig_id == 3:
        destdir = os.path.join(plots_path, 'figure3')
        if not os.path.exists(destdir):
            os.makedirs(destdir)
        aggregate_data(destdir)
    else:
        raise ValueError(f"Invalid figure id {fig_id}")


def init_logging(logger_name, logfile, log_level: str):
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: %s" % log_level)

    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")

    f_handler = logging.FileHandler(logfile)
    f_handler.setLevel(numeric_level)
    c_handler = logging.StreamHandler()
    c_handler.setLevel(numeric_level)
    f_handler.setFormatter(formatter)
    c_handler.setFormatter(formatter)
    
    logging.basicConfig(level=numeric_level, handlers=[f_handler, c_handler])
    logger = logging.getLogger(logger_name)
    return logger


def main():

    if not os.path.exists(plots_path):
        os.makedirs(plots_path)

    logfile = os.path.join(plots_path, 'aggregate_data.log')
    init_logging(None, logfile, 'info')
    
    global logger
    logger = logging.getLogger()

    for fig_id in [2]:
        aggregate_stats(fig_id)


if __name__ == '__main__':
    main()
