import os, sys, subprocess

all_args = sys.argv[1:]

config_files_dict = {
    'BarlowTwins': 'cc_BarlowTwins.yaml',
    'byol': 'cc_byol.yaml',
    'SimCLR': 'cc_SimCLR.yaml',
    'spectralReg': 'cc_spectralReg.yaml',
}
seeds = [1, 2, 3]

ssl_train_cmd = 'python python_scripts/train_model.py'
for arg in all_args:
    ssl_train_cmd += ' {}'.format(arg)
    if 'algorithm' in arg:
        assert arg.split('=')[1] in config_files_dict.keys(), "SSL Algorithm {} not implemented".format(arg.split('=')[1]) 
        ssl_train_cmd += ' --config-file configs/cc_{}.yaml'.format(arg.split('=')[1])
print(ssl_train_cmd)
os.system(ssl_train_cmd)

for seed in seeds:
    linear_eval_cmd = 'python python_scripts/train_model.py --config-file configs/cc_classifier.yaml --training.seed={}'.format(seed)
    for arg in all_args:
        if 'log_interval' in arg:
            linear_eval_cmd += ' --eval.epoch={}'.format(arg.split('=')[1])
        if 'algorithm' in arg:
            linear_eval_cmd += ' --eval.train_algorithm={}'.format(arg.split('=')[1])
            arg = ' --training.algorithm=linear'
        linear_eval_cmd += ' {}'.format(arg)
    print(linear_eval_cmd)
    os.system(linear_eval_cmd)