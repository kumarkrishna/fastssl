#!/usr/bin/env bash
#SBATCH --array=0-19%10
#SBATCH --partition=long
#SBATCH --gres=gpu:2g.20gb:1
##SBATCH --reservation=DGXA100
#SBATCH --mem=16GB
#SBATCH --time=5:00:00
#SBATCH --cpus-per-gpu=4
#SBATCH --output=sbatch_out/exp_adaptive_barlowtwins.%A.%a.out
#SBATCH --error=sbatch_err/exp_adaptive_barlowtwins.%A.%a.err
#SBATCH --job-name=exp_adaptive_barlowtwins

. /etc/profile
module load anaconda/3
module load cuda/11.6
conda activate ffcv_eg

lamda_arr=(0.00001 0.00002 0.00005 0.0001 0.0002 0.0005 0.001 0.002 0.005 0.01)
adaptive=('True' 'False')
# lambd=0.00397897
pdim=2048
dataset='cifar10'
seeds=3

if [ $dataset = 'stl10' ]
then
    batch_size=256
else
    batch_size=512
fi

len1=${#adaptive[@]}
lidx=$((SLURM_ARRAY_TASK_ID/len1))
aidx=$((SLURM_ARRAY_TASK_ID%len1))
lambd=${lamda_arr[$lidx]}

# checkpt_dir=$SCRATCH/fastssl/checkpoints_adaptiveLamda
if [ ${adaptive[$aidx]} = 'True' ]
then
    # run BarlowTwins with adaptive lambda config
    checkpt_dir=$SCRATCH/fastssl/checkpoints_adaptiveLamda
    adaptive_str='--training.adaptive_ssl=True'
else
    # run standard BarlowTwins config
    checkpt_dir=$SCRATCH/fastssl/checkpoints_noadaptiveLamda
    adaptive_str=''
fi


for (( seed=1; seed<=$seeds; seed++ ))
do
    echo "Running seed" $seed
    # Let's train a SSL (BarlowTwins) model with the above hyperparams
    python scripts/train_model.py --config-file configs/cc_barlow_twins.yaml \
                                --training.lambd=$lambd \
                                --training.projector_dim=$pdim \
                                --training.dataset=$dataset \
                                --training.ckpt_dir=$checkpt_dir \
                                --training.batch_size=$batch_size \
                                --training.seed=$seed \
                                --training.track_alpha=True \
                                --training.log_interval=5 \
                                $adaptive_str
                                # --training.adaptive_ssl=True

    # Let's precache features, should take ~35 seconds (rtx8000)
    python scripts/train_model.py --config-file configs/cc_precache.yaml \
                                --training.lambd=$lambd \
                                --training.projector_dim=$pdim \
                                --training.dataset=$dataset \
                                --training.ckpt_dir=$checkpt_dir \
                                --training.batch_size=$batch_size \
                                --training.seed=$seed \
                                --training.model=resnet50feat

    # run linear eval on precached features from model: using default seed 42
    python scripts/train_model.py --config-file configs/cc_classifier.yaml \
                                --training.lambd=$lambd \
                                --training.projector_dim=$pdim \
                                --training.dataset=$dataset \
                                --training.ckpt_dir=$checkpt_dir \
                                --training.seed=$seed
done
# dataset='stl10'

# cp $SLURM_TMPDIR/*.pth $checkpt_dir/resnet50_checkpoints/
