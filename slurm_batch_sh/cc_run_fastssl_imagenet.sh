#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
##SBATCH --account=def-siamakx-ab
##SBATCH --account=def-siddiqi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=32G
#SBATCH --time=26:00:00
#SBATCH --array=15-23
#SBATCH --output=output/experiment_imagenet-%A.%a.out

echo "My SLURM_ARRAY_JOB_ID is $SLURM_ARRAY_JOB_ID."
echo "My SLURM_ARRAY_TASK_ID is $SLURM_ARRAY_TASK_ID"

ffcv_data_dir=~/scratch/ffcv_dataset/imagenet/ffcv
compute_node_data_dir=$SLURM_TMPDIR/ImageNet

echo 'Copying Imagenet'
ml gcc python/3.9 cuda/11.4 opencv/4.5.5
source ~/ffcvenv/bin/activate
mkdir $compute_node_data_dir
cp -r $ffcv_data_dir $compute_node_data_dir
echo 'Imagenet Copied ..'

lamda_arr=(5e-3)
proj_arr=(4096)
checkpt_dir='checkpoints_design_hparams_stl10'

lenL=${#lamda_arr[@]}
lidx=$((SLURM_ARRAY_TASK_ID%lenL))
pidx=$((SLURM_ARRAY_TASK_ID/lenL))


mkdir ~/scratch/bt_checkpoints/ckpt_${lamda_arr[$lidx]}_${proj_arr[$pidx]}

dataset=imagenetImage
batch_size=128

python scripts/train_model.py \
            --config-file configs/cc_barlow_twins.yaml \
            --training.lambd=${lamda_arr[$lidx]} \
            --training.projector_dim=${proj_arr[$pidx]} \
            --training.dataset=$dataset \
            --training.ckpt_dir=$checkpt_dir \
            --training.batch_size=$batch_size \


cp -r $SLURM_TMPDIR/* ~/scratch/bt_checkpoints/ckpt_${lamda_arr[$lidx]}_${proj_arr[$pidx]}