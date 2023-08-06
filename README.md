## FastSSL

Train *better* models *faster* with optimized SSL pipelines.

### Installation

```
git clone git@github.com:facebookresearch/FFCV-SSL.git
cd FFCV-SSL
conda create -y -n ffcv-ssl python=3.9 cupy pkg-config compilers libjpeg-turbo opencv pytorch torchvision torchaudio pytorch-cuda=11.7 numba -c pytorch -c nvidia -c conda-forge
conda activate ffcv-ssl
pip install -e . -r requirements.txt
```

## Training

### Train Model from Scratch

barlow twins
```
python scripts/trainer.py --config-file configs/barlow_twins.yaml
```

### Evaluate pretrained model from checkpoint

```
python scripts/trainer.py --config-file configs/eval_barlow_twins.yaml
```




TO BE DEPRECATED
================

### Training Models

Training SSL   
```python scripts/train_model.py --config-file configs/barlow_twins.yaml```

Training Classifier  
```python scripts/train_model.py --config-file configs/classifier.yaml```

Train on cluster 
```python scripts/train_model.py --config-file config/cc_barlow_twins.yaml```    
