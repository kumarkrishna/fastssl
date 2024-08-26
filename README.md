## FastSSL

Train *better* models *faster* with optimized SSL pipelines.

### Installation

To install and use this library, follow the steps below.

1. Step 1: Clone the [FFCV-SSL library](https://github.com/facebookresearch/FFCV-SSL)
2. Step 2: Install the conda env and core dependencies using the following commands:  
   ```
    echo "Creating new conda environment ======"
    conda create --prefix <env_location>/<env_name> -y python=3.9 conda conda-libmamba-solver -c conda-forge
    conda activate <env_location>/<env_name>
    
    echo "Setting up conda env variables and config options =========="
    export CONDA_EXE="$(hash -r; which conda)"
    conda config --set solver libmamba
    conda config --set channel_priority flexible
    
    echo "Installing torch and other important packages+dependencies ============"
    conda install -y cupy pkg-config compilers libjpeg-turbo opencv=4.7.0 pytorch=2.1.2 torchvision=0.16.2 pytorch-cuda=12.1 torchmetrics numba=0.56.2 terminaltables matplotlib scikit-learn pandas assertpy pytz -c pytorch -c nvidia -c conda-forge
    
    cd <FFCV-SSL folder>
    pip install -e .
   ```

3. Step 3: Install the auxilliary requirments and the fastssl library:  
```pip install -r requirements.txt -e .```

#### Notes
1. Installing ffcv has led to issues due to conflicts in torch/opencv versions. Check [Issue #45](https://github.com/kumarkrishna/fastssl/issues/45) for any updates to installation steps.


### Training Models

Training SSL   
```python scripts/train_model.py --config-file configs/barlow_twins.yaml```

Training Classifier  
```python scripts/train_model.py --config-file configs/classifier.yaml```

Train on cluster 
```python scripts/train_model.py --config-file config/cc_barlow_twins.yaml```    
