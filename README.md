## FastSSL

Train *better* models *faster* with optimized SSL pipelines.

To install and using this library use the following command: 

```pip install -r requirements.txt -e .```


Training SSL   
```python scripts/train_model.py --config-file configs/barlow_twins.yaml```

Training Classifier  
```python scripts/train_model.py --config-file configs/classifier.yaml```

Train on cluster 
```python scripts/train_model.py --config-file config/cc_barlow_twins.yaml```    