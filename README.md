## FastSSL

To install and using this library use the following command: 

```pip install -r requirements.txt -e .```


Training SSL   
```python scripts/train_model.py --training.log_interval 10 --training.epochs 50```

Training Classifier  
```python scripts/train_model.py --training.algorithm linear --training.model linear --training.epochs 100```
