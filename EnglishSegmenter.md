# English Segmenter

The object is an instance of Forward Model (defined in ForwardModel.py) trained on the English dictionnary provided in _data/words_alpha.txt_.

## Parameters of training

* __Method of training__: Baum-Welch algorithm
* __Number of iterations__: 807
* __Final log-likelihood__: -10325891

## Retrieve object

Import `pickle` and run the following command:

```python
with open("EnglishSegmenter.txt","rb") as f:
   fm = pickle.load(f)
```

