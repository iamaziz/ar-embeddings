Code, embeddings, and datasets used in the paper:

A. Altowayan and L. Tao _"Word Embeddings for Arabic Sentiment Analysis"_, IEEE BigData 2016 Workshop


##### How to run:

Make sure to unzip `embeddings/arabic-news.tar.gz`, then run

`$ python asa.py --vectors embeddings/arabic-news.bin --dataset datasets/LABR-book-reviews.csv`

```
[2017-04-08 12:59:20,387] INFO: loading projection weights from embeddings/arabic-news.bin
[2017-04-08 12:59:23,408] INFO: loaded (159175, 300) matrix from embeddings/arabic-news.bin
[2017-04-08 12:59:23,408] INFO: precomputing L2-norms of word weight vectors
[2017-04-08 12:59:24,525] INFO: dataset datasets/LABR-book-reviews.csv (16448, 2). Split: 14803 training and 1645 testing.
[2017-04-08 12:59:24,526] INFO: Tokenizing the training dataset ..
[2017-04-08 12:59:24,950] INFO:  ... total 927007 training tokens.
[2017-04-08 12:59:24,950] INFO: Tokenizing the testing dataset ..
[2017-04-08 12:59:25,003] INFO:  ... total 110705 testing tokens.
[2017-04-08 12:59:25,003] INFO: Vectorizing training tokens ..
[2017-04-08 12:59:27,414] INFO:  ... total 14803 training
[2017-04-08 12:59:27,415] INFO: Vectorizing testing tokens ..
[2017-04-08 12:59:27,723] INFO:  ... total 1645 testing
[2017-04-08 12:59:27,848] INFO: Done loading and vectorizing data.
[2017-04-08 12:59:27,848] INFO: --- Sentiment CLASSIFIERS ---
[2017-04-08 12:59:27,848] INFO: fitting ...
[2017-04-08 13:02:03,397] INFO: results ...
	MacAvg. 80.41% F1. 79.95% P. 81.37 R. 78.58 : LinearSVC
	MacAvg. 77.31% F1. 76.79% P. 78.10 R. 75.52 : RandomForestClassifier
	MacAvg. 63.93% F1. 57.42% P. 72.88 R. 47.37 : GaussianNB
	MacAvg. 80.84% F1. 80.50% P. 81.45 R. 79.56 : NuSVC
	MacAvg. 81.15% F1. 80.77% P. 81.89 R. 79.68 : LogisticRegressionCV
	MacAvg. 78.97% F1. 79.00% P. 78.34 R. 79.68 : SGDClassifier
[2017-04-08 13:02:03,397] INFO: DONE!
```

##### Dependencies:

Check out `requirements.txt` file.
To install the dependencies:

> `$ pip install -r requirements.txt`
