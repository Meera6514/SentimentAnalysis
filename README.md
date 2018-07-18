# Twitter Sentiment Analysis

## Dataset Information

I compare some classification methods to do sentiment analysis of tweets (a binary classification problem). The data set used is from Kaggle. The training dataset is a csv file of type `tweet_id,sentiment,tweet` where the `tweet_id` is a unique integer identifying the tweet, `sentiment` is either `1` (positive) or `0` (negative), and `tweet` is the tweet enclosed in `""`. Similarly, the test dataset is a csv file of type `tweet_id,tweet`.
The initial data set is pre-processed using stats.py and saved as train-processed.csv for training data set and test-processed.csv for testing data set.


## Requirements
All code is created and compiled using Anaconda distribution Python 3.6.4 on Macbook
There are some general library requirements for the project and some which are specific to individual methods. The general requirements are as follows.  
* `numpy`
* `scikit-learn`
* `scipy`
* `nltk`

The library requirements specific to some methods are:
* `keras` with `TensorFlow` backend for CNN.


## Usage
## NOTE : prior to running these files, run utils.py, which basically has helper  functions for the models.
## I am using 10% of my training data to see the accuracy of my models

### Naive Bayes
1. Run `naivebayes.py`. With `TRAIN = True` it will show the accuracy results on 10% validation dataset.

### Maximum Entropy
2. Run `maxent-nltk.py <>` to run MaxEnt model of NLTK. With `TRAIN = True` it will show the accuracy results on 10% validation dataset.

### SVM
3. Run `svm.py`. With `TRAIN = True` it will show the accuracy results on 10% validation dataset.

### Convolutional Neural Networks
4. Run `cnn.py`. This will run the 4-Conv-NN (4 conv layers neural network) model as described in the report. To run other versions of CNN, just comment or remove the lines where Conv layers are added. Will validate using 10% data and save models for each epoch in `./models/`.


## Information about other files

* 'glove-seeds.txt`: GloVe words vectors from StanfordNLP of dimension 200 for seeding word embeddings.
* `Plots.ipynb`: IPython notebook used to generate plots in report and some other plots for analysis.
