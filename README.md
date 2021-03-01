# Naive Bayes Classifier for Text Classification

Naive Bayes classifiers have been successfully applied to classifying text documents. We will implement the Naive Bayes algorithm to tackle the \20 Newsgroups" classification problem.

## Data Set
The 20 Newsgroups data set is a collection of approximately 20,000 newsgroup documents, partitioned (nearly) evenly across 20 different newsgroups. It was originally collected by Ken Lang, probably for his Newsweeder: Learning to filter netnews [1] paper, though he did not explicitly mention this collection. The 20 newsgroups collection has become a popular data set for
experiments in text applications of machine learning techniques, such as text classification and text clustering.

The data is organized into 20 different newsgroups, each corresponding to a different topic. Here is a list of the 20 newsgroups:

* alt.atheism
* comp.graphics
* comp.os.ms-windows.misc
* comp.sys.ibm.pc.hardware
* comp.sys.mac.hardware
* comp.windows.x
* misc.forsale
* rec.autos rec.motorcycles
* rec.sport.baseball rec.sport.hockey
* sci.crypt
* sci.electronics
* sci.med
* sci.med
* soc.religion.christian
* talk.politics.guns
* talk.politics.mideast
* talk.politics.misc
* talk.religion.misc



The original data set is available at [the link](http://qwone.com/~jason/20Newsgroups/ "20Newgroups"). In this project, we won’t need to process the original data set. Instead, a processed version of the data set is provided (see __20newsgroups.zip__). This processed version represents 18824 documents which have been divided into two subsets: training (11269 documents) and testing (7505 documents).
After unzipping the file, you will find six files: **map.csv**, **train_label.csv**, **train_data.csv**, **test_label.csv**, **test_data.csv**, **vocabulary.txt**. The **vocabulary.txt** contains all distinct words and other tokens in the 18824 documents. **train data.csv** and **test_data.csv** are formatted **"docIdx, wordIdx, count"**, where **docIdx** is the document id, **wordIdx** represents the word
id (in correspondence to **vocabulary.txt**) and count is the frequency of the word in the document. **train_label.csv** and **test_label.csv** are simply a list of label id’s indicating which newsgroup each document belongs to (with the row number representing the document id). The **map.csv** maps from label id’s to label names.


## What we do

In general, we will implement a Pythom program that takes the input files, builds a Naive Bayes classifier, and outputs relevant statistics. We will learn our Naive Bayes classifier from the training data (**train_label.csv**, **train_data.csv**),
then evaluate its performance on the testing data (**test_label.csv**, **test_data.csv**). Specifically, our program will accomplish the following two tasks:

### Learn the Naive Bayes Model

We will implement the *multinomial model* (*"bag of words"* model). In the learning phase, you will estimate the required probability terms using the training data.


### Evaluate the Performance of the Classifier

We will evaluate our Naive Bayes classifiers on both the training and the testing data. We will use our Naive Bayes classifiers to make classification decision on these data set and calculate relevant statistics such as overall accuracy, class accuracy, confusion matrix, etc. When making classification decision, consider only words
found in Vocabulary.

<!-- ___ -->

# Specification

Language used: 
* Python 3.8

Additional packages used: 

* numpy 1.18.1

* pandas 1.0.1

* scikit-learn 0.22.1

* scipy 1.4.1	(comes with scikit-learn 0.22.1)
