
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, confusion_matrix

# Training label
train_label = open('20newsgroups/train_label.csv')

# pi is the fraction of each class
pi = {}

# Set a class index for each document as key
for i in range(1, 21):
    pi[i] = 0

# Extract values from training labels
lines = train_label.readlines()

# Get total number of documents
total = len(lines)

# Count the occurrence of each class
for line in lines:
    val = int(line.split()[0])
    pi[val] += 1

# Divide the count of each class by total documents
for key in pi:
    pi[key] /= total

# Print the probability (prior) of each class:
print("Class prior of each class:")
for c in range(1, 21):
    print("P( Omega =", c, "):", pi[c])

# Training data
train_data = open('20newsgroups/train_data.csv')
df = pd.read_csv(train_data, delimiter=',', names=['docIdx', 'wordIdx', 'count'])

# Training label
label = []
train_label = open('20newsgroups/train_label.csv')
lines = train_label.readlines()
for line in lines:
    label.append(int(line.split()[0]))

# Increase label length to match docIdx
docIdx = df['docIdx'].values
i = 0
new_label = []
for index in range(len(docIdx)-1):

    new_label.append(label[i])
    if docIdx[index] != docIdx[index+1]:
        i += 1

new_label.append(label[i]) # for-loop ignores last value

# Add label column
df['classIdx'] = new_label

# Calculate the size of Vocabulary
V = len(open('20newsgroups/vocabulary.txt').readlines())

# Alpha value for smoothing in Laplace estimate for Bayesian estimator
a = 1

# Calculate probability of each word based on class
pb_ij = df.groupby(['classIdx', 'wordIdx'])
pb_j = df.groupby(['classIdx'])
PMLE = (pb_ij['count'].sum()) / (pb_j['count'].sum())
PBE = (pb_ij['count'].sum() + a) / (pb_j['count'].sum() + V)

# Unstack series
PMLE = PMLE.unstack()
PBE = PBE.unstack()

# Expand the data-frames to cover all words in Vocabulary
for col in range(len(PMLE.columns) + 1, V + 1):
    PMLE[col] = np.nan
    PBE[col] = np.nan

# Replace NaN as word count with 0 for PMLE or a/(count+|V|) for PBE
for c in range(1, 21):
    PMLE.loc[c, :] = PMLE.loc[c, :].fillna(0)
    PBE.loc[c, :] = PBE.loc[c, :].fillna(a/(pb_j['count'].sum()[c] + V))

# Convert to dictionary for greater speed
# PMLE/PBE_dict = {wordIdx: {classIdx:  PMLE/PBE,  classIdx: PMLE/PBE, ..20.., classIdx: PMLE/PBE}, ..53975..,
# wordIdx: {classIdx:  PMLE/PBE,  classIdx: PMLE/PBE, ...}}
PMLE_dict = PMLE.to_dict()
PBE_dict = PBE.to_dict()

def MNB(df, smooth=False):
    '''
    Multinomial Naive Bayes classifier
    :param df [Pandas Dataframe]: Dataframe of data
    :param smooth [bool]: Use Bayesian estimator if True, use Maximum Likelihood estimator if False
    :return predict [list]: Predicted class ID
    '''
    # Using dictionaries for greater speed
    # df_dict = {'docIdx': {0: 1, 1: 1, 2: 1, .., 1467344: 1}, 'wordIdx' : {0: 1, 1: 2, 2: 3, .., 1467344: 53958},
    # 'count': {0: 4, 1: 2, 2: 10, .., 1467344: 1}, 'classIdx': {0: 1, 1: 1, 2: 1, .., 1467344: 20}}
    df_dict = df.to_dict()
    new_dict = {}
    prediction = []

    # new_dict = {docIdx: {wordIdx: count, wordIdx: count, ...}, docIdx: {wordIdx: count, wordIdx: count, ...},
    # ..11269.., docIdx: {wordIdx: count, wordIdx: count, ...}}
    for idx in range(len(df_dict['docIdx'])):
        docIdx = df_dict['docIdx'][idx]
        wordIdx = df_dict['wordIdx'][idx]
        count = df_dict['count'][idx]
        try:
            new_dict[docIdx][wordIdx] = count
        except:
            new_dict[df_dict['docIdx'][idx]] = {}
            new_dict[docIdx][wordIdx] = count

    # Calculating the scores for each doc
    for docIdx in range(1, len(new_dict) + 1):
        score_dict = {}
        # Creating a probability row for each class
        for classIdx in range(1, 21):

            # initiate the value
            score_dict[classIdx] = 0

            # For each word:
            for wordIdx in new_dict[docIdx]:

                if smooth:
                    probability = PBE_dict[wordIdx][classIdx]
                    Nk = new_dict[docIdx][wordIdx]
                    score_dict[classIdx] += Nk * np.log(probability)

                else:
                    try:
                        probability = PMLE_dict[wordIdx][classIdx]
                        Nk = new_dict[docIdx][wordIdx]
                        score_dict[classIdx] += Nk * np.log(probability)
                    except:
                        # Missing word will have 0*log(n/V) = 0
                        score_dict[classIdx] += 0

            # Add final with prior
            score_dict[classIdx] += np.log(pi[classIdx])

        # Get class with max probability for the given docIdx
        max_score = max(score_dict, key=score_dict.get)
        prediction.append(max_score)

    return prediction

MLE_predict = MNB(df, smooth=False)
BE_predict = MNB(df, smooth=True)

# Get list of labels
train_label = pd.read_csv('20newsgroups/train_label.csv', names=['t'])
train_label = train_label['t'].tolist()
total = len(train_label)
models = [MLE_predict, BE_predict]
strings = ['Maximum Likelihood estimator', 'Bayesian estimator']

for m, s in zip(models, strings):
    correct = 0
    for i, j in zip(m, train_label):
        if i == j:
            correct += 1
    print("Overall Accuracy of", s, "on train data:", correct/total)

    print("Class Accuracy of", s, "on train data:")

    for c in range(1, 21):

        print("Group", c, ":", recall_score(train_label, m, average=None)[c-1])

    print("Confusion matrix of", s, "on train data:\n", confusion_matrix(train_label, m))


# Get test data
test_data = open('20newsgroups/test_data.csv')
df = pd.read_csv(test_data, delimiter=',', names=['docIdx', 'wordIdx', 'count'])

# Get list of labels
test_label = pd.read_csv('20newsgroups/test_label.csv', names=['t'])
test_label = test_label['t'].tolist()

# MNB Calculation
MLE_predict = MNB(df, smooth=False)
BE_predict = MNB(df, smooth=True)

total = len(test_label)
models = [MLE_predict, BE_predict]
strings = ['Maximum Likelihood estimator', 'Bayesian estimator']

for m, s in zip(models, strings):
    correct = 0
    for i, j in zip(m, test_label):
        if i == j:
            correct += 1
    print("Overall Accuracy of", s, "on test data:", correct/total)

    print("Class Accuracy of", s, "on test data:")

    for c in range(1, 21):
        print("Group", c, ":", recall_score(test_label, m, average=None)[c - 1])

    print("Confusion matrix of", s, "on test data:\n", confusion_matrix(test_label, m))
=======
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, confusion_matrix

# Training label
train_label = open('20newsgroups/train_label.csv')

# pi is the fraction of each class
pi = {}

# Set a class index for each document as key
for i in range(1, 21):
    pi[i] = 0

# Extract values from training labels
lines = train_label.readlines()

# Get total number of documents
total = len(lines)

# Count the occurrence of each class
for line in lines:
    val = int(line.split()[0])
    pi[val] += 1

# Divide the count of each class by total documents
for key in pi:
    pi[key] /= total

# Print the probability (prior) of each class:
print("Class prior of each class:")
for c in range(1, 21):
    print("P( Omega =", c, "):", pi[c])

# Training data
train_data = open('20newsgroups/train_data.csv')
df = pd.read_csv(train_data, delimiter=',', names=['docIdx', 'wordIdx', 'count'])

# Training label
label = []
train_label = open('20newsgroups/train_label.csv')
lines = train_label.readlines()
for line in lines:
    label.append(int(line.split()[0]))

# Increase label length to match docIdx
docIdx = df['docIdx'].values
i = 0
new_label = []
for index in range(len(docIdx)-1):

    new_label.append(label[i])
    if docIdx[index] != docIdx[index+1]:
        i += 1

new_label.append(label[i]) # for-loop ignores last value

# Add label column
df['classIdx'] = new_label

# Calculate the size of Vocabulary
V = len(open('20newsgroups/vocabulary.txt').readlines())

# Alpha value for smoothing in Laplace estimate for Bayesian estimator
a = 1

# Calculate probability of each word based on class
pb_ij = df.groupby(['classIdx', 'wordIdx'])
pb_j = df.groupby(['classIdx'])
PMLE = (pb_ij['count'].sum()) / (pb_j['count'].sum())
PBE = (pb_ij['count'].sum() + a) / (pb_j['count'].sum() + V)

# Unstack series
PMLE = PMLE.unstack()
PBE = PBE.unstack()

# Expand the data-frames to cover all words in Vocabulary
for col in range(len(PMLE.columns) + 1, V + 1):
    PMLE[col] = np.nan
    PBE[col] = np.nan

# Replace NaN as word count with 0 for PMLE or a/(count+|V|) for PBE
for c in range(1, 21):
    PMLE.loc[c, :] = PMLE.loc[c, :].fillna(0)
    PBE.loc[c, :] = PBE.loc[c, :].fillna(a/(pb_j['count'].sum()[c] + V))

# Convert to dictionary for greater speed
# PMLE/PBE_dict = {wordIdx: {classIdx:  PMLE/PBE,  classIdx: PMLE/PBE, ..20.., classIdx: PMLE/PBE}, ..53975..,
# wordIdx: {classIdx:  PMLE/PBE,  classIdx: PMLE/PBE, ...}}
PMLE_dict = PMLE.to_dict()
PBE_dict = PBE.to_dict()

def MNB(df, smooth=False):
    '''
    Multinomial Naive Bayes classifier
    :param df [Pandas Dataframe]: Dataframe of data
    :param smooth [bool]: Use Bayesian estimator if True, use Maximum Likelihood estimator if False
    :return predict [list]: Predicted class ID
    '''
    # Using dictionaries for greater speed
    # df_dict = {'docIdx': {0: 1, 1: 1, 2: 1, .., 1467344: 1}, 'wordIdx' : {0: 1, 1: 2, 2: 3, .., 1467344: 53958},
    # 'count': {0: 4, 1: 2, 2: 10, .., 1467344: 1}, 'classIdx': {0: 1, 1: 1, 2: 1, .., 1467344: 20}}
    df_dict = df.to_dict()
    new_dict = {}
    prediction = []

    # new_dict = {docIdx: {wordIdx: count, wordIdx: count, ...}, docIdx: {wordIdx: count, wordIdx: count, ...},
    # ..11269.., docIdx: {wordIdx: count, wordIdx: count, ...}}
    for idx in range(len(df_dict['docIdx'])):
        docIdx = df_dict['docIdx'][idx]
        wordIdx = df_dict['wordIdx'][idx]
        count = df_dict['count'][idx]
        try:
            new_dict[docIdx][wordIdx] = count
        except:
            new_dict[df_dict['docIdx'][idx]] = {}
            new_dict[docIdx][wordIdx] = count

    # Calculating the scores for each doc
    for docIdx in range(1, len(new_dict) + 1):
        score_dict = {}
        # Creating a probability row for each class
        for classIdx in range(1, 21):

            # initiate the value
            score_dict[classIdx] = 0

            # For each word:
            for wordIdx in new_dict[docIdx]:

                if smooth:
                    probability = PBE_dict[wordIdx][classIdx]
                    Nk = new_dict[docIdx][wordIdx]
                    score_dict[classIdx] += Nk * np.log(probability)

                else:
                    try:
                        probability = PMLE_dict[wordIdx][classIdx]
                        Nk = new_dict[docIdx][wordIdx]
                        score_dict[classIdx] += Nk * np.log(probability)
                    except:
                        # Missing word will have 0*log(n/V) = 0
                        score_dict[classIdx] += 0

            # Add final with prior
            score_dict[classIdx] += np.log(pi[classIdx])

        # Get class with max probability for the given docIdx
        max_score = max(score_dict, key=score_dict.get)
        prediction.append(max_score)

    return prediction

MLE_predict = MNB(df, smooth=False)
BE_predict = MNB(df, smooth=True)

# Get list of labels
train_label = pd.read_csv('20newsgroups/train_label.csv', names=['t'])
train_label = train_label['t'].tolist()
total = len(train_label)
models = [MLE_predict, BE_predict]
strings = ['Maximum Likelihood estimator', 'Bayesian estimator']

for m, s in zip(models, strings):
    correct = 0
    for i, j in zip(m, train_label):
        if i == j:
            correct += 1
    print("Overall Accuracy of", s, "on train data:", correct/total)

    print("Class Accuracy of", s, "on train data:")

    for c in range(1, 21):

        print("Group", c, ":", recall_score(train_label, m, average=None)[c-1])

    print("Confusion matrix of", s, "on train data:\n", confusion_matrix(train_label, m))


# Get test data
test_data = open('20newsgroups/test_data.csv')
df = pd.read_csv(test_data, delimiter=',', names=['docIdx', 'wordIdx', 'count'])

# Get list of labels
test_label = pd.read_csv('20newsgroups/test_label.csv', names=['t'])
test_label = test_label['t'].tolist()

# MNB Calculation
MLE_predict = MNB(df, smooth=False)
BE_predict = MNB(df, smooth=True)

total = len(test_label)
models = [MLE_predict, BE_predict]
strings = ['Maximum Likelihood estimator', 'Bayesian estimator']

for m, s in zip(models, strings):
    correct = 0
    for i, j in zip(m, test_label):
        if i == j:
            correct += 1
    print("Overall Accuracy of", s, "on test data:", correct/total)

    print("Class Accuracy of", s, "on test data:")

    for c in range(1, 21):
        print("Group", c, ":", recall_score(test_label, m, average=None)[c - 1])

    print("Confusion matrix of", s, "on test data:\n", confusion_matrix(test_label, m))
