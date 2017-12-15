# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 16:02:58 2016

@author: CS328

Final project : Room Identification Training

This is the solution script for training a model for identifying
room from audio data. The script loads all labelled room
audio data files in the specified directory. It extracts features
from the raw data and trains and evaluates a classifier to identify
the room.

"""
from __future__ import division

import os
import sys
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from features import FeatureExtractor
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import pickle


# %%---------------------------------------------------------------------------
#
#		                 Load Data From Disk
#
# -----------------------------------------------------------------------------

data_dir = 'data'  # directory where the data files are stored

output_dir = 'training_output'  # directory where the classifier(s) are stored

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# the filenames should be in the form 'room-data-subject-1.csv', e.g. 'room-data-Erik-1.csv'. If they
# are not, that's OK but the progress output will look nonsensical

class_names = 'eng_lab_304 eng_lab_hallway_box eng_lab_307B eng_lab_323'.split()
# class_names = 'chris_bedroom downstairs_bathroom kitchen living_room staircase alex_bedroom upstairs_bathroom'.split()

data = np.zeros((0, 8002))  # 8002 = 1 (timestamp) + 8000 (for 8kHz audio data) + 1 (label)

for filename in os.listdir(data_dir):
    if filename.endswith(".csv") and filename.startswith("room-data"):
        filename_components = filename.split("-")  # split by the '-' character
        room = filename_components[2]
        print("Loading data for {}.".format(room))
        if room not in class_names:
            class_names.append(room)
        room_label = class_names.index(room)
        sys.stdout.flush()
        data_file = os.path.join(data_dir, filename)
        data_for_current_room = np.genfromtxt(data_file, delimiter=',')

        #print("Shuffling data")     #shuffles the data for individual rooms
        #data_for_current_room = shuffle(data_for_current_room)

        print("Loaded {} raw labeled audio data samples.".format(len(data_for_current_room)))
        sys.stdout.flush()
        data = np.append(data, data_for_current_room, axis=0)

print("Found data for {} rooms : {}".format(len(class_names), ", ".join(class_names)))

# %%---------------------------------------------------------------------------
#
#		                Extract Features & Labels
#
# -----------------------------------------------------------------------------

# change debug to True to show print statements we've included:
feature_extractor = FeatureExtractor(debug=False)

# You may need to change this depending on how you compute your features
# n_features = 20 + 0 + 75  # 20 formant features + 16 pitch contour features + 75 mfcc delta coefficients
n_features = feature_extractor.get_n_features() #11 + 17 + 975# + 923 # 11 formant features, 17 pitch contour features, 975 mfcc deltas, 923 delta deltas
# n_features = 11 + 0 + 0  # 11 formant features, 975 mfcc deltas

print("Extracting features and labels for {} audio windows...".format(data.shape[0]))
sys.stdout.flush()

X = np.zeros((0, n_features))
y = np.zeros(0,)

data_size = len(data)
data_scaling = arrange(1,data_size+1)
data_scaling = shuffle(data_scaling)

print("Shuffling data")
data = shuffle(data)

all_freqs = []
for _ in range(len(class_names)):
    all_freqs += [[]]

for i, window_with_timestamp_and_label in enumerate(data):
    window = window_with_timestamp_and_label[1:-1]
    label = data[i, -1]

    freqs = np.abs(np.fft.fft(window))
    freqs = freqs[1:int(freqs.shape[0]/2)]
    all_freqs[int(label)] += [freqs]
    # all_freqs[int(label)] += [freqs/np.max(freqs)] # normalize each window

    # print("Extracting features for window " + str(i) + "...")
    x = feature_extractor.extract_features(window)

    x = x*data_scaling[i]/data_size      #this line makes the data increase linearly from 0% to 100%

    if (len(x) != X.shape[1]):
        print("Received feature vector of length {}. Expected feature vector of length {}.".format(len(x), X.shape[1]))
    X = np.append(X, np.reshape(x, (1, -1)), axis=0)
    y = np.append(y, label)

show_graphs = True

show_mean_freqs = True
show_vars = True

var_array = []      #the array used to create the variance file

if show_graphs:
    print("Graphing...")
    import matplotlib.pyplot as plt

    for i, freqs in enumerate(all_freqs):
        all_freqs[i] = np.array(freqs)

    if show_mean_freqs:     #creates a graph of the frequencies
        for i, freqs in enumerate(all_freqs):
            if freqs.shape[0] > 0:
                plt.plot(np.mean(freqs, axis=0))
                plt.title("mean freqencies for " + class_names[i])
                plt.show()

    if show_vars:       #creates a graph of the variance
        for i, freqs in enumerate(all_freqs):
            if freqs.shape[0] > 0:
                var_temp = np.var(freqs, axis=0)
                plt.plot(var_temp)
                var_array = np.append(var_array, var_temp)      #adds to the variance array
                plt.title("variance for " + class_names[i])
                plt.show()


#creates variance file
with open('training_output/variance.pickle', 'wb') as f:  # 'wb' stands for 'write bytes'
    pickle.dump(var_array, f)

exit()

# print("oversampling the data")
# from imblearn.over_sampling import SMOTE
# sm = SMOTE(random_state=42)
# X, y = sm.fit_sample(X, y)

# print("undersampling the data")
# from imblearn.under_sampling import RandomUnderSampler
# rus = RandomUnderSampler(return_indices=True)
# X, y, idx_resampled = rus.fit_sample(X, y)


print("Finished feature extraction over {} windows".format(len(X)))
print("Unique labels found: {}".format(set(y)))
sys.stdout.flush()


# %%---------------------------------------------------------------------------
#
#		                Train & Evaluate Classifier
#
# -----------------------------------------------------------------------------

def evaluate_model(clf):
    n = len(y)
    num_classes = len(class_names)

    # TODO: Train and evaluate your decision tree classifier over 10-fold CV.
    # Report average accuracy, precision and recall metrics.

    cv = KFold(n_splits=10, shuffle=True, random_state=None)

    # used for storing average accuracy, precision, recall
    labels = class_names

    label_nums = list(range(len(labels)))
    metric_sums = [[0, 0, 0] for label in labels]  # sum of accuracy/precision/recall for each label
    metric_counts = [0] * len(labels)  # number of times label scores were updated

    # used for averaging the confusion matrices
    conf_array = []

    for i, (train_indexes, test_indexes) in enumerate(cv.split(X)):
        # print("Fold {}".format(i+1))

        # split into training and testing
        X_train = X[train_indexes, :]
        y_train = y[train_indexes]
        X_test = X[test_indexes, :]
        y_test = y[test_indexes]

        # print("y_train", y_train)
        # print("y_test", y_test)

        # fit the model
        clf.fit(X_train, y_train)

        # make predictions
        y_pred = clf.predict(X_test)
        # print("y_pred", y_pred)

        # conf = confusion_matrix(y_test, y_pred)
        conf = confusion_matrix(y_test, y_pred, labels=label_nums)
        conf_array += [conf]

        for label in label_nums:
            # compute true/false positives/negatives
            tp = conf[label][label]
            tn = sum([conf[i][i] for i in range(num_classes) if i != label])
            fp = sum([conf[i][label] for i in range(num_classes) if i != label])
            fn = sum([conf[label][i] for i in range(num_classes) if i != label])

            if tp > 0:
                # compute accuracy, precision, recall
                accuracy = (tp + tn) / (tp + fp + tn + fn)
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)

                # store the values so we can compute the averages later
                metric_sums[label][0] += accuracy
                metric_sums[label][1] += precision
                metric_sums[label][2] += recall
                metric_counts[label] += 1

    conf_array = np.array(conf_array)
    conf_mean = np.mean(conf_array, axis=0)
    # DEBUG
    print("Average confusion matrix:")
    print(conf_mean)
    # DEBUG

    # DEBUG
    # chris_col = conf_mean[:,1]
    # print("chris_col: {}".format(','.join([str(x) for x in chris_col])))
    # DEBUG

    # average accuracy, precision, recall over the k-folds
    averages = np.zeros(shape=(len(labels), 3))
    for label in label_nums:
        count = metric_counts[label]
        count = 1 if count == 0 else count
        accuracy, precision, recall = metric_sums[label][0] / count, metric_sums[label][1] / count, metric_sums[label][2] / count
        averages[label, :] = [accuracy, precision, recall]
        # DEBUG
        print("{:>8} | avg accuracy: {:.3f} avg precision: {:.3f} avg recall: {:.3f}".format(class_names[label], accuracy, precision, recall))
    avg_accuracy, avg_precision, avg_recall = np.mean(averages[:,0]), np.mean(averages[:,1]), np.mean(averages[:,2])
    print("{:>8} | avg accuracy: {:.3f} avg precision: {:.3f} avg recall: {:.3f}".format("average", avg_accuracy, avg_precision, avg_recall))
    # print("averages: {}".format(averages))
    
    f_score = (2*avg_precision*avg_recall)/(avg_precision+avg_recall)
    return f_score
    # return np.mean(averages[1:])  # this is kinda bad, but it's nice to have a single number




best_clf = None
best_score = 0


# # Decision Tree
# for i in [10, None]:
#     print("~" * 20)
#     print("Evaluating decision tree with max_depth=7 max_features={}".format(i))
#     clf = DecisionTreeClassifier(criterion="entropy", max_depth=7, max_features=i)
#     score = evaluate_model(clf)
#     # print("Used features:", clf.n_features_)
#     print("average precision recall: {:.3f}".format(score))

#     if score > best_score:
#         best_clf = clf
#         best_score = score


# # Random Forest
for i in [10, 20, 50]:  # 100 takes too long to train with double the features
# for i in [1, 2, 3]:  # 100 takes too long to train with double the features
# for i in [1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50]:  # 100 takes too long to train with double the features
# for i in list(range(1,41)):  # 100 takes too long to train with double the features
    print("~" * 20)
    print("Evaluating Random Forest with n_estimators={}".format(i))
    clf = RandomForestClassifier(n_estimators=i)
    score = evaluate_model(clf)
    print("average precision recall: {:.3f}".format(score))

    if score > best_score:
        best_clf = clf
        best_score = score


# # Random Forest
# parameters = {'n_estimators':[50],'criterion':['gini'],'max_depth':[50],'min_samples_leaf':[2],'class_weight':['balanced']}
# # parameters = {'n_estimators':[10],'criterion':['gini','entropy'],'max_depth':[5,10,50,100],'min_samples_leaf':[1,2,10],'class_weight':['balanced']}
# print("~" * 20)
# print("Evaluating Random Forest with grid search")
# clf = RandomForestClassifier()
# clf = GridSearchCV(clf, parameters)
# score = evaluate_model(clf)
# print(clf.best_params_)
# print("average precision recall: {:.3f}".format(score))
# if score > best_score:
#     best_clf = clf
#     best_score = score



# # KNN
# for i in [3]:
#     print("~" * 20)
#     print("Evaluating k-NN with k={}".format(i))
#     clf = KNeighborsClassifier(n_neighbors=i, weights='distance')
#     score = evaluate_model(clf)
#     print("average precision recall: {:.3f}".format(score))

#     if score > best_score:
#         best_clf = clf
#         best_score = score


# # Logistic Regression
# for i in [1E-1, 1E0, 1E1, 1E2]:
#     print("~" * 20)
#     print("Evaluating Logistic Regression with C={}".format(i))
#     clf = LogisticRegression(C=i)
#     score = evaluate_model(clf)
#     print("average precision recall: {:.3f}".format(score))

#     if score > best_score:
#         best_clf = clf
#         best_score = score


# # Support Vector Machine
# for i in [1E-3, 1E-2, 1E-1, 1E0, 1E1, 1E2]:
#     print("~" * 20)
#     print("Evaluating Support Vector Machine with C={}".format(i))
#     clf = SVC()
#     score = evaluate_model(clf)
#     print("average precision recall: {:.3f}".format(score))

#     if score > best_score:
#         best_clf = clf
#         best_score = score



# TODO: Once you have collected data, train your best model on the entire
# dataset. Then save it to disk as follows:
# when ready, set this to the best model you found, trained on all the data:
best_classifier = best_clf
best_clf.fit(X, y)
with open('training_output/classifier.pickle', 'wb') as f:  # 'wb' stands for 'write bytes'
    pickle.dump(best_classifier, f)
