#!/usr/bin/python3
"""
PROJECT: Classifiers based on Deep Learning and Domain Adaptation for EEG
         signals from different subjects
File:   metrics.py include complementary metrics to evaluate ResRecNet.
        
AUTOR:  PhD Student. Magdiel Jiménez Guarneros
        email: magdiel.jg@inaoep.mx
        Instituto Nacional de Astrofísica, Óptica y Electrónica (INAOE)
"""

import numpy as np
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import f1_score
from preprocessing.modules_jmnzg import format_digits_logits


class meanAccuracy:

    def filename(self):
        return "meanAccuracy"

    def run(self, y_true, y_pred):
        errors = 0
        s_a = []

        for i in range(len(y_true)):
            if y_true[i] != y_pred[i]:
                errors +=1
                s_a.append(0) #
            else:
                s_a.append(1) #

        acc = 1 - float(errors/float(y_true.shape[0]))

        return acc



def AUC(y_true, y_pred):

    nb_classes = np.unique(y_true)
    y_t = format_digits_logits(y_true, nb_classes)
    y_p = format_digits_logits(y_pred, nb_classes)

    n_events = y_t.shape[1]
    scores = np.zeros((n_events,), 'float32')
    for i in np.arange(n_events):
        fpr, tpr, _ = roc_curve(y_t[:, i], y_p[:, i])
        scores[i] = auc(fpr, tpr)

    return scores.mean()

    
class meanFmeasure:

    def filename(self):
        return "meanFmeasure"

    def run(self, y_true, y_pred):
        fmeasure = f1_score(y_true, y_pred, average='weighted')
        return fmeasure.mean()

    

# the scoring mechanism used by the competition leaderboard
class score_classifier_auc:

    def filename(self):
        return "Score"
    
    # convert the output of classifier predictions into (Seizure, Early) pair
    def translate_prediction(self, prediction, y_classes):
        if len(prediction) == 3:
            # S is 1.0 when ictal <=15 or >15
            # S is 0.0 when interictal is highest
            ictalLTE15, ictalGT15, interictal = prediction
            S = ictalLTE15 + ictalGT15
            E = ictalLTE15
            return S, E
        elif len(prediction) == 2:
            # 1.0 doesn't exist for Patient_4, i.e. there is no late seizure data
            if not np.any(y_classes == 1.0):
                ictalLTE15, interictal = prediction
                S = ictalLTE15
                E = ictalLTE15
                # y[i] = 0 # ictal <= 15
                # y[i] = 1 # ictal > 15
                # y[i] = 2 # interictal
                return S, E
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()
        
        

    def run(self, predictions, y_test, y_classes):
    
        #seizures prediction (include early)
        S_predictions = []
        #early seizures 
        E_predictions = []

        #After ajust the classes 
        #Vector que indica donde hay ataques (early 0 o seizure 1)
        S_y_cv = [1.0 if (x == 0.0 or x == 1.0) else 0.0 for x in y_test]
        #Vector que indica donde el ataque es early 0
        E_y_cv = [1.0 if x == 0.0 else 0.0 for x in y_test]


        for i in range(len(predictions)):
            #probabilities for classes 0,1,2
            p = predictions[i] 
            #get probabilities for classes seizure and early
            S, E = self.translate_prediction(p, y_classes)
            #append probabilities
            S_predictions.append(S)
            E_predictions.append(E)

        #Calculate roc curve
        fpr, tpr, thresholds = roc_curve(S_y_cv, S_predictions)
        #calculate AUC
        S_roc_auc = auc(fpr, tpr)
        #Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(E_y_cv, E_predictions)
        #calculate AUC
        E_roc_auc = auc(fpr, tpr)

        #Calculate score of competition
        score = 0.5 * (S_roc_auc + E_roc_auc)
        
        return score


def AUC(y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    return auc(fpr, tpr)

def kappa(y_true, y_pred):
    O = cm(y_true, y_pred)

    N = max(max(y_true), max(y_pred)) + 1
    W = np.zeros((N, N), 'float32')
    for i in np.arange(N):
        for j in np.arange(N):
            W[i, j] = (i - j) ** 2
    W /= ((N - 1) ** 2)

    hist_true = np.bincount(y_true, minlength = N)
    hist_pred = np.bincount(y_pred, minlength = N)
    E = np.outer(hist_true, hist_pred).astype('float32') / len(y_true)

    return 1 - (np.sum(W * O) / np.sum(W * E))

def confusion(y_true, y_pred):
    return cm(y_true, y_pred)

def ordinal_test(ys):
    pred = np.zeros((len(ys),), 'int32')
    for i, y in enumerate(ys):
        idx = -2
        for j in np.arange(len(y)):
            if y[j] < 0.5:
                idx = j - 1
                break
        if idx == -1:
            idx = 0
        if idx == -2:
            idx = len(y) - 1

        pred[i] = idx

    return pred



