#!/usr/bin/python3
"""
PROJECT: Classifiers based on Deep Learning and Domain Adaptation for EEG
         signals from different subjects
File:   run_main.py used to run "Residual Recurrent Neural Network" (ResRecNet)
        applied on Cognitive Load.
        
        Jiménez-Guarneros M., Gómez-Gil P. "Cross-subject classi-
        fication of cognitive loads using a recurrent-residual 
        deep network". IEEE Symposium Series on Computational Inte-
        lligence (IEEE SSCI 2017).
        
AUTOR:  PhD Student. Magdiel Jiménez Guarneros
        email: magdiel.jg@inaoep.mx
        Instituto Nacional de Astrofísica, Óptica y Electrónica (INAOE)
"""
    
import numpy as np
from preprocessing.modules_pbashivan import load_bashivan_data
from evaluation.loocv import loocv
from preprocessing.modules_pbashivan import split_cognitive_load_data
from train_resrnn import RESRNNTrainer
import argparse
import random
import tensorflow as tf
import os

# ** COGNITIVE LOAD DATA **
# Command to run ResRecNet on terminal:
#CUDA_VISIBLE_DEVICES=0 python3 run_main.py --model resrecnet --dataset pbashivan --output outputs/resrecnet


def loocv_cognitive_load(X, Y, subjects, model, output, args):
    """
    Leave One Out Cross Validation (LOOCV) on Cognitive Load DATA
    *******************************************************
    Params
    X: all dataset
    Y: labels of classes.
    subjects: labels of subject.
    model: architecture to be used.
    """
    list_metrics_clsf = []
    
    np.random.seed(args.seed)
    random.seed(args.seed)
    tf.set_random_seed(args.seed)
    
    print("SEED:",args.seed)
        
    # Extract pairs between indicies and subjects
    fold_pairs = loocv(subjects)
    
    # Iterate over fold_pairs
    for foldNum, fold in enumerate(fold_pairs):
        print('Beginning fold {0} out of {1}'.format(foldNum+1, len(fold_pairs)))
        
        # Divide the dataset into train, validation and test sets
        (Sx_train, Sy_train), (Sx_valid, Sy_valid), (Tx_train, Ty_train), (Tx_test, Ty_test), d_train, y_classes = split_cognitive_load_data(X, Y, subjects, fold)
        
        print("Sx_train-shape:",Sx_train.shape, "Sx_valid-shape:",Sx_valid.shape)
        print("Tx_train-shape:",Tx_train.shape, "Tx_test-shape:",Tx_test.shape)
        print("y_classes:", y_classes)
        
        if model == "resrecnet":
            
            classification_metrics = RESRNNTrainer(Sx_train, Sy_train, Sx_valid, Sy_valid, Tx_train, Ty_train, 
                "resrecnet", output+"/sub_"+str(foldNum+1), iterations=30,  seed=args.seed,
                batch_size=64, display=1, lr=0.001, snapshot=10, 
                solver="adam", n_classes = len(y_classes), 
                stepsize = 10)
        else:
            raise Exception("Unknown model %s." % model)
    
        # add to list
        list_metrics_clsf.append(classification_metrics)
        print()
    
    # To np array
    list_metrics_clsf = np.array(list_metrics_clsf)
    
    print("CLASSIFICATION METRICS:")
    for i in range(len(list_metrics_clsf[0])):
        mean = list_metrics_clsf[:,i].mean()
        print("Metric [",(i+1),"] = ", list_metrics_clsf[:,i]," Mean:", mean)
    
    np.savetxt(output+"/metrics-classification.csv", list_metrics_clsf, delimiter=",", fmt='%0.4f')
    

def main(args):
    
    # Create directorys
    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    
    # Load dataset
    if args.dataset == "pbashivan":
        # load cognitive load data
        X, y, subjects = load_bashivan_data("/home/magdiel/Descargas/datasets/PBashivan/", 
                            n_channels=64, n_windows=7, n_bands=3, generate_images=False, 
                            size_image=16, visualize=False)
        # run model
        loocv_cognitive_load(X, y, subjects, args.model, args.output, args)
    
    else:
        raise Exception("Unknown dataset %s." % args.model)


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="resrnn", help='name of model to execute')
parser.add_argument('--dataset', type=str, default="pbashivan", help='name of database')
parser.add_argument('--output', type=str, default="test01", help='name of output folder results')
parser.add_argument('--dir_resume', type=str, default="outputs/resume", help='folder for resume')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate value')
parser.add_argument('--seed', type=int, default=123, help='seed')
args = parser.parse_args()

# Call main module
main(args)
