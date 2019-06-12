"""
PROJECT: Classifiers based on Deep Learning and Domain Adaptation for EEG
         signals from different subjects
File:   loocv.py include complementary functions to train ResRecNet.
        
AUTOR:  PhD Student. Magdiel Jiménez Guarneros
        email: magdiel.jg@inaoep.mx
        Instituto Nacional de Astrofísica, Óptica y Electrónica (INAOE)
"""

import numpy as np

def loocv(domains):
    """
    Generate Leave-Subject-Out cross validation
    """
    
    fold_pairs = []
    
    for i in np.unique(domains):
        #print(i)
        ts = domains == i       #return array with True where the index i is equal to indices in subjNumbers
        tr = np.squeeze(np.nonzero(np.bitwise_not(ts))) #first return array with Trues where the index i is equal to indices in subjNumbers but inverted
                                                        #after convert this array of numbers.
        ts = np.squeeze(np.nonzero(ts))                 #conver ts with trues to array with numbers
        
        
        np.random.shuffle(tr)       # Shuffle indices
        np.random.shuffle(ts)
        fold_pairs.append((tr, ts))
    
    
    return fold_pairs




