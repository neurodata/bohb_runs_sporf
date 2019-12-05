import logging
logging.basicConfig(level=logging.WARNING)

import argparse

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres

from hpbandster.optimizers import BOHB as BOHB
import matplotlib.pyplot as plt
import hpbandster.visualization as hpvis
#from hpbandster.examples.commons import MyWorker

from Workers import allWorker as MyWorker

import numpy
import time

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from hpbandster.core.worker import Worker

import openml

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

from rerf.rerfClassifier import rerfClassifier

def getID(tid = 6, test_size = 0.5, random_seed = 1234):

    dataset = openml.datasets.get_dataset(tid)

    # Print a summary
    print("This is dataset '%s', the target feature is '%s'" %
          (dataset.name, dataset.default_target_attribute))
    print("URL: %s" % dataset.url)
    print(dataset.description[:500])
    print("\n\n\n")

    X, Y, attribute_names,_ = dataset.get_data(target=dataset.default_target_attribute)
    
    if Y.dtype.name == "category":
        y = Y.cat.codes
    else:
        y = Y


    X, X_test, y, y_test = train_test_split(X, y, test_size = test_size,\
                                            random_state = random_seed) 

    return(X, y, X_test, y_test)


X, y, X_test, y_test = getID()

## failed (1,0,0), (4,0,1), (3,0,8)

#(1,0,0) = depth = 9. mf = 2.0, fc = 1.0

#(3,0,8) = depth = 6236. mf = 2.25, fc = 2.0

#(4,0,1) = depth = 5644. mf = 2.25, fc = 1.0

#(0,0,6) d:3050, mf = 1, fc = 2.0


clf = rerfClassifier(n_estimators = 7,\
                     max_features = 1.0,
                     max_depth = 3050,\
                     feature_combinations = 2.0, n_jobs = 1)



clf.fit(X, y)

clf.predict(X_test)
