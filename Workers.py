import numpy
import time

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from hpbandster.core.worker import Worker

import openml

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

# using v2.0.5.1 https://github.com/neurodata/SPORF/tree/v2.0.5.1
from rerf.rerfClassifier import rerfClassifier


'''

X, y, X_test, y_test = getID(dataID = 15, test_size = 0.5, val_size = 0.2, random_seed = 1234)

'''

def getTimes(res):

    runs = res.get_all_runs()
    out = []
    for ri in runs:
        times = ri['time_stamps']
        total = times['finished'] - times['started']
        out.append(total)

    return(out)


def getID(dataID = 6, test_size = 0.3, val_size = 0.2, random_seed = 1234):

    dataset = openml.datasets.get_dataset(dataID)

    # Print a summary
    print("This is dataset '%s', the target feature is '%s'" %
          (dataset.name, dataset.default_target_attribute))
    print("URL: %s" % dataset.url)
    print(dataset.description[:500])
    print("\n\n\n")

    ## X_original, Y_original
    Xo, Yo, attribute_names,_ = dataset.get_data(target=dataset.default_target_attribute)

    if Yo.dtype.name == "category":
        y = Yo.cat.codes
    else:
        y = Yo

    if Xo.isna().sum().sum() > 0:
        ind = ~Xo.isna().any(axis = 1)
        Xo.dropna(inplace = True)
        y = y[ind]
        

    if numpy.any(Xo.dtypes == 'category'):
        cat_columns = Xo.select_dtypes(['category']).columns

        Xo[cat_columns] = Xo[cat_columns].apply(lambda x: x.cat.codes)


    Xa, X_test, ya, y_test = train_test_split(Xo, y, test_size = test_size,\
                                              random_state = random_seed) 

    ## Further subset into train and validation
    X, X_val, y, y_val = train_test_split(Xa, ya, test_size = val_size,\
                                          random_state = random_seed)

    return(X, y, X_val, y_val, X_test, y_test)



class allWorker(Worker):

    def __init__(self, *args, sleep_interval=0, n_jobs = 1, dataID, **kwargs):
        super().__init__(*args, **kwargs)

        self.sleep_interval = sleep_interval
        self.n_jobs = n_jobs
        self.dataID = dataID 

        self.X, self.y, self.X_val, self.y_val, self.X_test, self.y_test =\
          getID(dataID = self.dataID, test_size = 0.5, val_size = 0.2,\
                random_seed = 0)



    def compute(self, config, budget, **kwargs):
        """

        Args:
            config: dictionary containing the sampled configurations by the optimizer.
            budget: (int) number of trees the model is allowed to use in training.

        Returns:
            dictionary with mandatory fields:
                'loss' (scalar)
                'info' (dict)

            clf = RandomForestClassifier(n_estimators = int(budget),\
                                         n_jobs = n_jobs,\
                                         max_features = 0.9,\
                                         max_depth = None)
        """

        if config['clf'] == "skrf":
            clf = RandomForestClassifier(n_estimators = int(budget),\
                                         n_jobs = self.n_jobs,\
                                         max_features = config['max_features_sk'],\
                                         max_depth = config['max_depth'])

        elif config['clf'] == "sporf": 
            clf = rerfClassifier(n_estimators = int(budget),\
                                 max_features = config['max_features_sporf'],\
                                 max_depth = config['max_depth'],\
                                 feature_combinations = config['sporf_fc'],\
                                 n_jobs = self.n_jobs,\
                                 projection_matrix = "RerF",\
                                )
    
        clf.fit(self.X, self.y)
        #clf.fit(X, y)

        train_pred = clf.predict(self.X)
        train_accuracy = metrics.accuracy_score(self.y, train_pred)


        y_val_hat = clf.predict(self.X_val)
        val_accuracy = metrics.accuracy_score(self.y_val, y_val_hat)

        y_test_hat = clf.predict(self.X_test)
        test_accuracy = metrics.accuracy_score(self.y_test, y_test_hat)

        return({  
                    # this is the a mandatory field to run hyperband
                    'loss': float(1-val_accuracy),
                    # can be used for any user-defined information - also mandatory
                    'info': {
                             "test_loss" : 1 - test_accuracy,
                             "test_accuracy" : test_accuracy,
                             "val_accuracy"  : val_accuracy,
                             "train_accuracy": train_accuracy,
                            }  
                })

    @staticmethod
    def get_configspace():
        cs = CS.ConfigurationSpace()

        clf = CSH.CategoricalHyperparameter('clf', ['skrf', 'sporf'])
        sporf_fc = CSH.CategoricalHyperparameter('sporf_fc', [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 8.0])

        cs.add_hyperparameters([clf, sporf_fc])

        mf_sporf = CSH.UniformFloatHyperparameter('max_features_sporf', lower=0.01, upper=4.0)
        mf_sk = CSH.UniformFloatHyperparameter('max_features_sk', lower=0.01, upper=0.9)

        #mf_sporf = CSH.CategoricalHyperparameter('max_features_sporf', [i / 100 for i in range(25, 420, 25)])
        #mf_sk = CSH.CategoricalHyperparameter('max_features_sk', [i / 100 for i in range(25,100,25)] + [0.9])

        #cs.add_hyperparameters([mf_sporf, mf_sk])
        cs.add_hyperparameters([mf_sporf, mf_sk])
        cs.add_hyperparameter(CS.UniformIntegerHyperparameter('max_depth', log = True,\
                                                              lower=2, upper=65535))

        # The hyperparameter sporf_fc will only be used if the
        # classifier is sporf.
        cond1 = CS.EqualsCondition(sporf_fc, clf, "sporf")
        cs.add_condition(cond1)

        ## Set max_features \in (0,4) if clf == 'sporf'
        cond2 = CS.EqualsCondition(mf_sporf, clf, "sporf")
        cs.add_condition(cond2)

        ## Set max_features \in (0,1) if clf == 'skrf'
        cond3 = CS.EqualsCondition(mf_sk, clf, "skrf")
        cs.add_condition(cond3)

        return(cs)



