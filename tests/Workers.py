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



class MyWorker(Worker):

    def __init__(self, *args, sleep_interval=0, **kwargs):
        super().__init__(*args, **kwargs)

        self.sleep_interval = sleep_interval

    def compute(self, config, budget, **kwargs):
        """
        Simple example for a compute function
        The loss is just a the config + some noise (that decreases with the budget)

        For dramatization, the function can sleep for a given interval to emphasizes
        the speed ups achievable with parallel workers.

        Args:
            config: dictionary containing the sampled configurations by the optimizer
            budget: (float) amount of time/epochs/etc. the model can use to train

        Returns:
            dictionary with mandatory fields:
                'loss' (scalar)
                'info' (dict)
        """

        res = numpy.clip(config['x'] + numpy.random.randn()/budget, config['x']/2, 1.5*config['x'])
        time.sleep(self.sleep_interval)

        return({
                    'loss': float(res),  # this is the a mandatory field to run hyperband
                    'info': res  # can be used for any user-defined information - also mandatory
                })

    @staticmethod
    def get_configspace():
        config_space = CS.ConfigurationSpace()
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('x', lower=0, upper=1))
        return(config_space)



class allWorker(Worker):

    def __init__(self, *args, sleep_interval=0, n_jobs = 1, **kwargs):
        super().__init__(*args, **kwargs)

        self.sleep_interval = sleep_interval
        self.n_jobs = n_jobs

        self.X, self.y, self.X_test, self.y_test = getID(tid = 11, test_size = 0.5, random_seed = 0)



    def compute(self, config, budget, **kwargs):
        """

        Args:
            config: dictionary containing the sampled configurations by the optimizer.
            budget: (int) number of trees the model is allowed to use in training.

        Returns:
            dictionary with mandatory fields:
                'loss' (scalar)
                'info' (dict)
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

        train_pred = clf.predict(self.X)
        train_accuracy = metrics.accuracy_score(self.y, train_pred)

        yhat = clf.predict(self.X_test)

        res = metrics.accuracy_score(self.y_test, yhat)

        return({  
                    # this is the a mandatory field to run hyperband
                    'loss': float(1-res),
                    # can be used for any user-defined information - also mandatory
                    'info': {"test_accuracy": res, 
                             "train_accuracy": train_accuracy,
                            }  
                })

    @staticmethod
    def get_configspace():
        cs = CS.ConfigurationSpace()

        clf = CSH.CategoricalHyperparameter('clf', ['skrf', 'sporf'])
        sporf_fc = CSH.CategoricalHyperparameter('sporf_fc', [1.0, 1.5, 2.0, 2.5, 3.0])

        cs.add_hyperparameters([clf, sporf_fc])

        mf_sporf = CSH.UniformFloatHyperparameter('max_features_sporf', lower=0.01, upper=5.0)
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



class SKrfWorker(Worker):

    def __init__(self, *args, sleep_interval=0, **kwargs):
        super().__init__(*args, **kwargs)

        self.sleep_interval = sleep_interval

        self.X, self.y, self.X_test, self.y_test = getID(tid = 6, test_size = 0.5, random_seed = 0)



    def compute(self, config, budget, **kwargs):
        """

        Args:
            3onfig: dictionary containing the sampled configurations by the optimizer.
            budget: (int) number of trees the model is allowed to use in training.

        Returns:
            dictionary with mandatory fields:
                'loss' (scalar)
                'info' (dict)
        """

        time.sleep(self.sleep_interval)
        clf = RandomForestClassifier(n_estimators = int(budget),\
                                     max_features = config['max_features'],
                                     max_depth = config['max_depth'])
        clf.fit(self.X, self.y)

        yhat = clf.predict(self.X_test)

        res = 1 - metrics.accuracy_score(self.y_test, yhat)

        return({
                    'loss': float(res),  # this is the a mandatory field to run hyperband
                    'info': {"test_accuracy": 1 - res}  # can be used for any user-defined information - also mandatory
                })

    @staticmethod
    def get_configspace():
        config_space = CS.ConfigurationSpace()
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('max_features', lower=0.01, upper=0.9))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('max_depth', log = True,\
                                                                      lower=2, upper=8192))
        return(config_space)



class SporfWorker(Worker):

    def __init__(self, *args, sleep_interval=0, **kwargs):
        super().__init__(*args, **kwargs)

        self.sleep_interval = sleep_interval

    def compute(self, config, budget, **kwargs):
        """
        Simple example for a compute function
        The loss is just a the config + some noise (that decreases with the budget)

        For dramatization, the function can sleep for a given interval to emphasizes
        the speed ups achievable with parallel workers.

        Args:
            config: dictionary containing the sampled configurations by the optimizer
            budget: (float) amount of time/epochs/etc. the model can use to train

        Returns:
            dictionary with mandatory fields:
                'loss' (scalar)
                'info' (dict)
        """

        res = numpy.clip(config['x'] + numpy.random.randn()/budget, config['x']/2, 1.5*config['x'])
        time.sleep(self.sleep_interval)

        return({
                    'loss': float(res),  # this is the a mandatory field to run hyperband
                    'info': res  # can be used for any user-defined information - also mandatory
                })

    @staticmethod
    def get_configspace():
        config_space = CS.ConfigurationSpace()
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('x', lower=0, upper=1))
        return(config_space)




