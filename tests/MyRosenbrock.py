import numpy
import time

import ConfigSpace as CS
from hpbandster.core.worker import Worker

def f(x,y):
    a = 1
    b = 100
    return (a - x)**2 + b*(y - x**2)**2


class MyRosenbrock(Worker):

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

        res = f(config['x'], config['y'])

        return({
                    'loss': float(res),  # this is the a mandatory field to run hyperband
                    'info': res,  # can be used for any user-defined information - also mandatory
                    'x': config['x'],  # 
                    'y': config['y']  #
                })

    @staticmethod
    def get_configspace():
        config_space = CS.ConfigurationSpace()
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('x', lower=-5, upper=5))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('y', lower=-5, upper=5))
        return(config_space)
