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
from Workers import getID

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



parser = argparse.ArgumentParser(description='Example 1 - sequential and local execution.')
parser.add_argument('--min_budget',   type=int, help='Minimum budget used during the optimization.',    default=5)
parser.add_argument('--max_budget',   type=int, help='Maximum budget used during the optimization.',    default=100)
parser.add_argument('--n_iterations', type=int,   help='Number of iterations performed by the optimizer', default=4)
parser.add_argument('--n_workers', type=int,   help='Number of workers to run in parallel.', default=1)
parser.add_argument('--n_jobs', type=int,   help='Number of workers to run in parallel.', default=8)

args=parser.parse_args()



# Step 1: Start a nameserver (see example_1)
NS = hpns.NameServer(run_id='example2', host='127.0.0.1', port=None)
NS.start()

# Step 2: Start the workers
# Now we can instantiate the specified number of workers. To emphasize the effect,
# we introduce a sleep_interval of one second, which makes every function evaluation
# take a bit of time. Note the additional id argument that helps separating the
# individual workers. This is necessary because every worker uses its processes
# ID which is the same for all threads here.
workers=[]
for i in range(args.n_workers):
    w = MyWorker(sleep_interval = 0.05, nameserver='127.0.0.1',run_id='example2', id=i, n_jobs = args.n_jobs)
    w.run(background=True)
    workers.append(w)

# Step 3: Run an optimizer
# Now we can create an optimizer object and start the run.
# We add the min_n_workers argument to the run methods to make the optimizer wait
# for all workers to start. This is not mandatory, and workers can be added
# at any time, but if the timing of the run is essential, this can be used to
# synchronize all workers right at the start.
bohb = BOHB(  configspace = w.get_configspace(),
              run_id = 'example2',
              min_budget=args.min_budget, max_budget=args.max_budget
           )
res = bohb.run(n_iterations=args.n_iterations, min_n_workers=args.n_workers)

# Step 4: Shutdown
# After the optimizer run, we must shutdown the master and the nameserver.
bohb.shutdown(shutdown_workers=True)
NS.shutdown()

# Step 5: Analysis
# Each optimizer returns a hpbandster.core.result.Result object.
# It holds informations about the optimization run like the incumbent (=best) configuration.
# For further details about the Result object, see its documentation.
# Here we simply print out the best config and some statistics about the performed runs.
id2config = res.get_id2config_mapping()
incumbent = res.get_incumbent_id()

all_runs = res.get_all_runs()

print('Best found configuration:', id2config[incumbent]['config'])
print('A total of %i unique configurations where sampled.' % len(id2config.keys()))
print('A total of %i runs where executed.' % len(res.get_all_runs()))
print('Total budget corresponds to %.1f full function evaluations.'%(sum([r.budget for r in all_runs])/args.max_budget))
print('Total budget corresponds to %.1f full function evaluations.'%(sum([r.budget for r in all_runs])/args.max_budget))
print('The run took  %.1f seconds to complete.'%(all_runs[-1].time_stamps['finished'] - all_runs[0].time_stamps['started']))


d1 = res.get_pandas_dataframe()[0]
loss = res.get_pandas_dataframe()[1] 
d1['loss'] = loss

d1.to_csv()


if False:
    result = res
    # get all executed runs
    all_runs = result.get_all_runs()

    # get the 'dict' that translates config ids to the actual configurations
    id2conf = result.get_id2config_mapping()


    # Here is how you get he incumbent (best configuration)
    inc_id = result.get_incumbent_id()

    # let's grab the run on the highest budget
    inc_runs = result.get_runs_by_id(inc_id)
    inc_run = inc_runs[-1]


    # We have access to all information: the config, the loss observed during
    #optimization, and all the additional information
    inc_loss = inc_run.loss
    inc_config = id2conf[inc_id]['config']
    inc_train_loss = inc_run.info['train_accuracy']
    inc_test_loss = inc_run.info['test_accuracy']

    print('Best found configuration:')
    print(inc_config)
    print('It achieved accuracy of %f (train) and %f (test).'%(inc_train_loss, inc_test_loss))


    # Let's plot the observed losses grouped by budget,
    hpvis.losses_over_time(all_runs)

    # the number of concurent runs,
    hpvis.concurrent_runs_over_time(all_runs)

    # and the number of finished runs.
    hpvis.finished_runs_over_time(all_runs)

    # This one visualizes the spearman rank correlation coefficients of the losses
    # between different budgets.
    hpvis.correlation_across_budgets(result)

    # For model based optimizers, one might wonder how much the model actually helped.
    # The next plot compares the performance of configs picked by the model vs. random ones
    hpvis.performance_histogram_model_vs_random(all_runs, id2conf)

    plt.show()

    d1 = res.get_pandas_dataframe()[0]
    loss = res.get_pandas_dataframe()[1] 

    d1['loss'] = loss





if False:
    result = res
    # get all executed runs
    all_runs = result.get_all_runs()
    
    # get the 'dict' that translates config ids to the actual configurations
    id2conf = result.get_id2config_mapping()
    
    lcs = result.get_learning_curves()
    
    hpvis.interactive_HBS_plot(lcs, tool_tip_strings=hpvis.default_tool_tips(result, lcs))


