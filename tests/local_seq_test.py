import logging
logging.basicConfig(level=logging.WARNING)

import argparse

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres

import matplotlib.pyplot as plt
import hpbandster.visualization as hpvis

from hpbandster.optimizers import BOHB as BOHB
#from hpbandster.examples.commons import MyWorker
#from MyWorker import MyWorker
from Workers import SKrfWorker as MyWorker


parser = argparse.ArgumentParser(description='Example 1 - sequential and local execution.')
parser.add_argument('--min_budget',   type=int, help='Minimum budget used during the optimization.',    default=10)
parser.add_argument('--max_budget',   type=int, help='Maximum budget used during the optimization.',    default=500)
parser.add_argument('--n_iterations', type=int,   help='Number of iterations performed by the optimizer', default=3)
parser.add_argument('--openml_id', type=int, help='OpenML Data ID', default=6)
args=parser.parse_args()


# Step 1: Start a nameserver
# Every run needs a nameserver. It could be a 'static' server with a
# permanent address, but here it will be started for the local machine with the default port.
# The nameserver manages the concurrent running workers across all possible threads or clusternodes.
# Note the run_id argument. This uniquely identifies a run of any HpBandSter optimizer.
NS = hpns.NameServer(run_id='example1', host='127.0.0.1', port=None)
NS.start()

# Step 2: Start a worker
# Now we can instantiate a worker, providing the mandatory information
# Besides the sleep_interval, we need to define the nameserver information and
# the same run_id as above. After that, we can start the worker in the background,
# where it will wait for incoming configurations to evaluate.
w = MyWorker(nameserver='127.0.0.1',run_id='example1')
w.run(background=True)

# Step 3: Run an optimizer
# Now we can create an optimizer object and start the run.
# Here, we run BOHB, but that is not essential.
# The run method will return the `Result` that contains all runs performed.
bohb = BOHB(  configspace = w.get_configspace(),
              run_id = 'example1', nameserver='127.0.0.1',
              min_budget=args.min_budget, max_budget=args.max_budget
           )
res = bohb.run(n_iterations=args.n_iterations)

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

print('Best found configuration:', id2config[incumbent]['config'])
print('A total of %i unique configurations where sampled.' % len(id2config.keys()))
print('A total of %i runs where executed.' % len(res.get_all_runs()))
#print('Total budget corresponds to %.1f full function evaluations.'%(sum([r.budget for r in res.get_all_runs()])/args.max_budget))




result = res
if False:
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
    inc_test_loss = inc_run.info['test accuracy']

    print('Best found configuration:')
    print(inc_config)
    print('It achieved accuracies of %f (validation) and %f (test).'%(1-inc_loss, inc_test_loss))


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
