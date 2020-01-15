# What about BOHB?

Using [BOHB](https://github.com/automl/HpBandSter) we run some comparisons of [SPORF](https://github.com/neurodata/SPORF) v. [sklearn-RF](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).


I ran this on Synaptomes1 with 56 threads. The first run used 54
datasets from [openml_s_225](https://www.openml.org/s/225).


## To Run:

1. set `nJOBS` and `NIC` to appropriate values in `syn1MASTER.src`
2. The data-IDs come from `dataID_CC18.dat`
3. make sure there is a directory called `output`.
4. run `bash syn1MASTER.scr`


## Results:

[Run on Synaptomes1 of 100-Friendly](http://docs.neurodata.io/bohb_runs_sporf/Rplots/plotResults_Friendly.html)

[Run on Synaptomes1 of CC18](http://docs.neurodata.io/bohb_runs_sporf/Rplots/plotResults_CC18.html)


---
