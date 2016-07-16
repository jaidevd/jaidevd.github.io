Automate:
* Choice of algorithm
* Feature preprocessing
* Hyperparameter tuning

auto-sklearn
------------

* 15 classifiers, 14 feature extraction methods, 4 preprocessing methods.
110 total hyperparameters
* it automates ensembles!


The AutoML problem
------------------

* automatically produce test set predcitions for a dataset within fixed
computational budget.
* AutoML as a CASH problem: Combined algorithm selection and hyperparameter
optimization. ![](automl.png)


Additions by Feurer et al
-------------------------

* meta-learning step in the automl pipeline to warmstart the Bayes
  optimization.
* automated ensemble construction - iteratively use all classifiers found by
  the Bayesian optimization


Automated Ensemble Construction of Models
-----------------------------------------

* for models from the bayesian hyperparameter optimization, instead of
  discarding them, store all of them, and make an ensemble.
* 
