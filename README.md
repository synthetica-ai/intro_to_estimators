# Synthetica data science internship

## Eagle's eye view of the project's targets

* **Discover by yourself** the problems that arise in a production scope ml model building(Without the data curation part).
* Build neural network architecture to predict red wine quality. We will discuss it thoroughly after your make your first try. This is **not** the main purpose of the project.
* Save model as a .pb file.
* Demo http request to get predictions on production mode.
* Use **only** custom estimator to build all the minor skills needed to create anything you need in the long run.
* Use **only** git to become familiar with this toolset. Get used to staging some code change, committing it, pushing it on master or pulling team's changes.

## Worm's eye view of the project's targets
* Create train_input and eval_input and serving_input_receiver functions (located in estimator_functions.py).
* Create model_fn for all 3 modes ( training ,evaluation and        prediction) (located in estimator_functions.py).
* Create metaparameters with every hyperparameter and metadata needed (located in estimator_functions.py).
* Create estimator with parameters all the above things (located in train.py).
* train_and_evaluate the model with proper losses, metrics etc. Save logs inside logs folder.
* Export saved model (located in saved_model).



## File structure


The philosophy of this structure used in Synthetica is to create main training script (train.py) as minimal and readable as possible. We also like to separate and isolate the different parts of ml code and import everything when needed where it is needed (that's why init.py is inside those folders)



## General comment

* Don't hesitate to ask all the minor things that come up.
