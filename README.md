# OpenWorldLearningScenarios

## Installation

* Use the requirements.txt to install all the necessary packages
* Use the repo https://github.com/lab176344/scikit-THI_CIAD to install the package to run RF similarity

## Dataset 
Use the link (https://1drv.ms/f/s!ArclEiu1Sj0Oj4YbmkAV56WFwsd08A?e=wwUmRO) to download the dataset and paste it under data/

## Running the experiments

The experiment has 4 steps
1. First run the `selfsupervised_learning_scenario_temporal_shuffling.py`, this is to run the self supervised model with temporal shuffling. Alternativelym barlow twins based self supervised learning can also be run with `selfsupervised_learning_scenario_barlow_twins.py`

2. The second step is to run the supervised learning model, it is referred as closed-world model in paper. Run `supervised_learning_scenarios.py`

3. The third step is to run the OSR model `rf_evt_train.py` 

Adjsut step 2 and 3 with args parameter to change the number of closed and open set classes

4. Clustering of the open-set classes can be run finally by `clustering_scenarios.py`

