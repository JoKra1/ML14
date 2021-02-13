# ML14 - Course: Machine Learning at the RUG

## Description
For this course we designed a pipeline to classify articles as unreliable or reliable.
Three models were evaluated in their ability to achieve this goal: Logistic regression, a naive Bayes classifier, and random forest models.
All three models achieve accruracies above > 90% on the final set aside test-set. This repository contains all files necessary to replicate the entire pipeline and to visualize the results.

## Data
The data is available at: https://www.kaggle.com/c/fake-news/data

## Requirements
The Python libraries necessary to run the code are listed in the requirements.txt file.

## Scripts

### Data generation scripts
- create_train.py: Creates all TF-IDF vector related training sets (train_c_red...) and the pre-processed sets used by the Bayes classifier (cleaned_train...)
- create_test.py: Pre-processes and generates the training data based on the set aside test-set

### Data generation class
- encoder.py: includes the relevant pre-processing steps and functionality to enable TF-IDF vector generation for train and test set

### Model validation scripts
- grid_search_forest.py: Performs OOB error score based grid-search to determine optimal forest parameters. Exports data for visualization in R
- tree_validate.py: Validates best forest model, obtained from grid_search_forest.py, using a 10 fold. Also calculates feature importance.
- logistic_cs.py: Performs CV to determine optimal logistic regression parameters

### Visualization extraction scripts
- pca_viz_extract.py: extracts eigen-values for variance explained calculation and visualization in R. Also extracts component vectors for visualization in R (projections of TF-IDF vectors on first k principal components)
- visualize_sentinet_sentence_stats_I.py: Calculates sentence statistics I and SentiNet scores and exports data for visualization in R

### Test submission scripts
- test_trees.py: Calculates submission file for random forest

## Folders

- visualization_R: Contains visualization script in R used to generate Figures for report
- tf_models: Contains PCA and forest implementation
- scratch_builds: Contains PCA code used for variance calculations and a cross-validation script
- naive_bayesian: Contains all naive Bayes classifier related code
    - sentiwordnet: Contains Sentinet file
    - submission: Contains computations of submission file to Kaggle
    - encoding.py: Calculations of sentence statistics and SentiNet scores
    - ensembler.py: Used to obtain majority vote of final models for Kaggle ensemble submission
    - naiveBayes.py: naive Bayes classifier implementation.
- data: Contains submission files to Kaggle
