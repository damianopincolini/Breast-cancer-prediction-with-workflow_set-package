# Breast-cancer-prediction-with-workflow_set-package
As stated in the on line documentation (<https://workflowsets.tidymodels.org/>), the goal of workflowsets package is to allow users to create and easily fit a large number of models and preprocessing recipes. In fact, workflowsets can create a workflow set that holds multiple workflow objects. These objects can be created by crossing all combinations of preprocessors (e.g., formula, recipe, etc) and model specifications. This set can be tuned or resampled using a set of specific functions. Aiming to better understand how this package works, this project is conceived as a pretext for experimenting this tool from the R's tidymodels ecosystem.

I have picked Breast Cancer Wisconsin (Diagnostic) dataset from kaggle.com (<https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset?select=breast-cancer.csv>) which is suitable for the purpose of this work. Specifically, it contains some outcome of some cells analysis made to discover if a cancer exists or not. Breast cancer starts when cells in the breast begin to grow out of control. These cells usually form tumors that can be seen via X-ray or felt as lumps in the breast area. Starting from some measures of these cells, the key challenges is how to classify tumors as malignant (cancerous) or benign(non cancerous).

With this aim in mind, I want to train and test some different models combined with some different preprocess recipes.

This repository contains:
1. The dataset from UCI: breastCancer.csv.
2. The Quarto file which contains the code I've used to develop this project and turn it into a pdf report: BreastCancerProject.qmd.
3. The R file which contains R code I've used to develop this project: BreastCancerProject.R.
4. The final report in pdf format: BreastCancerProject.pdf.
