# 1. PREFACE

# 1.1. Project goal
# I want to train and test some different models combined with some different preprocess
# recipes.

# 1.2. Loading packages

pacman::p_load(tidyverse,
               SmartEDA,
               DescTools,
               factoextra,
               ggcorrplot,
               GGally,
               ggforce,
               corrplot,
               cowplot,
               tidymodels,
               randomForest,
               kknn,
               kernlab,
               broom,
               themis,
               rpart.plot,
               vip,
               shapviz,
               knitr)

# 1.3. Data loading and content analysis

# After downloading and saving the file breast-cancer.csv from kaggle (see link above),
# I save into R environment.


setwd("C:/Users/casa/Documents/Damiano/R/Mentoring/16. BreastCancer/RawData")
# the path must be customized. I've obviously used the path on my personal laptop.

DataOrigin <- read_csv("RawData/breastCancer.csv",
                       show_col_types = FALSE)

setwd("C:/Users/casa/Documents/Damiano/R/Mentoring/16. BreastCancer")
# the path must be customized.


# 1.4. DATASET ANALYSIS

# 1.4.1. Dataset structure and datatype analysis

DataExp1 <- ExpData(DataOrigin, type=1)

DataExp1|>
  kable()

DataExp2 <- ExpData(DataOrigin, type=2)

DataExp2|>
  kable()

summary(DataOrigin)

# To sum up what has arisen from this initial evaluation, dataset may need the following
# changes:
# 1. De-select "id" column which does not bring any information.
# 2. Convert "diagnosis" column from character to factor.
# 3. Rename predictors names by swapping blank spaces with "_" Since I haven't found any
# detail about the unit of measurement, I take for granted that the scale is the same
# for all predictors.

# 1.4.2. Data featuring (conversion to factor, binning, renaming, etc)

Data <- DataOrigin|>
  select(-id)|>
  mutate(diagnosis = as_factor(diagnosis))|>
  rename(concave_points_mean = 'concave points_mean',
         concave_points_se = 'concave points_se')

# 1.5. DATA PARTITIONING

set.seed(987, sample.kind = "Rounding")

DataSplit <- Data|>
  initial_split(prop = 0.80,
                strata = diagnosis)

DataTrain <- training(DataSplit)

DataTest <- testing(DataSplit)

# I want to check that target variable's proportion in the train and test dataset
# is the same as in the original dataset.

prop.table(table(Data$diagnosis))|>
  kable()

prop.table(table(DataTrain$diagnosis))|>
  kable()

prop.table(table(DataTest$diagnosis))|>
  kable()


# 02. EDA

# 2.1 Target analysis

# 2.2. Univariate analysis

# 2.2.1 distribution analysis for each predictor

# In order to have a bird's eye view of the different behaviour of how target variable
# "replies" to each single predictor, I prefer not to create a single plot for each
# feature; instead, I'll plot a wrapped collection of all of the plots I want to
# visualize in a single "frame".
# This should help reducing coding time and improving overall readability.
# To do that, I will reframe with pivot_longer command the DataTrain tibble bringing
#the 20 feature into one (longer) column that is going to include every single variable.

options(scipen = 999)

DataTrainLong <- DataTrain|>
  pivot_longer(cols = -diagnosis, 
               names_to = "variable",
               values_to = "value")

DataTrainLong|>
  ggplot(aes(x = value,
             fill = diagnosis))+
  geom_density(alpha = 0.30)+
  facet_wrap_paginate(~ variable,
                      ncol = 3,
                      nrow = 3,
                      scales = "free",
                      page = 1)+
  theme_minimal(base_size = 8)

DataTrainLong|>
  ggplot(aes(x = value,
             fill = diagnosis))+
  geom_density(alpha = 0.30)+
  facet_wrap_paginate(~ variable,
                      ncol = 3,
                      nrow = 3,
                      scales = "free",
                      page = 2)+
  theme_minimal(base_size = 8)

DataTrainLong|>
  ggplot(aes(x = value,
             fill = diagnosis))+
  geom_density(alpha = 0.30)+
  facet_wrap_paginate(~ variable,
                      ncol = 3,
                      nrow = 3,
                      scales = "free",
                      page = 3)+
  theme_minimal(base_size = 8)


# As far as skewness and kurtosis are concerned, SmartEDA package will help in getting
# another view of these measures.

EdaNum <- DataTrain|>
  ExpNumStat(by = "GA",
             gp = "diagnosis",
             Qnt = c(0.25, 0.75),
             Nlim = 2,
             MesofShape = 2,
             Outlier = TRUE,
             round = 2)

EdaDistributionShape <- EdaNum|>
  select(Vname, Group, Skewness, Kurtosis)|>
  filter(Group == 'diagnosis:All')|>
  arrange(desc(Skewness))

EdaDistributionShape|>
  ggplot()+
  geom_col(aes(reorder(Vname, Kurtosis), Kurtosis), alpha = 0.5, fill = "orange")+
  coord_flip()+
  labs(x = "Kurtosis", y = "Predictors",
       title ="Kurtosis")
  
EdaDistributionShape|>
  ggplot()+
  geom_col(aes(reorder(Vname, Skewness), Skewness), alpha = 0.5, fill = "blue")+
  coord_flip()+
  labs(x = "Skewness", y = "Predictors",
       title ="Skewness")

EdaDistributionShape|>
  ggplot()+
  geom_col(aes(reorder(Vname, Kurtosis), Kurtosis), alpha = 0.5, fill = "orange")+
  geom_col(aes(Vname, Skewness), alpha = 0.5, fill = "blue")+
  coord_flip()+
  labs(x = "Skewness and Kurtosis", y = "Predictors",
         title ="Distribution shape")+
  guides(fill = "color")


# 2.2.2. outlier detection

DataTrainLong|>
  ggplot(aes(x = value,
             fill = diagnosis))+
  geom_boxplot()+
  facet_wrap_paginate(~ variable,
                      ncol = 3,
                      nrow = 3,
                      scales = "free",
                      page = 1)+
  theme_minimal(base_size = 8)

DataTrainLong|>
  ggplot(aes(x = value,
             fill = diagnosis))+
  geom_boxplot()+
  facet_wrap_paginate(~ variable,
                      ncol = 3,
                      nrow = 3,
                      scales = "free",
                      page = 2)+
  theme_minimal(base_size = 8)

DataTrainLong|>
  ggplot(aes(x = value,
             fill = diagnosis))+
  geom_boxplot()+
  facet_wrap_paginate(~ variable,
                      ncol = 3,
                      nrow = 3,
                      scales = "free",
                      page = 3)+
  theme_minimal(base_size = 8)


# 2.3. Multivariate analysis

# 2.3.1 Correlation between numerical predictors

EdaCorMatr <- round(cor(DataTrain[,c(2:31)], use="complete.obs"), 1)

ggcorrplot(EdaCorMatr,
           hc.order = TRUE,
           type = "lower",
           lab = TRUE,
           lab_size = 1,
           tl.cex = 5,
           digits = 1)


# 2.3.2. Association between categorical predictors

# There are no categorical predictor to look for association.

# 2.3.3. Principal Component Analysis (factoextra approach)

# All predictors are numeric. This could be a good use case for PCA analysis in order
# to try to reduce multidimensionality.

# I start by running the PCA and storing the result in a variable pca_fit.
# There are two issues to consider here.
# First, the prcomp() function can only deal with numeric columns, so we need to remove
# all non-numeric columns from the data. This is straightforward using the
# where(is.numeric) tidyselect construct.
# Second, we normally want to scale the data values to unit variance before PCA.
# We do so by using the argument scale = TRUE in prcomp().

EdaPca<- DataTrain|>
  select(where(is.numeric))|>
  prcomp(scale=TRUE)

EdaPca %>%
  augment(DataTrain)|> # add original dataset back in
  ggplot(aes(.fittedPC1, .fittedPC2, color = diagnosis)) + 
  geom_point(size = 1.5) +
  scale_color_manual(values = c(M = "#D55E00", B = "#0072B2")) +
  theme_half_open(12) +
  background_grid()

# It is useful to look at the variance explained by each principal component.
# We can extract this information using the tidy() function from broom, now by setting
# the matrix argument to matrix = "eigenvalues". I'm using both a table and a couple
# of plots.

EdaPca |>
  tidy(matrix = "eigenvalues")|>
  kable()

EdaPca %>%
  tidy(matrix = "eigenvalues")|>
  ggplot(aes(PC, percent))+
  geom_col(fill = "#56B4E9", alpha = 0.8)+
  scale_x_continuous(breaks = 1:9)+
  scale_y_continuous(labels = scales::percent_format(),
                     expand = expansion(mult = c(0, 0.01)))+
  theme_minimal_hgrid(12)

EdaPca %>%
  tidy(matrix = "eigenvalues")|>
  ggplot(aes(PC, cumulative))+
  geom_col(fill = "#56B4E9", alpha = 0.8)+
  scale_x_continuous(breaks = 1:9)+
  scale_y_continuous(labels = scales::percent_format(),
                     expand = expansion(mult = c(0, 0.01)))+
  geom_hline(aes(yintercept = 0.80),
             color = "red",
             linetype = 1,
             linewidth = 1)+
  theme_minimal_hgrid(12)


# 2.4. Conclusions

# 1. From univariate analysis we have seen that some predictors show high skewness
# and kurtosis. A transformation to handle skewness is appropriate (Yeo-Johnson).
# 2.Outliers do exist. I assume every observation is fair and correct.
# 3. Quite a lot of predictors are correlated between themselves.
# It seems useful preprocess data in order to reduce multicollinearity.
# 4. PCA shows the relevance of 4 to 5 PC. Since I'm afraid of losing readibility of final
# result, I prefer, in this moment, not to consider PCA in preprocessing phase.


# 3. TRAINING AND TESTING THE CLASSIFICATION MODELS WITH WORKFLOW_SETS PACKAGE

# In the context of a classification project, explorative data analysis has highlighted
# the need for a transformation to reduce skewness and a feature selection aimed to
# drop some correlated predictors.
# Since my aim here is to test and experiment the use of workflow_sets package,
# which can handle multiple recipe and models, I'm going to prepare the following
# recipes:
# 1. No preprocessing step a part from target variable definition.
# 2. Target variable definition + step_corr.
# 3. Target variable definition + step_corr + step_YeoJohnson.
# 4. Target variable definition + step_corr + step_YeoJohnson + step_norm.

# As far as models are concerned, I will pick:
# 1. Decision tree
# 2. Random Forest
# 3. XG boost
# 4. knn
# 5. SVM

# For all of them, I want to optimize one or more hyperparameters.

# In order to select properly the recipe-model combination, I want to use the metric
# of sensitivity defined as True Positive/(True Positive + False Negative).
# Due to the specific nature of data (diasease prediction), the most important
# performance seems to be to correctly detect any effective positive case.
# Sensitivity tells how often the classifier predicts YES (malignant in this case)
# when it is actually YES.


# 3.1. Hyperparameter Tuning and metrics customization

set.seed(123, sample.kind = "Rounding")

WfSetCvFolds <- vfold_cv(DataTrain, v = 5)

custom_metrics <- metric_set(accuracy,
                             bal_accuracy,
                             specificity,
                             sensitivity,
                             f_meas)

# 3.2. Preprocessing recipes

WfSetRecipe1 <-
  recipe(diagnosis ~ ., data = DataTrain) 

WfSetRecipe2 <-
  recipe(diagnosis ~ ., data = DataTrain)|>
  step_corr(all_numeric_predictors())

WfSetRecipe3 <-
  recipe(diagnosis ~ ., data = DataTrain)|>
  step_corr(all_numeric_predictors())|>
  step_YeoJohnson(all_numeric_predictors())

WfSetRecipe4 <-
  recipe(diagnosis ~ ., data = DataTrain)|>
  step_corr(all_numeric_predictors())|>
  step_YeoJohnson(all_numeric_predictors())|>
  step_normalize(all_numeric_predictors())
  

# 3.3. Model Specifications

WfSetModDt <-
    decision_tree(tree_depth = tune(),
                  cost_complexity = tune())|>
    set_engine("rpart")|>
    set_mode("classification")

WfSetModRf <-
  rand_forest(mtry = tune(),
              trees = tune(),
              min_n = tune())|>
  set_engine("ranger")|>
  set_mode("classification")

WfSetModXgb <-
  boost_tree(tree_depth = tune(),
             trees = tune())|>
  set_engine("xgboost")|>
  set_mode("classification")

WfSetModKnn <-
  nearest_neighbor(weight_func = tune(),
                   dist_power = tune(),
                   neighbors = tune())|>
  set_engine("kknn")|>
  set_mode("classification")

WfSetModSvm <-
  svm_linear(cost = tune())|>
  set_engine("kernlab")|>
  set_mode("classification")


# 3.4. Workflow Sets

WfSetWorkflows <- workflow_set(
  preproc = list(recPlain = WfSetRecipe1,
                 recCorr = WfSetRecipe2,
                 recYJ = WfSetRecipe3,
                 recNorm = WfSetRecipe4),
  models = list(Dt = WfSetModDt,
                Rf = WfSetModRf,
                Xgb = WfSetModXgb,
                Knn = WfSetModKnn,
                Svm = WfSetModSvm))

# 3.5. Tuning

WfSetGridCtrl<- control_grid(
  save_pred = TRUE,
  parallel_over = "resamples",
  save_workflow = TRUE)

WfSetGridResults <-
  WfSetWorkflows %>%
  workflow_map(
    seed = 1503,
    resamples = WfSetCvFolds,
    grid = 5,
    metrics = custom_metrics, 
    control = WfSetGridCtrl)

WfSetRankResults <- WfSetGridResults|>
  rank_results(rank_metric = "sensitivity",
               select_best = FALSE)|>
  filter(.metric %in% c("accuracy", "bal_accuracy", "specificity", "sensitivity", "f_meas"))|>
  select(wflow_id, .metric, mean, rank)|>
  group_by(wflow_id, .metric)|>
  summarize(CvAvgScore = mean(mean))|>
  filter(.metric == "sensitivity")|>
  arrange(desc(CvAvgScore))

# The metric values are calculated from the cross-validation folds created from the
# training data (WfSetCvFolds).
# This means these results represent the model's performance on the training data,
# using the cross-validation process to estimate how well the model will generalize to
# new data.

# Let's see metric ranking.

kable(WfSetRankResults)

WfSetRankResults|>
  ggplot(aes(x = fct_reorder(wflow_id, CvAvgScore), y = CvAvgScore))+
  geom_col()+
  coord_flip()

# There are four recipe-model combinations that have produced the same highest result
# in term of sensitivity (the choosen metric): XgBoost model has led to the same
# sensitivity regardless the used preprocessing recipe.
# It seems that what has been done to the dataset before the training has not affected
# the final result.

# Generally speaking and with some slight approssimation, we can see that as far as tree
# models are concerned, there is little if any impact of preprocessing on final results.
# Things changes for knn and SVM

# Once I can choose the best performing model (after training), I want to test it.
# As said, there are four ex aequo workflows: I'll pick recCorr_Xgb combination
# (which has the most complete recipe).


# 4. BEST MODEL FINAL FIT AND TEST

WfSetBestResult1 <- 
  WfSetGridResults %>% 
  extract_workflow_set_result("recCorr_Xgb") %>%
  select_best(metric = "sensitivity")

WfSetBestResult1|>
  kable()

WfSetTestResults1 <- 
  WfSetGridResults %>% 
  extract_workflow("recCorr_Xgb") %>% 
  finalize_workflow(WfSetBestResult1) %>% 
  last_fit(split = DataSplit,
           metrics = custom_metrics)  # Specify the custom metrics here


collect_metrics(WfSetTestResults1)|>
  kable()

WfSetTestPredictions <- WfSetTestResults1 %>%
  collect_predictions()

WfSetConfMatrix <- WfSetTestPredictions|>
  conf_mat(truth = diagnosis,
           estimate = .pred_class,
           dnn=c("Prediction","Truth"),
           case_weights=NULL)

# print(WfSetConfMatrix)

# 5. CONCLUSION AND LESSONS LEARNED

# 1. Workflow set package sis a very powerful tool that allows to increase speed and
# accuracy when a certain amount of recipes and models are involved in a ML project.
# 2. An explorative analysis is (as always) useful to optimize the dataset in order to
# properly preprocess dataset and thus realize proper training and testing phases.
# 3. It seems also useful to tune the hyperparameters. To do that, I've followed a
# cross validation approach and I let workflow_set to define hyperparameters' value,
# by defining grid lenght, thus avoiding manually defined value ranges which could
# potentially introduce any kind of bias.
# 4. In the case of this project (breast cancer), the workflow set approach, has led
# to choose and XG boost model whose performance has been the same regardless each
# of the four preprocessing recipes that have been used.
# 5. In this context, the sensitivity (the metric I've decided to primarily use to
# rank workflows prediction attitude) performance on the test set has been equal
# to 0.86 against the 0.97 obtained during training session.
# It seems quite a bit" large change in performance that could be read as an
# "overfitting hint".


              