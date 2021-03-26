
## Tree-based Classification of Diabetes Data

## Consider the Diabetes dataset from Lecture 12.
## Train a classification tree for the categorical response variable
## It is a good idea to consider some derived features, such as, interactions!

## "Train" KNN and CART (Classification and Regression Tree) models and compare their predictive performance
## Are there any key influential variables? Are they same as the ones
## flagged by the Lasso fit earlier?

library(tidyverse)
library(tidymodels)
library(vip)
library(glue)
library(rcartocolor)
library(rattle)

diabetes <- read_csv("./Assignments_and_midterm/6970_midterm/diabetes.csv")

### tidying up data and creating predictors -----

# formula: all vars (.) and their interactons (^2)
f0 <- lm(positive ~ . * ., data = diabetes)

# take original cols and interactions (remove intercept)
diabetes <- as_tibble(model.matrix(f0)[, -1]) %>% 
  # add quadratic terms
  bind_cols(
    diabetes %>%
      select(preg:age) %>%
      mutate(across(everything(), function(x) x^2)) %>%
      rename_with( ~ str_c(., '^2'), .cols = everything())
  ) %>%
  # response variable needs to be a factor for classification
  mutate(positive = factor(diabetes$positive)) %>%
  relocate(positive)
rm(f0)

### data splitting, creating folds, preprocessing
# data splitting: 75% train, 25% test
set.seed(1545)
diabetes_split <- initial_split(diabetes, prop = 0.75)
train <- training(diabetes_split)
test <- testing(diabetes_split)

# create 10 folds for CV
ten_fold_cv <- vfold_cv(train, v = 10)

# recipe: sets formula & scaling as a preprocessing step
rec <- recipe(positive ~ ., data = train) %>%
  step_normalize(all_predictors()) %>% 
  prep()


# Decision tree models ----

# specify decision tree model with parameters for tuning
tree_spec <- decision_tree(cost_complexity = tune(),
                           tree_depth = tune()) %>% 
  set_engine("rpart") %>% 
  set_mode("classification")

# set up grid of tunings to search using grid_* functions
# functions from dials:: to automatically select values
# levels sets the length for each parameter
tree_grid <- grid_regular(cost_complexity(),
                          tree_depth(),
                          levels = c(5,10))

# check that it's the right size before accidentally training like 25000 models
tree_grid

# Combine models and data into workflow for tuning/fitting/predicting.
# workflow comes in handy later when we want to retrain the best model
# and predict the test data
tree_workflow <- workflow() %>% 
  add_recipe(rec) %>%
  add_model(tree_spec)

# Now we do the 10-fold CV to estimate model performance
tree_res <- 
  tree_workflow %>% 
  tune_grid(resamples = ten_fold_cv,
            grid = tree_grid)

# pick the model with best parameters based on chosen metric
best_tree <- tree_res %>% select_best("roc_auc")

# update the workflow by setting best model
final_tree_wf <- tree_workflow %>% finalize_workflow(best_tree)

# fit final_tree model to full train set
final_tree <- final_tree_wf %>% fit(data = train) 
# can use this fitted tree to find out variable importance

# last_fit() fits the final model on full training set and evaluates  on the testing data.
final_tree_fit <- final_tree_wf %>% last_fit(diabetes_split) 



# K-NN models -----

## k nearest neighbors model
knn_spec <- nearest_neighbor(neighbors = tune()) %>% 
  set_engine('kknn') %>% 
  set_mode("classification")

# tune grid 
knn_grid <- grid_regular(neighbors(range = c(1, 200)),
                         levels = 50)
knn_grid

# workflow
knn_wflow <- workflow() %>% 
  add_recipe(rec) %>% 
  add_model(knn_spec)
knn_wflow

# apply 10-fold cv to workflow for model tuning
knn_res <- knn_wflow %>% 
  tune_grid(resamples = ten_fold_cv, grid = knn_grid)

# update the workflow by selecting the best model by some metric
best_knn <- knn_res %>% select_best("roc_auc")
final_knn_wf <- knn_wflow %>% finalize_workflow(best_knn)

# final_tree object has the selected fitted model 
final_knn <- final_knn_wf %>% fit(data = train) 

# last_fit() fits the final model on full training set and evaluates  on the testing data.
final_knn_fit <- final_knn_wf %>% last_fit(diabetes_split) 


## Logistic regression ----

# penalty is 'lambda', mixture is 'alpha'
glmnet_spec <- 
  logistic_reg(penalty = tune(), mixture = 0) %>% 
  set_engine("glmnet") %>% 
  set_mode("classification")

# setup a new workflow
glmnet_wflow <- 
  workflow() %>% 
  add_recipe(rec) %>% 
  add_model(glmnet_spec)

# create grid of tuning params
glmnet_grid <- grid_regular(penalty(), levels = 100)
glmnet_grid

# do ten-fold cv for all models in grid
glmnet_res <- glmnet_wflow %>%
  tune_grid(resamples = ten_fold_cv,
            grid = glmnet_grid)

# pick best model and finalize workflow
best_glmnet <- glmnet_res %>% select_best(metric = "roc_auc")
final_glmnet_wflow <- glmnet_wflow %>% finalize_workflow(best_glmnet)

# retrain the best model
final_glmnet <- final_glmnet_wflow %>% fit(data = train)

# fit, predict, collect metrics for best model (uses split obj)
final_glmnet_fit <- final_glmnet_wflow %>% last_fit(diabetes_split)


## Plot the results ----

# visualize the CV results for decision trees
tree_cv_plot <- tree_res %>% 
  collect_metrics() %>% 
  mutate(cost_complexity = as_factor(cost_complexity)) %>% 
  filter(.metric == "roc_auc") %>% 
  ggplot(aes(x = tree_depth, 
             y = mean,
             color = cost_complexity)
  ) +
  geom_path(alpha = 0.5) +
  geom_jitter(width = 0.05, height = 0.0005) +
  rcartocolor::scale_color_carto_d(palette = "Vivid") +
  theme_bw() +
  labs(title = "Decision tree tuning", y = "ROC AUC")

# visualize the CV results for knn models
knn_cv_plot <- knn_res %>% 
  collect_metrics() %>% 
  filter(.metric == "roc_auc") %>% 
  ggplot(aes(x = neighbors, y = mean)) +
  geom_path() +
  geom_point() +
  theme_bw() +
  labs(title = "k-nearest neighbors CV results", y = "ROC AUC")

# visualize the CV results for knn models
glm_cv_plot <- glmnet_res %>% 
  collect_metrics() %>% 
  filter(.metric == "roc_auc" & mean >0.80) %>% 
  ggplot(aes(x = penalty, y = mean)) +
  geom_point() + 
  scale_x_log10() +
  theme_bw() +
  labs(title = "Logistic regression CV results")

# plot roc curve for decision tree
tree_curve <- 
  final_tree_fit %>%
  collect_predictions() %>% 
  roc_curve(truth = positive, estimate = .pred_0) %>% 
  autoplot() +
  labs(subtitle = glue("Best decision tree:
                       Cost complexity = {best_tree$cost_complexity}
                       Tree depth = {best_tree$tree_depth}"))
# plot roc curve for knn
knn_curve <- 
  final_knn_fit %>%
  collect_predictions() %>% 
  roc_curve(truth = positive, estimate = .pred_0) %>% 
  autoplot() +
  labs(subtitle = glue("Best knn model
                       k = {best_knn$neighbors}"))
# plot roc curve for glmnet
glm_curve <- 
  final_glmnet_fit %>%
  collect_predictions() %>% 
  roc_curve(truth = positive, estimate = .pred_0) %>% 
  autoplot() +
  labs(subtitle = glue("Best lasso logistic regression model
                       lambda = {round(best_glmnet$penalty, 3)}"))


# vip package to estimate variable importance.
# only works with certain models (not knn)
tree_vip <- final_tree %>% 
  pull_workflow_fit() %>% 
  vip::vip() +
  theme_bw() + 
  labs(subtitle = "Decision tree")

glm_vip <- final_glmnet %>% 
  pull_workflow_fit() %>% 
  vip::vip(geom = 'col') +
  theme_bw() + 
  labs(subtitle = "Lasso logistic regression")

tree_vip + glm_vip

## Nice panels for comparison
# cv results
(knn_cv_plot + glm_cv_plot) / (tree_cv_plot + guide_area())

# test results for selected model
tree_curve + knn_curve + glm_curve

# create a table with the results
bind_rows(
  final_tree_fit %>% 
    collect_metrics() %>% 
    mutate(model = 'tree'),
  final_knn_fit %>%
    collect_metrics() %>% 
    mutate(model = 'knn'),
  final_glmnet_fit %>%
    collect_metrics() %>% 
    mutate(model = 'lasso')
  
  ) %>% 
  select(model, .metric, .estimate) %>% 
  pivot_wider(names_from = .metric, values_from = .estimate) %>% 
  pander::pander(caption = "The final results are in:")


rpart_fit <- final_tree %>% 
  pull_workflow_fit()

rattle::fancyRpartPlot(model = rpart_fit$fit,
                       type = 1, 
                       palettes=c("BuPu", "YlOrRd"),
                       caption = NULL) 
