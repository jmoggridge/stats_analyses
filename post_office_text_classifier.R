library(tidyverse)
# devtools::install_github("tidymodels/parsnip")
library(parsnip)
library(textrecipes)
library(themis)
library(tidymodels)


# tidy tues post offices
post_offices <- read_csv("https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2021/2021-04-13/post_offices.csv")

# train a model to learn which state a post office is located in using tokenization.

post_offices %>% 
  count(state, sort = TRUE)

post_offices %>% 
  filter(state == "HI") %>% 
  pull(name)

set.seed(123)

po_split <- post_offices %>% 
  mutate(state = case_when(state == "HI" ~"Hawaii",
                           TRUE ~ "Other")) %>% 
  select(name, state) %>% 
  initial_split(strata = state)
po_split

po_test <- testing(po_split)
po_train <- training(po_split)

set.seed(234)
po_folds <- vfold_cv(po_train, strata = state)
po_folds


# Text preprocessing: subwords tokenization
  
po_recipe <- recipe(state ~ name, data = po_train) %>% 
  # split into tokens
  step_tokenize(name, engine = "tokenizers.bpe",
                training_options = list(vocab_size = 200)) %>% 
  # take best 200 token
  step_tokenfilter(name, max_tokens = 200) %>% 
  step_tf(name) %>% 
  # normalize
  step_normalize(all_predictors()) %>% 
  step_smote(state)

po_recipe %>% prep() %>% bake(new_data = NULL)


# support vector machines classifier
# no class probabilities, only hard class predictions

# svm_linear() model spec isn't working??
svm_spec <- svm_poly(degree = 1) %>% 
  set_mode("classification") %>% 
  set_engine("kernlab")

# finalize workflow
po_wf <- workflow() %>% 
  add_recipe(po_recipe) %>% 
  add_model(svm_spec)
  
# do cross-validation
set.seed(234)
po_rs <- fit_resamples(
  po_wf, 
  po_folds, 
  metrics = metric_set(accuracy, sens, spec)
  )

# check cv results
collect_metrics(po_wf)  

# fit to full training data
final_fit <- last_fit(
  po_wf, 
  po_split, 
  metrics = metric_set(accuracy, sens, spec)
  )

# confusion matrix
collect_metrics(final_fit) %>% 
  conf_mat(state, .pred_class) %>% 
  autoplot(type = 'heatmap')




