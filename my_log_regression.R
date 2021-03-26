
## Idea: Compute all cut-offs on the training set; then,
##       use test set to choose the final model.
## Essentially, the test set in this case becomes a 'validation set'.

## Try it yourself with i) Lasso alone and, then ii) the Elastic Net framework!
###+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

library(glmnet)
library(ggplot2)
library(GGally)
library(pROC)
library(MASS)

setwd("/Users/jasonmoggridge/Documents/BINF_6970_stats/class12")
diab2 <- read.csv("diabetes2.csv")

# formula: all vars (.) and their interactons (^2)
f0 <- lm(positive ~ . ^ 2, data = diab2[, -10])
# remove the intercept
X <- as_tibble(model.matrix(f0)[, -1]) 
X <- X %>% 
  bind_cols(
    X %>% 
      select(preg:age) %>% 
      mutate(across(everything(), function(x) x^2)) %>% 
      rename_with(~str_c(., '^2'), .cols = everything()))
  
glimpse(X)

cvx <- cv.glmnet(X, y,
  nfolds = 10,
  family = "binomial",
  alpha = 1,
  type.measure = "auc"
)
plot(cvx)

## fitted values and test set predictions
prds.train <- predict(cvx,
                      newx = X[-test.index, ],
                      type = "response",
                      s = cvx$lambda.min)[, 1]
prds.test <- predict(cvx,
                     newx = X[test.index, ],
                     type = "response",
                     s = cvx$lambda.min)[, 1]
