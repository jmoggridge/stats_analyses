# A function to perform LOOCV for k-Nearest Neighbors
# knn_loocv(x, y, k): do loocv at each value of k, parse results

library(tidyverse)
library(class)

# leave_one_out() trains & predicts a single obs
leave_one_out <- function(i, x, y, k) {
  class::knn(x[-i, ], x[i, ], cl = y[-i], k = k)
}

# knn_loocv() - do loocv for a given k
# Predicts each observation by 
# accuracy is the sum of correct class / total
knn_loocv <- function(x, y, k) {
  predictions <- map_int(seq(nrow(x)), ~ leave_one_out(.x, x, y, k))
  accuracy <- sum(diag(table(predictions, y))) / length(y)
  return(list(k = k, accuracy = accuracy, predictions = predictions))
}

# knn_loocv() does loocv to tune the k hyperparameter for k-Nearest Neighbors. Input a vector of k values to test.
tune_knn_loocv <- function(x, y, k_vector = NULL) {
  if(is.null(k_vector)) k_vector <- seq(1, 10, 2)
  # do the loocv for each k, rearrange results to tibble
  results <- map(k_vector, ~ knn_loocv(x, y, k=.x)) %>% 
    transpose() %>% as_tibble() %>%
    unnest(cols = c(k, accuracy))

  best_tune <- results %>% 
    filter(accuracy == max(accuracy)) %>% 
    slice_tail(n = 1) %>% 
    pull(k)
  return(list(best_tune = best_tune, results = results))
}

# plotting function for knn_cv object from knn_loocv
plot_knn_cv <- function(tune_loocv) {
  ggplot(tune_loocv$result,
         aes(x = k, y = accuracy)) +
    geom_path(color = 'gray') +
    geom_point(alpha = 0.9) + 
    theme_bw() +
    labs(subtitle = paste('best: k =', tune_loocv$best_tune))
}

###

loocv_result <- knn_loocv(x = iris %>% select(-Species),
          y = iris$Species,
          k = 50)
tune_result <- tune_knn_loocv(x = iris %>% select(-Species),
               y = iris$Species,
               k = 1:100)
plot_knn_cv(tune_result)

library(palmerpenguins)
library(skimr)

# example KNN on penguins data
glimpse(palmerpenguins::penguins)

penguins2 <- palmerpenguins::penguins %>%
  # drop non-measuremnent data; keep species as class labels
  select(-c(island, year)) %>%
  # drop any rows missing data
  filter(across(everything(), ~ !is.na(.x))) %>%
  # standardize everything
  mutate(across(where(is.numeric), ~ (.x - mean(.x)) / sd(.x))) %>%
  mutate(sex = ifelse(sex == 'male', 1, 0))

skimr::skim(penguins2)
X <- penguins2 %>% select(-c(species))
Y <- as.integer(penguins2$species)
cv <- knn_loocv(X, Y, K = seq(1, 50, 2))
plot_knn_cv(cv)


i <- 1
k <- 20
class::knn(train = x[-i, ], test = x[i, ], cl = y[-i], k = k)

# predict each observation by leave one out
predictions <- map_int(seq(nrow(x)), ~ leave_one_out(.x, x, y, k))
# accuracy is the sum of correct classifications / total
accuracy <- sum(diag(table(predictions, y))) / length(y)
accuracy
