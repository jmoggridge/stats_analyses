library(tidyverse)
library(ISLR)
library(boot)

loocv_glm_poly <- function(degree){
  cv.glm(glmfit = glm(formula = mpg ~ poly(horsepower, degree),
                      data = Auto), 
         data = Auto, 
         K = nrow(train))
} 

tencv_glm_poly <- function(degree){
  cv.glm(glmfit = glm(formula = mpg ~ poly(horsepower, degree),
                      data = Auto), 
         data = Auto, 
         K = 10)
} 

cv_grid <- tibble(degree = 1:10) %>% 
  rowwise() %>% 
  mutate(loocv = map(degree, loocv_glm_poly),
         tencv = map(degree, tencv_glm_poly)) %>% 
  ungroup() %>% 
  mutate(delta = map(loocv, 3),
         loo_err = map_dbl(delta, 1),
         delta = map(tencv, 3),
         cv_err = map_dbl(delta, 2)
         )
glimpse(cv_grid)

cv_grid %>% 
  pivot_longer(cols = c(loo_err, cv_err)) %>% 
  ggplot(aes(x=degree, y=value, color = name)) +
  geom_point() +
  geom_path()
