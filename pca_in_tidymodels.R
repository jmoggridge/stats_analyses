library(tidyverse)
library(tidymodels)

load(here::here('./data/geneexpression2.rda'))

dat2 <- t(dat)

dat <- 
  dat %>% 
  rownames_to_column() %>% 
  tibble() %>% 
  mutate(cell_type = str_extract(rowname, "_[A-Z]+_"),
         cell_type = as_factor(str_remove_all(cell_type, "_")),
         status = as_factor(str_extract(rowname, "^[A-Z]+"))) %>% 
  relocate(status, cell_type) %>% 
  mutate(cell_type= case_when(
    cell_type == "EFFE" ~"effector",
    cell_type == "NAI" ~"naive",
    cell_type == "MEM" ~"memory"))
dat
table(dat$status, dat$cell_type)
sum(is.na(dat))

dat2 <- dat2 %>% 
  data.frame() %>% 
  rownames_to_column(var = 'gene')

glimpse(dat2)


## corplot but this data has too many variables, 
# library(corrr)
# dat %>%
#   select(-1:3) %>%
#   # cor matrix
#   correlate() %>%
#   # arrange highest at top
#   rearrange() %>%
#   # remove upper triangle
#   shave() %>%
#   rplot()


## Principal component analysis

dat_recipe <- dat %>% 
  recipe(status ~ ., data = .) %>% 
  update_role(rowname, cell_type, new_role = "id") %>% 
  step_center(all_predictors()) %>% 
  step_normalize(all_predictors()) %>% 
  step_pca(all_predictors())



dat_prep <- prep(dat_recipe)

tidy(dat_prep)
tidy(dat_prep, 1)
# tidy pca
summary(tidy(dat_prep,3) %>% filter(component %in% c('PC1', 'PC2', 'PC3')))

library(tidytext)

# contributions to PCs
tidy(dat_prep, 3) %>% 
  filter(component %in% c('PC1', 'PC2', 'PC3')) %>%
  mutate(terms = reorder_within(x = terms, by = abs(value),
                                within = component)) %>% 
  ggplot(aes(x=abs(value), y=terms, fill = value>0)) +
  geom_col(show.legend = F) +
  facet_wrap(component~., scales = 'free_y') +
  scale_y_reordered() +
  labs(y=NULL, fill = 'sign', x = "abs contribution to PC")
 
library(ggrepel)
# plot the projection 
my_pca <- juice(dat_prep)
 
my_pca %>% 
  ggplot(aes(x = PC1, y=PC2, color = cell_type, 
             shape = status)) +
  geom_point(size = 2.5, alpha = 0.9) +
  geom_text_repel(aes(label = rowname), 
                           size = 3, family = 'sans') +
  theme_bw()

my_pca  %>% 
  ggplot(aes(x = PC1, y=PC3, color = cell_type, 
             shape = status)) +
  geom_point(size = 2.5, alpha = 0.9) +
  geom_text_repel(aes(label = rowname), 
                           size = 3, family = 'sans') +
  theme_bw()

my_pca  %>% 
  ggplot(aes(x = PC2, y=PC3, color = cell_type, 
             shape = status)) +
  geom_point(size = 2.5, alpha = 0.9) +
  geom_text_repel(aes(label = rowname), 
                           size = 3, family = 'sans') +
  theme_bw()

my_pca <- my_pca %>% 
  mutate(text = glue::glue(" <b>id: {rowname} </b>
                            cell: {cell_type}
                            status: {status}"),
         pch = as.numeric(status)) 
# my_pca
plotly::plot_ly(data = my_pca,
    type = "scatter3d", mode = "markers",
    x = ~PC1, y=~PC2, z=~PC3,
    color = ~cell_type,
    symbol = ~pch, 
    symbols = c('circle','diamond'),
    opacity = 0.9, 
    hovertext = ~text)

rotation <- dat_prep$steps[[3]]$res$rotation

# scree plot
sdev <- dat_prep$steps[[3]]$res$sdev
sdev <- (sdev^2)*100/sum(sdev^2)
ggplot(tibble(x=1:length(sdev), y = sdev),
       aes(x=x, y=y)) +
  geom_col() +
  labs(subtitle = "Scree plot", 
       y = "percent variance explained",
       x = 'PC'
       ) +
  theme_bw()

# biplot
rotation <- dat_prep$steps[[3]]$res$rotation
rotation %>% 
  data.frame() %>% 
  rownames_to_column() %>% 
  tibble() %>% 
  select(1:4) %>% 
  ggplot(aes(x=PC1, y=PC2)) +
  geom_hline(yintercept = 0, size = 0.05) +
  geom_vline(xintercept = 0, size = 0.05) +
  geom_point(color = 'blue', alpha = 0.5) +
  geom_text_repel(aes(label = rowname),
                  segment.colour = "gray", size = 2.39) +
  labs(subtitle = 'biplot') +
  theme_bw()



dat2_recipe <- dat2 %>% 
  recipe(gene ~., data=.) %>% 
  step_normalize(all_predictors()) %>% 
  step_pca(all_predictors())
dat2_prep <- prep(dat2_recipe)

# scree plot
sdev <- dat2_prep$steps[[2]]$res$sdev
sdev <- (sdev^2)*100/sum(sdev^2)
ggplot(tibble(x=1:length(sdev), y = sdev),
       aes(x=x, y=y)) +
  geom_col() +
  labs(subtitle = "Scree plot", 
       y = "percent variance explained",
       x = 'PC'
  ) +
  theme_bw()


# biplot
rotation <- dat2_prep$steps[[2]]$res$rotation
rotation %>% 
  data.frame() %>% 
  rownames_to_column() %>% 
  mutate(cell_type = str_extract(rowname, "_[A-Z]+_"),
         cell_type = as_factor(str_remove_all(cell_type, "_")),
         status = as_factor(str_extract(rowname, "^[A-Z]+"))) %>% 
  relocate(status, cell_type) %>% 
  mutate(cell_type= case_when(
    cell_type == "EFFE" ~"effector",
    cell_type == "NAI" ~"naive",
    cell_type == "MEM" ~"memory")) %>% 
  # select() %>% 
  ggplot(aes(x=PC1, y=PC2, color = cell_type, shape = status)) +
  # geom_hline(yintercept = 0, size = 0.15) +
  # geom_vline(xintercept = 0, size = 0.15) +
  geom_point(alpha = 0.5) +
  geom_text_repel(aes(label = rowname), alpha = 1,
                  segment.colour = "gray", size = 2.39) +
  labs(subtitle = 'biplot') +
  theme_bw()
