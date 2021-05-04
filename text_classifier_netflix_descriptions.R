


library(tidyverse)
library(tidymodels)
library(lubridate)
library(tidytext)

theme_set(theme_minimal())

netflix_titles <- read_csv("https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2021/2021-04-20/netflix_titles.csv")

glimpse(netflix_titles)


netflix_titles <- netflix_titles %>% 
  mutate(date_added = mdy(date_added))


netflix_titles %>% 
  slice(1:10) %>% 
  pull(description)


netflix_titles %>% 
  unnest_tokens(word, description) %>% 
  anti_join(get_stopwords()) %>% 
  count(type, word, sort = TRUE) %>% 
  group_by(type) %>% 
  slice_max(n, n = 15) %>% 
  ungroup() %>% 
  mutate(word = reorder_within(word, by = n, within = type)) %>% 
  ggplot(aes(n, word, fill = type)) +
  geom_col(show.legend = F) +
  scale_y_reordered() +
  facet_wrap(~type, scales = 'free_y') +
  labs(x = 'Count', y ='',
       subtitle = 'Netflix descriptions word count by type')

# train a machine learning model that learns to classify movies and tv shows based on the words in the description





