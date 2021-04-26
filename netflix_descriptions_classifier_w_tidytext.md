Untitled
================
J Moggridge
25/04/2021

-   [Netflix description classifier](#netflix-description-classifier)

## Netflix description classifier

``` r
library(tidyverse)
library(tidymodels)
library(lubridate)
library(tidytext)
theme_set(theme_minimal())
```

``` r
netflix_titles <- read_csv("https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2021/2021-04-20/netflix_titles.csv")

glimpse(netflix_titles)
```

    ## Rows: 7,787
    ## Columns: 12
    ## $ show_id      <chr> "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s1…
    ## $ type         <chr> "TV Show", "Movie", "Movie", "Movie", "Movie", "TV Show",…
    ## $ title        <chr> "3%", "7:19", "23:59", "9", "21", "46", "122", "187", "70…
    ## $ director     <chr> NA, "Jorge Michel Grau", "Gilbert Chan", "Shane Acker", "…
    ## $ cast         <chr> "João Miguel, Bianca Comparato, Michel Gomes, Rodolfo Val…
    ## $ country      <chr> "Brazil", "Mexico", "Singapore", "United States", "United…
    ## $ date_added   <chr> "August 14, 2020", "December 23, 2016", "December 20, 201…
    ## $ release_year <dbl> 2020, 2016, 2011, 2009, 2008, 2016, 2019, 1997, 2019, 200…
    ## $ rating       <chr> "TV-MA", "TV-MA", "R", "PG-13", "PG-13", "TV-MA", "TV-MA"…
    ## $ duration     <chr> "4 Seasons", "93 min", "78 min", "80 min", "123 min", "1 …
    ## $ listed_in    <chr> "International TV Shows, TV Dramas, TV Sci-Fi & Fantasy",…
    ## $ description  <chr> "In a future where the elite inhabit an island paradise f…

``` r
netflix_titles <- netflix_titles %>% 
  mutate(date_added = mdy(date_added))

netflix_titles %>% 
  slice(1:10) %>% 
  pull(description)
```

    ##  [1] "In a future where the elite inhabit an island paradise far from the crowded slums, you get one chance to join the 3% saved from squalor."                                                                                                       
    ##  [2] "After a devastating earthquake hits Mexico City, trapped survivors from all walks of life wait to be rescued while trying desperately to stay alive."                                                                                           
    ##  [3] "When an army recruit is found dead, his fellow soldiers are forced to confront a terrifying secret that's haunting their jungle island training camp."                                                                                          
    ##  [4] "In a postapocalyptic world, rag-doll robots hide in fear from dangerous machines out to exterminate them, until a brave newcomer joins the group."                                                                                              
    ##  [5] "A brilliant group of students become card-counting experts with the intent of swindling millions out of Las Vegas casinos by playing blackjack."                                                                                                
    ##  [6] "A genetics professor experiments with a treatment for his comatose sister that blends medical and shamanic cures, but unlocks a shocking side effect."                                                                                          
    ##  [7] "After an awful accident, a couple admitted to a grisly hospital are separated and must find each other to escape — before death finds them."                                                                                                    
    ##  [8] "After one of his high school students attacks him, dedicated teacher Trevor Garfield grows weary of the gang warfare in the New York City school system and moves to California to teach there, thinking it must be a less hostile environment."
    ##  [9] "When a doctor goes missing, his psychiatrist wife treats the bizarre medical condition of a psychic patient, who knows much more than he's leading on."                                                                                         
    ## [10] "An architect and his wife move into a castle that is slated to become a luxury hotel. But something inside is determined to stop the renovation."

``` r
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
```

    ## Joining, by = "word"

![](netflix_descriptions_classifier_w_tidytext_files/figure-gfm/unnamed-chunk-3-1.png)<!-- -->

``` r
# train a machine learning model that learns to classify movies and tv shows based on the words in the description
```
