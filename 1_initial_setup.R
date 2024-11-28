# L03 Variable Selection ----
# Initial data checks & data splitting

# Load package(s)
library(tidymodels)
library(tidyverse)
library(here)

# handle common conflicts
tidymodels_prefer()

## load data
load(here("data/rideshare.rda"))

rideshare <- rideshare |>
  janitor::clean_names() |>
  filter(!is.na(price)) |>
  mutate(across(where(is.character), as.factor))

# Data quality check ----

# check missingness
naniar::miss_var_summary(rideshare) |>
  filter(n_miss > 0)

# check target variable
ggplot(rideshare, aes(price)) +
  geom_density()

ggplot(rideshare, aes(price)) +
  geom_density() +
  scale_x_log10()

# transform target variable
rideshare <- rideshare |>
  mutate(price_log10 = log10(price))

# downsample 0.5% of the data
set.seed(1234)
rideshare_downsample <- rideshare |>
  initial_split(prop = 0.005, strata = price)

# extract downsampled data
rideshare_small <- training(rideshare_downsample)
# extract remaining data for eda
rideshare_eda <- testing(rideshare_downsample)

# Initial split & resamples ----
set.seed(3013)
rideshare_split <- initial_split(rideshare_small, prop = 0.8, strata = price)

rideshare_training <- training(rideshare_split)
rideshare_testing <- testing(rideshare_split)

# resamples: v-fold
rideshare_folds <- rideshare_training |>
  vfold_cv(v = 5, repeats = 3, strata = price)

# save datasets
save(
  rideshare_training,
  file = here("data-splitting/rideshare_training.rda")
  )

save(
  rideshare_testing,
  file = here("data-splitting/rideshare_testing.rda")
  )

save(
  rideshare_folds,
  file = here("data-splitting/rideshare_folds.rda")
  )
