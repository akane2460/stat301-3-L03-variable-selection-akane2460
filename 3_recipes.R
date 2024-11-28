# L03 Variable Selection ----
# Setup preprocessing/recipes/feature engineering

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(here)

# handle common conflicts
tidymodels_prefer()

# load training data & variable selection results ----
load(here("data-splitting/rideshare_training.rda"))
load(here("results/var_select_fit_lasso.rda"))
load(here("results/var_keep_rf.rda"))

###############################################################################
# Recipe with variables selected by lasso regression
###############################################################################

# predict var info ----
var_select_lasso <- var_select_fit_lasso |> tidy()

# getting important variables
numeric_vars <- rideshare_training |> 
  select(where(is.numeric)) |> 
  colnames()

factor_vars <- rideshare_training |> 
  select(where(is.factor)) |> 
  colnames()

# important variables
imp_vars <- var_select_lasso |> 
  filter(estimate != 0) |> 
  pull(term)

# getting numeric
imp_numeric <- imp_vars[imp_vars %in% numeric_vars]

# factor tricky because dummy renaming
num_true <- map(
  factor_vars,
  ~ startsWith(imp_vars, prefix = .x) |> 
    sum()
)

# assign raw names from dataset 
names(num_true) <- factor_vars

# at least one factor level = important
imp_factor <- enframe(unlist(num_true)) |> 
  filter(value != 0) |> 
  pull(name)

var_keep <- c(imp_numeric, imp_factor)
var_keep

rideshare_lasso <- rideshare_training |> 
  select(all_of(var_keep), price_log10)

# recipe with lasso variable selection
recipe_lasso <- recipe(price_log10 ~ ., data = rideshare_training) |> 
  step_select(any_of( !!var_keep), price_log10) |> 
  update_role(price, new_role = "original_scale") |> 
  step_dummy(all_nominal_predictors()) |> 
  step_nzv(all_predictors()) |> 
  step_normalize(all_predictors())

# check recipe
recipe_lasso |> 
  prep() |> 
  bake(new_data = NULL) |> 
  glimpse()

# save lasso recipe
save(
  recipe_lasso,
  file = here("recipes/recipe_lasso_rec1.rda")
)

###############################################################################
# Recipe with variables selected by random forest (variable importance)
###############################################################################
recipe_rf <- recipe(price_log10 ~ ., rideshare_training) |> 
  step_select(all_of( !!var_keep_rf), price_log10) |> 
  step_zv(all_predictors()) |> 
  step_dummy(all_nominal_predictors(), one_hot = TRUE) |> 
  step_normalize(all_numeric_predictors())

# check recipe
recipe_rf |> 
  prep() |> 
  bake(new_data = NULL) |> 
  glimpse()

save(
  recipe_rf,
  file = here("recipes/recipe_rf_rec2.rda")
)
