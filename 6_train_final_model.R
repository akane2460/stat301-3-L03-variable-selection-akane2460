# L03 Variable Selection ----
# Train final model

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(here)
library(tictoc)
library(doMC)

# Handle conflicts
tidymodels_prefer()

# parallel processing ----
num_cores <- parallel::detectCores(logical = TRUE)
registerDoMC(cores = num_cores - 1)

# load data
load(here("data-splitting/rideshare_training.rda"))
load(here("results/var_keep_rf.rda"))

# model recipe----
# variables to remove
var_remove <- setdiff(
  names(rideshare_training),
  c(var_keep_rf, "price", "price_log10")
)

# recipe with lasso variable selection
recipe_final <- recipe(price_log10 ~ ., data = rideshare_training) |> 
  step_rm(any_of( !!var_remove )) |> 
  update_role(price, new_role = "original_scale") |> 
  step_dummy(all_nominal_predictors()) |> 
  step_nzv(all_predictors()) |> 
  step_normalize(all_predictors())

# check recipe
recipe_final |> 
  prep() |> 
  bake(new_data = NULL) |> 
  glimpse()

# model spec----
svm_radial_spec <- svm_rbf(
  mode = "regression", 
  cost = 23, # optimal hyperparameter values found in 4_tune_rec2_radial
  rbf_sigma = .000745
) |>
  set_engine("kernlab")

# define workflows ----
svm_radial_wflow <- workflow() |> 
  add_model(svm_radial_spec) |> 
  add_recipe(recipe_final)

# train final model ----
# set seed
set.seed(0208234)

final_fit <- fit(svm_radial_wflow, rideshare_training)

# save final model----
save(final_fit, file = here("results/final_fit.rda"))

