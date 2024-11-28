# L03 Variable Selection ----
# Variable selection using random forest

# load package(s) ----
library(tidyverse)
library(tidymodels)
library(here)
library(doMC)

# handle common conflicts
tidymodels_prefer()

# parallel processing ----
num_cores <- parallel::detectCores(logical = TRUE)
registerDoMC(cores = num_cores - 1)

# create resamples/folds ----
load(here("data-splitting/rideshare_training.rda"))

set.seed(987)
rf_folds <- 
  rideshare_training |> 
  vfold_cv(v = 5, repeats = 1, strata = price)


# basic recipe ----
recipe_basic <- recipe(price_log10 ~ ., data = rideshare_training) |> 
  step_rm(price, id, datetime, timezone) |> 
  step_dummy(all_nominal_predictors()) |> 
  step_nzv(all_predictors()) |>
  step_normalize(all_predictors())

# checking the recipe 
recipe_basic |> 
  prep() |> 
  bake(new_data = NULL)


# model specifications ----
rf_spec <-
  rand_forest(
    trees = 1000,
    mtry = tune(),
    min_n = tune()
  ) |> 
  set_engine("ranger", importance = "impurity") |> 
  set_mode("regression")

# define workflows ----
rf_wflow <-
  workflow() |> 
  add_model(rf_spec) |> 
  add_recipe(recipe_basic)

# hyperparameter tuning values ----
hardhat::extract_parameter_set_dials(rf_spec)

rf_params <- hardhat::extract_parameter_set_dials(rf_spec) |> 
  update(mtry = mtry(range = c(1, 10)), 
         min_n = min_n(range = c(5, 20)))

# build tuning grid
rf_grid <- grid_regular(rf_params, levels = 4)

# fit workflow/model ----
rf_tuned <- 
  rf_wflow |> 
  tune_grid(
    resamples = rf_folds, 
    grid = rf_grid,
    metrics = metric_set(rmse),
    control = control_grid(save_workflow = TRUE)
  )

# extract best model (optimal tuning parameters)
optimal_wflow <- 
  extract_workflow(rf_tuned) |> 
  finalize_workflow(select_best(rf_tuned, metric = "rmse"))

# fit best model/results
var_select_fit_rf <- fit(optimal_wflow, rideshare_training)

# look at results
var_select_rf <- var_select_fit_rf |> 
  extract_fit_parsnip() |> 
  vip::vi()

numeric_vars_rf <- rideshare_training |> select(where(is.numeric)) |> colnames()

imp_var_rf <- var_select_rf |> 
  slice_max(Importance, n = 20) |> 
  pull(Variable)

imp_numeric_rf <- imp_var_rf[imp_var_rf %in% numeric_vars_rf]

num_true_rf <- map(factor_vars,
                   ~ startsWith(imp_var_rf, prefix = .x) |> 
                     sum())

names(num_true_rf) <- factor_vars

imp_factor_rf <- enframe(unlist(num_true_rf)) |> 
  filter(value != 0) |> 
  pull(name)

var_keep_rf <- c(imp_numeric_rf, imp_factor_rf)

# write out variable selection results ----
save(var_keep_rf, file = here("results/var_keep_rf.rda"))
