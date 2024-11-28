# L03 Variable Selection ----
# Variable selection using lasso regression

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
lasso_folds <- 
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
lasso_spec <-
  linear_reg(
    mixture = 1,
    penalty = tune()
  ) |> 
  set_mode("regression") |> 
  set_engine("glmnet")

# define workflows ----
lasso_wflow <-
  workflow() |> 
  add_model(lasso_spec) |> 
  add_recipe(recipe_basic)

# hyperparameter tuning values ----
hardhat::extract_parameter_set_dials(lasso_spec)

lasso_params <- hardhat::extract_parameter_set_dials(lasso_spec) |> 
  update(
    penalty = penalty(c(-3, 0))
  )

# build tuning grid
lasso_grid <- grid_regular(lasso_params, levels = 5)

# fit workflow/model ----
lasso_tuned <- 
  lasso_wflow |> 
  tune_grid(
    resamples = lasso_folds, 
    grid = lasso_grid,
    metrics = metric_set(rmse),
    control = control_grid(save_workflow = TRUE)
  )

# extract best model (optimal tuning parameters)
optimal_wflow <- 
  extract_workflow(lasso_tuned) |> 
  finalize_workflow(select_best(lasso_tuned, metric = "rmse"))

# fit best model/results
var_select_fit_lasso <- fit(optimal_wflow, rideshare_training)

# look at results
var_select_lasso <- var_select_fit_lasso |>  tidy()

var_select_lasso |> filter(estimate != 0) |> select(term) |> knitr::kable()


# write out variable selection results ----
save(
  var_select_fit_lasso, 
  file = here("results/var_select_fit_lasso.rda")
)
