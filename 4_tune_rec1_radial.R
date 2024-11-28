# L03 Variable Selection ----
# Define and fit rec 1 radial

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

# load resamples ----
load(here("data-splitting/rideshare_folds.rda"))

# load preprocessing/recipe ----
load(here("recipes/recipe_lasso_rec1.rda"))

# model specifications ----
svm_radial_model <- svm_rbf(
  mode = "regression", 
  cost = tune(),
  rbf_sigma = tune()
) |>
  set_engine("kernlab")

# define workflows ----
svm_radial_wflow <- workflow() |> 
  add_model(svm_radial_model) |> 
  add_recipe(recipe_lasso)

# hyperparameter tuning values ----
svm_radial_params <- hardhat::extract_parameter_set_dials(svm_radial_model)

svm_radial_grid <- grid_latin_hypercube(svm_radial_params, size = 50)

# fit workflow/model ----
tic("SVM RADIAL: REC 1") # start clock

# tuning code in here
tune_svm_radial_rec1 <- svm_radial_wflow |> 
  tune_grid(
    resamples = rideshare_folds,
    grid = svm_radial_grid,
    control = control_grid(save_workflow = TRUE),
    metrics = metric_set(rmse)
  )

toc(log = TRUE) # stop clock

# Extract runtime info
time_log <- tic.log(format = FALSE)

tictoc_svm_radial_rec1 <- tibble(
  model = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  runtime = end_time - start_time
)

# write out results (fitted/trained workflows & runtime info) ----
save(tune_svm_radial_rec1, tictoc_svm_radial_rec1, 
     file = here("results/tune_svm_radial_rec1.rda"))

