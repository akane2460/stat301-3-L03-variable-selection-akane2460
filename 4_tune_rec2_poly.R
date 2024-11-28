# L03 Variable Selection ----
# Define and fit rec 2 poly

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
load(here("recipes/recipe_rf_rec2.rda"))

# model specifications ----
svm_poly_model <- svm_poly(
  mode = "regression",
  cost = tune(),
  degree = tune(),
  scale_factor = tune()
) |>
  set_engine("kernlab")

# define workflows ----
svm_poly_wflow <- workflow() |> 
  add_model(svm_poly_model) |> 
  add_recipe(recipe_rf)

# hyperparameter tuning values ----
svm_poly_params <- hardhat::extract_parameter_set_dials(svm_poly_model)

svm_poly_grid <- grid_latin_hypercube(svm_poly_params, size = 50)

# fit workflow/model ----
tic("SVM POLY: REC 2") # start clock

# tuning code in here
tune_svm_poly_rec2 <- svm_poly_wflow |> 
  tune_grid(
    resamples = rideshare_folds,
    grid = svm_poly_grid,
    control = control_grid(save_workflow = TRUE),
    metrics = metric_set(rmse)
  )

toc(log = TRUE) # stop clock

# Extract runtime info
time_log <- tic.log(format = FALSE)

tictoc_svm_poly_rec2 <- tibble(
  model = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  runtime = end_time - start_time
)

# write out results (fitted/trained workflows & runtime info) ----
save(tune_svm_poly_rec2, tictoc_svm_poly_rec2, 
     file = here("results/tune_svm_poly_rec2.rda"))
