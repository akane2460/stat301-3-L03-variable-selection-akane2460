# L03 Variable Selection ----
# Model selection/comparison & analysis

# Load package(s) & set seed ----
library(tidymodels)
library(tidyverse)
library(vip)

# Handle conflicts
tidymodels_prefer()

# load model results
load(here("results/tune_svm_poly_rec1.rda"))
load(here("results/tune_svm_poly_rec2.rda"))
load(here("results/tune_svm_radial_rec2.rda"))
load(here("results/tune_svm_radial_rec1.rda"))

runtimes_models <- bind_rows(tictoc_svm_poly_rec1, tictoc_svm_poly_rec2,
                             tictoc_svm_radial_rec1, tictoc_svm_radial_rec2) |>
  select(runtime)

runtimes_models <- runtimes_models |>
  mutate(model = c("svm_poly_lasso", "svm_poly_rf",
                   "svm_radial_lasso", "svm_radial_rf"))

model_results <- as_workflow_set(
  svm_poly_lasso = tune_svm_poly_rec1,
  svm_poly_rf = tune_svm_poly_rec2,
  svm_radial_lasso = tune_svm_radial_rec1,
  svm_radial_rf = tune_svm_radial_rec2
)

model_results_rmse <- model_results |>
  collect_metrics() |>
  filter(.metric == "rmse") |>
  slice_min(mean, by = wflow_id) |> 
  select(wflow_id, mean, std_err)


results_runtimes <- bind_cols(model_results_rmse, runtimes_models) |> 
  select(-wflow_id) |> 
  mutate(recipe = c("lasso", "rf", "lasso", "rf")) |> 
  relocate(model, .before = 1) |> 
  relocate(recipe, .before = 2) |> 
  knitr::kable()

