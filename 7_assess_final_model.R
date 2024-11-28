# L03 Variable Selection ----
# Assess final model

# Load package(s) & set seed ----
library(tidymodels)
library(tidyverse)
library(here)

# Handle conflicts
tidymodels_prefer()

# load data----
load(here("results/final_fit.rda"))
load(here("data-splitting/rideshare_testing.rda"))

# predictions---
final_predict <- rideshare_testing |>  
  bind_cols(predict(final_fit, rideshare_testing)) |>
  select(price_log10, .pred)

# metrics----
ames_metrics <- metric_set(rmse, rsq)

final_fit_metrics <- ames_metrics(final_predict, 
                                  truth = price_log10, 
                                  estimate = .pred)
final_fit_metrics |> 
  knitr::kable()

# plot----
original_scale_plot <- final_predict |> 
  ggplot(aes(x = price_log10, y = .pred)) +
  geom_point(alpha = .4) +
  labs(x = "Price (log 10)", y = "Predicted Price (log 10)", title = "Uber Prices (log 10)")

ggsave(here("results/original_scale_plot.png"), original_scale_plot)
