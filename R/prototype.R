# Create prototype

library(tidyverse)
library(googledrive)
library(tidymodels)
library(censored)
library(embed)
# library(finetune)
library(vip)


doParallel::registerDoParallel()
theme_set(theme_light())


# Download and unzip file ------------------------------------------------------
# Only run this the first time
dir.create("data/")
googledrive::drive_download("https://drive.google.com/file/d/17-k5lBWsBudijdfAJ6oJxwVXMd_EzVlL/view?usp=drive_link", path = "data/dr2023.zip", overwrite = FALSE)

# Password to unzip: data23
system(paste("unzip -P", "data23", "data/dr2023.zip", "-d data"))

# Read training file -----------------------------------------------------------
# See data description here: https://eval.dataracing.hu/web/challenges/challenge-page/5/overview

loan_raw <- read_csv("data/training_data.csv")

# Process data ------------------------------------------------------------

loan <-
    loan_raw |> 
    # Get less data to try out models (remove this later!)
    slice_sample(prop = .05) |> 
    mutate(TARGET_EVENT_DAY = if_else(TARGET_EVENT == "E", NA_real_, TARGET_EVENT_DAY),
           TARGET_EVENT = if_else(TARGET_EVENT == "K", 1L, 0L)) |> 
    # Survival regression needs this new variable
    mutate(surv = Surv(TARGET_EVENT_DAY, TARGET_EVENT), .keep = "unused") |>
    force()


# EDA ---------------------------------------------------------------------
glimpse(loan)

count(loan, TARGET_EVENT)

n_distinct(loan_raw$BORROWER_ID)
n_distinct(loan_raw$CONTRACT_ID)


# Modeling ----------------------------------------------------------------
# Partition
set.seed(123)

loan_split <- initial_split(loan)
loan_training <- training(loan_split)
loan_testing <- testing(loan_split)

loan_folds <- vfold_cv(loan_training, v = 5)

# Feature engineering
loan_rec <- 
    recipe(surv ~ CONTRACT_INCOME + CONTRACT_INSTALMENT_AMOUNT + CONTRACT_INTEREST_RATE + CONTRACT_MARKET_VALUE, data = loan_training) |> 
    step_string2factor(all_string_predictors()) |> 
    # update_role(BORROWER_ID, new_role = "id") |>
    step_embed(BORROWER_ID, outcome = surv) |>
    force()
    
# Model spec
bt_spec <- 
    boost_tree(
            trees = tune(),
            min_n = tune(),
            mtry = tune(),
            learn_rate = 0.01) |> 
    set_mode("censored regression") |> 
    set_engine("mboost")

ph_spec <- 
    proportional_hazards(penalty = .5, mixture = 1) |> 
    set_engine("glmnet") |> 
    set_mode("censored regression") 

# Workflow
loan_wf <- 
    workflow() |>
    add_recipe(loan_rec) |> 
    add_model(ph_spec)

# Fit model
set.seed(345)

ph_plain <- 
    fit(loan_wf, loan_training)

extract_fit_parsnip(ph_plain)

ph_plain |> 
roc_auc_survival() |> 
    roc_auc()

predict(ph_plain, loan_training, type = "linear_pred") |> 
    pull() |> 
    qplot()

predict(
    ph_plain, 
    loan_training, 
    type = "survival",
    eval_time = c(100, 500, 1000)) |> 
    slice(1) |> 
    unnest(col = .pred)

augment(ph_plain, loan_training, type = "linear_pred", eval_time = seq(0, 1500, 50))

plain_fit <- fit_resamples(loan_wf, loan_folds, eval_time = seq(0, 1500, 50))

ph_fit <- 
    ph_spec |> 
    tune_grid(surv ~ ., 
              grid = 5,
              # eval_time = 20,
              resamples = loan_folds)

bt_fit <- 
    bt_spec |> 
    tune_grid(surv ~ ., 
              grid = 5,
              eval_time = 20,
              # metrics = metric_set(mn_log_loss, roc_auc),
              resamples = loan_folds)

# Check results
show_best(ph_fit)

ph_last <- 
    loan_wf %>%
    finalize_workflow(select_best(ph_fit, "mn_log_loss")) %>%
    last_fit(loan_split)

xgb_last

collect_predictions(xgb_last) %>%
    mn_log_loss(TARGET_EVENT, .pred_HR)


extract_workflow(xgb_last) %>%
    extract_fit_parsnip() %>%
    vip(geom = "point", num_features = 15)
    


