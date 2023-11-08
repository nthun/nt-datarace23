# Create prototype

library(tidyverse)
library(googledrive)
library(tidymodels)
library(censored)
library(embed)
library(finetune)
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
    slice_sample(prop = .25) |> 
    mutate(TARGET_EVENT_DAY = if_else(TARGET_EVENT == "E", NA_real_, TARGET_EVENT_DAY),
           TARGET_EVENT = if_else(TARGET_EVENT == "K", 1L, 0L)) |> 
    # mutate(surv = Surv(TARGET_EVENT_DAY, TARGET_EVENT), .keep = "unused") |> 
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
    recipe(surv ~ ., data = loan_training) |> 
    step_string2factor(all_string_predictors()) |> 
    update_role(BORROWER_ID, new_role = "id") |> 
    step_embed(CONTRACT_ID, outcome = TARGET_EVENT)
    
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
    proportional_hazards(penalty = tune(), mixture = 1) |> 
    set_engine("glmnet") |> 
    set_mode("censored regression") 

# Workflow
loan_wf <- 
    workflow(preprocessor = loan_rec,
    spec = bt_spec)

# Fit model
set.seed(345)

ph_fit <- 
    ph_spec |> 
    tune_grid(surv ~ ., 
              grid = 5,
              eval_time = 20,
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
    


