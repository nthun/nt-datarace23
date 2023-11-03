# Create prototype

library(tidyverse)
library(googledrive)
library(tidymodels)


# Download and unzip file ------------------------------------------------------
googledrive::drive_download("https://drive.google.com/file/d/17-k5lBWsBudijdfAJ6oJxwVXMd_EzVlL/view?usp=drive_link", path = "data/dr2023.zip")

# Password to unzip: data23
system(paste("unzip -P", "data23", "data/dr2023.zip", "-d data"))

# Read training file -----------------------------------------------------------
# See data description here: https://eval.dataracing.hu/web/challenges/challenge-page/5/overview

loan_raw <- read_csv("data/training_data.csv")
submission_example <- read_csv("data/data_submission_example.csv")


glimpse(loan_raw)
glimpse(submission_example)

count(loan_raw, TARGET_EVENT)
