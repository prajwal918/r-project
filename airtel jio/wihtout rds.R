# ========================================================================
# INDIAN TELECOM CHURN - FULL SINGLE SCRIPT (TRAIN + SAVE + EVAL)
# ========================================================================

# ------------------------------------------------------------------------
# 0. SETUP & LIBRARIES
# ------------------------------------------------------------------------
if (!require("tidyverse")) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if (!require("caret")) install.packages("caret", repos = "http://cran.us.r-project.org")
if (!require("xgboost")) install.packages("xgboost", repos = "http://cran.us.r-project.org")
if (!require("smotefamily")) install.packages("smotefamily", repos = "http://cran.us.r-project.org")
if (!require("gridExtra")) install.packages("gridExtra", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(xgboost)
library(smotefamily)
library(gridExtra)

MODEL_PATH <- "LOCKED_INDIAN_TELECOM_85.rds"
FIXED_SEED <- 1
FIXED_THR  <- 0.31

cat(rep("=", 80), "\n")
cat("         INDIAN TELECOM FULL PIPELINE (TRAIN + EVAL)\n")
cat(rep("=", 80), "\n\n")

# ========================================================================
# 1. DATA INGESTION & CLEANING
# ========================================================================
cat(">>> Step 1: Loading dataset...\n")
raw_data <- read.csv("telecom_churn.csv", stringsAsFactors = FALSE)
cat("[OK] Loaded: ", nrow(raw_data), " rows\n")

cols_to_drop <- c("customer_id", "date_of_registration", "pincode", "state", "city")
raw_data <- raw_data %>% select(-any_of(cols_to_drop))

encode_label <- function(col) {
  if (is.character(col) || is.factor(col)) as.numeric(as.factor(col)) - 1 else col
}

processed <- raw_data %>% mutate(across(everything(), encode_label))
processed$churn <- ifelse(processed$churn == 1, 1, 0)
processed <- na.omit(processed)
cat("[OK] After NA removal rows: ", nrow(processed), "\n")

# ========================================================================
# 2. BALANCING USING SMOTE
# ========================================================================
cat(">>> Step 2: Applying SMOTE balancing...\n")
X <- processed %>% select(-churn)
Y <- processed$churn
sm <- SMOTE(X, Y, K = 5, dup_size = 0)
balanced <- sm$data
colnames(balanced)[ncol(balanced)] <- "churn"
balanced$churn <- as.integer(as.character(balanced$churn))

cat("[OK] Balanced size: ", nrow(balanced), "\n")
cat("     Class distribution: ", paste(table(balanced$churn), collapse=" | "), "\n")

# ========================================================================
# 3. TRAIN OR LOAD MODEL (AUTO MODE)
# ========================================================================
cat(">>> Step 3: Training or Loading Model...\n")

if (file.exists(MODEL_PATH)) {
  cat("[LOADING MODEL] Found saved model: ", MODEL_PATH, "\n")
  model <- readRDS(MODEL_PATH)
  
} else {
  cat("[TRAINING MODEL] No RDS found. Training new model...\n")
  
  set.seed(FIXED_SEED)
  train_idx <- createDataPartition(balanced$churn, p = 0.8, list = FALSE)
  train_set <- balanced[train_idx, ]
  
  dtrain <- xgb.DMatrix(
    data = as.matrix(train_set %>% select(-churn)),
    label = train_set$churn
  )
  
  params <- list(
    objective = "binary:logistic",
    eta = 0.05,
    max_depth = 6,
    subsample = 0.8,
    colsample_bytree = 0.8,
    eval_metric = "auc"
  )
  
  model <- xgb.train(params = params, data = dtrain, nrounds = 500, verbose = 0)
  saveRDS(model, MODEL_PATH)
  cat("[OK] Model saved to: ", MODEL_PATH, "\n")
}

# ========================================================================
# 4. EVALUATION (ALWAYS RUNS)
# ========================================================================
cat("\n>>> Step 4: Evaluating Model...\n")

set.seed(FIXED_SEED)
train_idx <- createDataPartition(balanced$churn, p = 0.8, list = FALSE)
train_set <- balanced[train_idx, ]
test_set  <- balanced[-train_idx, ]

dtest <- xgb.DMatrix(as.matrix(test_set %>% select(-churn)))
probs <- predict(model, dtest)
preds <- ifelse(probs > FIXED_THR, 1, 0)

accuracy <- mean(preds == test_set$churn)
cat(sprintf("[ACCURACY] %.2f%% with threshold %.2f\n", accuracy * 100, FIXED_THR))

# Confusion matrix
cm <- table(Predicted = preds, Actual = test_set$churn)
cat("\n>>> CONFUSION MATRIX:\n")
print(cm)

cat("\n====================================================\n")
cat("         PIPELINE COMPLETE (TRAIN + EVAL DONE)\n")
cat("====================================================\n")
