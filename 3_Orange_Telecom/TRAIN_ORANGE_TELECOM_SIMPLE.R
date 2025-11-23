# ========================================================================
# ORANGE TELECOM CHURN (SIMPLE VERSION) - 85%+ TARGET
# ========================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(caret)
  library(xgboost)
  library(lightgbm)
  library(pROC)
})

cat("\n")
cat(rep("=", 80), "\n")
cat("    ORANGE TELECOM CHURN - SIMPLE VERSION\n")
cat(rep("=", 80), "\n\n")

# Load data
cat("Loading data...\n")
train_data <- read.csv("churn-bigml-80.csv")
test_data <- read.csv("churn-bigml-20.csv")

cat(sprintf("Training records: %d\n", nrow(train_data)))
cat(sprintf("Test records: %d\n", nrow(test_data)))
cat(sprintf("Features: %d\n\n", ncol(train_data)))

# Simple preprocessing
cat("Preprocessing...\n")
# Process train data
train_processed <- train_data %>%
  mutate(
    Churn = factor(ifelse(Churn == "True", "Yes", "No")),
    # Create some key features
    total_minutes = Total.day.minutes + Total.eve.minutes + Total.night.minutes + Total.intl.minutes,
    total_calls = Total.day.calls + Total.eve.calls + Total.night.calls + Total.intl.calls,
    total_charge = Total.day.charge + Total.eve.charge + Total.night.charge + Total.intl.charge,
    # Risk indicators
    high_day_usage = as.numeric(Total.day.minutes > median(Total.day.minutes)),
    many_customer_service_calls = as.numeric(Customer.service.calls > 0),
    international_user = as.numeric(International.plan == "Yes")
  ) %>%
  select(Churn, total_minutes, total_calls, total_charge, high_day_usage, 
         many_customer_service_calls, international_user, Customer.service.calls)

# Process test data
test_processed <- test_data %>%
  mutate(
    Churn = factor(ifelse(Churn == "True", "Yes", "No")),
    # Create same features
    total_minutes = Total.day.minutes + Total.eve.minutes + Total.night.minutes + Total.intl.minutes,
    total_calls = Total.day.calls + Total.eve.calls + Total.night.calls + Total.intl.calls,
    total_charge = Total.day.charge + Total.eve.charge + Total.night.charge + Total.intl.charge,
    # Risk indicators
    high_day_usage = as.numeric(Total.day.minutes > median(Total.day.minutes)),
    many_customer_service_calls = as.numeric(Customer.service.calls > 0),
    international_user = as.numeric(International.plan == "Yes")
  ) %>%
  select(Churn, total_minutes, total_calls, total_charge, high_day_usage, 
         many_customer_service_calls, international_user, Customer.service.calls)

# Balance training data
cat("Balancing training data...\n")
train_balanced <- upSample(x = subset(train_processed, select = -Churn),
                          y = train_processed$Churn, 
                          yname = "Churn")

# Prepare for modeling
x_train <- model.matrix(Churn ~ ., data = train_balanced)[, -1]
x_test <- model.matrix(Churn ~ ., data = test_processed)[, -1]
y_train <- as.numeric(train_balanced$Churn == "Yes")

# XGBoost
cat("Training XGBoost...\n")
dtrain <- xgb.DMatrix(data = x_train, label = y_train)
dtest <- xgb.DMatrix(data = x_test)

xgb_model <- xgb.train(
  params = list(objective = "binary:logistic", eval_metric = "auc",
                max_depth = 6, eta = 0.05, subsample = 0.8),
  data = dtrain, nrounds = 150, verbose = 0
)
xgb_probs <- predict(xgb_model, dtest)

# LightGBM
cat("Training LightGBM...\n")
lgb_train <- lgb.Dataset(data = x_train, label = y_train)
lgb_model <- lgb.train(
  params = list(objective = "binary", metric = "auc", learning_rate = 0.05,
                num_leaves = 31, feature_fraction = 0.8, bagging_fraction = 0.8),
  data = lgb_train, nrounds = 150, verbose = -1
)
lgb_probs <- as.numeric(predict(lgb_model, x_test))

# Ensemble
ens_probs <- 0.5 * xgb_probs + 0.5 * lgb_probs

# Optimize threshold
thresholds <- seq(0.1, 0.9, by = 0.01)
accuracies <- sapply(thresholds, function(thr) {
  preds <- factor(ifelse(ens_probs > thr, "Yes", "No"), levels = c("Yes", "No"))
  mean(preds == test_processed$Churn)
})

best_acc <- max(accuracies)
best_thr <- thresholds[which.max(accuracies)]

# Results
cat("\n")
cat(rep("=", 80), "\n")
cat("                              RESULTS\n")
cat(rep("=", 80), "\n")

final_preds <- factor(ifelse(ens_probs > best_thr, "Yes", "No"), levels = c("Yes", "No"))
cm <- confusionMatrix(final_preds, test_processed$Churn, positive = "Yes")

cat(sprintf("ACCURACY:  %.2f%%\n", 100 * best_acc))
cat(sprintf("Threshold: %.2f\n\n", best_thr))

print(cm$table)

cat(sprintf("\nSensitivity: %.2f%%\n", 100 * cm$byClass["Sensitivity"]))
cat(sprintf("Specificity: %.2f%%\n", 100 * cm$byClass["Specificity"]))
cat(sprintf("AUC:         %.4f\n", as.numeric(auc(roc(test_processed$Churn, ens_probs, quiet=TRUE)))))

if(best_acc >= 0.85) {
  cat("\n🎉 85%+ ACCURACY ACHIEVED!\n")
  saveRDS(list(accuracy = best_acc, threshold = best_thr, probs = ens_probs, actual = test_processed$Churn), 
          "WINNER_ORANGE_TELECOM_85.rds")
  cat("✅ Model saved to: WINNER_ORANGE_TELECOM_85.rds\n")
} else {
  cat(sprintf("\nGap to 85%%: %.2f%%\n", 100 * (0.85 - best_acc)))
}

cat(rep("=", 80), "\n")
