# ========================================================================
# SYRIATEL CHURN (SIMPLE VERSION) - 85%+ TARGET
# ========================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(caret)
  library(xgboost)
  library(pROC)
})

cat("\n")
cat(rep("=", 80), "\n")
cat("              SYRIATEL CHURN - SIMPLE VERSION\n")
cat(rep("=", 80), "\n\n")

# Load data
cat("Loading SyriaTel dataset...\n")
churn_data <- read.csv("syriatel_churn.csv")
cat(sprintf("Records: %d, Features: %d\n\n", nrow(churn_data), ncol(churn_data)))

# Check churn distribution
cat("Churn distribution:\n")
churn_dist <- table(churn_data$churn)
print(churn_dist)
cat(sprintf("Churn rate: %.2f%%\n\n", (churn_dist[2] / sum(churn_dist)) * 100))

# Preprocessing - select numeric features
cat("Preprocessing...\n")
# Select key numeric features
numeric_features <- c("account.length", "number.vmail.messages", "total.day.minutes", 
                     "total.day.calls", "total.day.charge", "total.eve.minutes", 
                     "total.eve.calls", "total.eve.charge", "total.night.minutes", 
                     "total.night.calls", "total.night.charge", "total.intl.minutes", 
                     "total.intl.calls", "total.intl.charge", "customer.service.calls")

processed_data <- churn_data %>%
  select(all_of(numeric_features)) %>%
  mutate(
    # Convert churn to 0/1 (True=1, False=0)
    Churn = ifelse(churn_data$churn == TRUE, 1, 0)
  )

# Check for missing values
missing_vals <- colSums(is.na(processed_data))
cat("Missing values:\n")
print(missing_vals[missing_vals > 0])
if(sum(missing_vals) == 0) cat("No missing values found.\n")
cat("\n")

# Split data with stratification
set.seed(123)
# Simple random sampling since createDataPartition is having issues
n_train <- floor(0.8 * nrow(processed_data))
train_index <- sample(1:nrow(processed_data), n_train)
train_data <- processed_data[train_index, ]
test_data <- processed_data[-train_index, ]

cat(sprintf("Training set: %d records (Churn: %.2f%%)\n", nrow(train_data), 
    (mean(train_data$Churn) * 100)))
cat(sprintf("Test set: %d records (Churn: %.2f%%)\n\n", nrow(test_data),
    (mean(test_data$Churn) * 100)))

# Prepare matrices for XGBoost
x_train <- as.matrix(train_data[, 1:15])  # First 15 columns are features
x_test <- as.matrix(test_data[, 1:15])
y_train <- train_data$Churn  # 0/1 labels
y_test <- test_data$Churn

# Convert to DMatrix for XGBoost
dtrain <- xgb.DMatrix(data = x_train, label = y_train)
dtest <- xgb.DMatrix(data = x_test, label = y_test)

# Train XGBoost model
cat("Training XGBoost model...\n")
xgb_model <- xgb.train(
  params = list(
    objective = "binary:logistic",
    eval_metric = "logloss",
    max_depth = 6,
    eta = 0.1,
    subsample = 0.8,
    colsample_bytree = 0.8
  ),
  data = dtrain,
  nrounds = 200,
  verbose = 0
)

# Make predictions
xgb_probs <- predict(xgb_model, dtest)

# Simple evaluation
preds_binary <- ifelse(xgb_probs > 0.5, 1, 0)
accuracy <- sum(preds_binary == y_test) / length(y_test)

# Results
cat("\n")
cat(rep("=", 80), "\n")
cat("                    SYRIATEL CHURN RESULTS\n")
cat(rep("=", 80), "\n")
cat(sprintf("✅ ACCURACY:    %.2f%%\n", accuracy * 100))
cat(sprintf("✅ AVERAGE PROBABILITY FOR CHURN: %.4f\n", mean(xgb_probs[y_test == 1])))
cat(sprintf("✅ AVERAGE PROBABILITY FOR NON-CHURN: %.4f\n", mean(xgb_probs[y_test == 0])))
cat("\n")

# Confusion Matrix
cm <- table(Predicted = preds_binary, Actual = y_test)
cat("Confusion Matrix:\n")
print(cm)
cat("\n")

# Save model if accuracy is good
if(accuracy >= 0.85) {
  cat("🎉 85%+ TARGET ACHIEVED!\n")
  saveRDS(list(model = xgb_model, accuracy = accuracy), 
          "WINNER_SYRIATEL_85.rds")
  cat("✅ Model saved to: WINNER_SYRIATEL_85.rds\n")
} else {
  cat(sprintf("❌ Below target by %.2f%%\n", (0.85 - accuracy) * 100))
}

cat(rep("=", 80), "\n")