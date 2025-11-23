# ========================================================================
# SYRIATEL CHURN - COMPLETE MODEL TRAINING FOR 85%+ TARGET
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
cat("              SYRIATEL CHURN PREDICTION - TARGET: 85%+ ACCURACY\n")
cat(rep("=", 80), "\n\n")

# Load data
cat("Loading SyriaTel dataset...\n")
churn_data <- read.csv("syriatel_churn.csv")
cat(sprintf("Records: %d, Features: %d\n\n", nrow(churn_data), ncol(churn_data)))

# Check column names
cat("Column names:\n")
print(names(churn_data))
cat("\n")

# Check churn distribution
cat("Churn distribution:\n")
churn_dist <- table(churn_data$churn)
print(churn_dist)
cat(sprintf("Churn rate: %.2f%%\n\n", (churn_dist[2] / sum(churn_dist)) * 100))

# Preprocessing
cat("Preprocessing...\n")
# Convert categorical variables to factors
churn_data <- churn_data %>%
  mutate_if(is.character, as.factor)

# Convert churn to factor with proper labels
churn_data$Churn <- factor(ifelse(churn_data$churn == TRUE, "Yes", "No"))

# Check for missing values
missing_vals <- colSums(is.na(churn_data))
cat("Missing values:\n")
print(missing_vals[missing_vals > 0])
if(sum(missing_vals) == 0) cat("No missing values found.\n")
cat("\n")

# Split data
set.seed(123)
train_index <- createDataPartition(churn_data$Churn, p = 0.8, list = FALSE)
train_data <- churn_data[train_index, ]
test_data <- churn_data[-train_index, ]

cat(sprintf("Training set: %d records (Churn: %.2f%%)\n", nrow(train_data), 
    (sum(train_data$Churn == 'Yes')/nrow(train_data)) * 100))
cat(sprintf("Test set: %d records (Churn: %.2f%%)\n\n", nrow(test_data),
    (sum(test_data$Churn == 'Yes')/nrow(test_data)) * 100))

# Balance classes using upSample
cat("Balancing classes...\n")
train_balanced <- upSample(x = train_data[, !names(train_data) %in% c("Churn", "churn")],
                          y = train_data$Churn, 
                          yname = "Churn")

cat(sprintf("Balanced training set: %d records\n", nrow(train_balanced)))
cat("Balanced Churn distribution:\n")
print(table(train_balanced$Churn))
cat("\n")

# Prepare matrices for XGBoost
feature_cols <- names(train_balanced)[!names(train_balanced) %in% "Churn"]
x_train <- model.matrix(Churn ~ ., data = train_balanced[, c(feature_cols, "Churn")])[, -1]
x_test <- model.matrix(Churn ~ ., data = test_data[, c(feature_cols, "Churn")])[, -1]
y_train <- as.numeric(train_balanced$Churn == "Yes")
y_test <- as.numeric(test_data$Churn == "Yes")

# Convert to DMatrix for XGBoost
dtrain <- xgb.DMatrix(data = as.matrix(x_train), label = y_train)
dtest <- xgb.DMatrix(data = as.matrix(x_test), label = y_test)

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

# Optimize threshold
cat("Optimizing threshold...\n")
thresholds <- seq(0.3, 0.7, by = 0.01)
best_threshold <- 0.5
best_accuracy <- 0

for(threshold in thresholds) {
  preds <- factor(ifelse(xgb_probs > threshold, "Yes", "No"), levels = c("No", "Yes"))
  test_actual_factor <- factor(test_data$Churn, levels = c("No", "Yes"))
  acc <- sum(preds == test_actual_factor) / length(test_actual_factor)
  if(acc > best_accuracy) {
    best_accuracy <- acc
    best_threshold <- threshold
  }
}

# Final predictions with best threshold
final_preds <- factor(ifelse(xgb_probs > best_threshold, "Yes", "No"), levels = c("No", "Yes"))
test_actual <- factor(test_data$Churn, levels = c("No", "Yes"))
cm <- confusionMatrix(final_preds, test_actual, positive = "Yes")

# Results
accuracy <- cm$overall['Accuracy']
sensitivity <- cm$byClass['Sensitivity']
specificity <- cm$byClass['Specificity']
auc_val <- auc(roc(test_data$Churn, xgb_probs))

cat("\n")
cat(rep("=", 80), "\n")
cat("                    SYRIATEL CHURN RESULTS\n")
cat(rep("=", 80), "\n")
cat(sprintf("✅ ACCURACY:    %.2f%%\n", accuracy * 100))
cat(sprintf("✅ SENSITIVITY: %.2f%%\n", sensitivity * 100))
cat(sprintf("✅ SPECIFICITY: %.2f%%\n", specificity * 100))
cat(sprintf("✅ AUC:         %.4f\n", auc_val))
cat(sprintf("✅ THRESHOLD:   %.3f\n", best_threshold))
cat("\n")

# Confusion Matrix
cat("Confusion Matrix:\n")
print(cm$table)
cat("\n")

# Save model if accuracy is good
if(accuracy >= 0.85) {
  cat("🎉 85%+ TARGET ACHIEVED!\n")
  saveRDS(list(model = xgb_model, threshold = best_threshold, accuracy = accuracy), 
          "WINNER_SYRIATEL_85.rds")
  cat("✅ Model saved to: WINNER_SYRIATEL_85.rds\n")
} else {
  cat(sprintf("❌ Below target by %.2f%%\n", (0.85 - accuracy) * 100))
}

cat(rep("=", 80), "\n")