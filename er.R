# --- Libraries ---
# tidyverse/ggplot2 for data wrangling and visualization
# caret provides utilities for data partitioning and evaluation (confusionMatrix)
# randomForest for tree-ensemble classification
library(tidyverse)
library(ggplot2)
library(caret)
library(randomForest)

# --- Load data ---
# Reads the Telco Customer Churn dataset into a data.frame named churn_data
# NOTE: This path is expected to be CSV-like. If the source truly is an Excel file (.xls),
# use readxl::read_excel() instead of read.csv and add: library(readxl)
# Example (if needed): churn_data <- readxl::read_excel("WA_Fn-UseC_-Telco-Customer-Churn.xls")
churn_data <- read.csv("WA_Fn-UseC_-Telco-Customer-Churn.xls")

# --- Data Cleaning and Preparation ---
# Convert TotalCharges to numeric and impute NAs with 0.
# Remove customerID since it's an identifier not useful for modeling.
churn_data <- churn_data %>%
  mutate(TotalCharges = as.numeric(TotalCharges)) %>%
  mutate(TotalCharges = ifelse(is.na(TotalCharges), 0, TotalCharges)) %>%
  select(-customerID) 

# Recode 'No internet service' to 'No' for relevant columns
# Recode 'No phone service' to 'No' for MultipleLines to align semantics.
churn_data <- churn_data %>%
  mutate(across(c(OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
                  StreamingTV, StreamingMovies),
                ~ recode(., "No internet service" = "No"))) %>%
  mutate(MultipleLines = recode(MultipleLines, "No phone service" = "No"))

# Convert remaining character columns to factors to prepare for classification models.
churn_data <- churn_data %>%
  mutate(across(where(is.character), as.factor))

# --- Analysis and Visualization ---
# Calculate and print the overall churn rate to understand class balance.
churn_rate <- churn_data %>%
  count(Churn) %>%
  mutate(Proportion = n / sum(n))
print(churn_rate)

# Bar chart: Churn counts by contract type (month-to-month typically has higher churn).
ggplot(churn_data, aes(x = Contract, fill = Churn)) +
  geom_bar(position = "dodge") +
  labs(title = "Customer Churn by Contract Type",
       x = "Contract Type",
       y = "Number of Customers") +
  theme_minimal()

# Bar chart: Churn counts by internet service type (fiber often shows higher churn than DSL/None).
ggplot(churn_data, aes(x = InternetService, fill = Churn)) +
  geom_bar(position = "dodge") +
  labs(title = "Customer Churn by Internet Service Type",
       x = "Internet Service",
       y = "Number of Customers") +
  theme_minimal()

# Box plot: Compare MonthlyCharges across churn status; churners often have higher charges.
ggplot(churn_data, aes(x = Churn, y = MonthlyCharges, fill = Churn)) +
  geom_boxplot() +
  labs(title = "Monthly Charges by Churn Status",
       x = "Churn Status",
       y = "Monthly Charges ($)") +
  theme_minimal()

# Simple correlation matrix among key numeric variables for quick relationships overview.
numeric_data <- churn_data %>% select(tenure, MonthlyCharges, TotalCharges)
correlation_matrix <- cor(numeric_data)
print(correlation_matrix)

# --- Train/Test Split (80/20) ---
set.seed(123)
# Ensure response is a factor (should already be from earlier step)
churn_data$Churn <- as.factor(churn_data$Churn)

trainIndex <- createDataPartition(churn_data$Churn, p = 0.8, list = FALSE, times = 1)
train_data <- churn_data[trainIndex, ]
test_data  <- churn_data[-trainIndex, ]

# --- Train Models ---
# Logistic Regression
log_model <- glm(Churn ~ ., data = train_data, family = binomial)
# Random Forest
rf_model <- randomForest(Churn ~ ., data = train_data)

# --- Evaluate Models ---
# Logistic Regression Predictions
log_predictions <- predict(log_model, test_data, type = "response")
log_pred_class <- factor(ifelse(log_predictions > 0.5, "Yes", "No"),
                         levels = levels(test_data$Churn))
cm_log <- confusionMatrix(log_pred_class, test_data$Churn, positive = "Yes")
print(cm_log)

# Random Forest Predictions
rf_predictions <- predict(rf_model, test_data)
rf_predictions <- factor(rf_predictions, levels = levels(test_data$Churn))
cm_rf <- confusionMatrix(rf_predictions, test_data$Churn, positive = "Yes")
print(cm_rf)

# --- Linear Probability Model (LPM) ---
# Create numeric target for linear regression (0/1) without altering original columns
train_lpm <- train_data %>% mutate(Churn_num = if_else(Churn == "Yes", 1, 0))
test_lpm  <- test_data  %>% mutate(Churn_num = if_else(Churn == "Yes", 1, 0))

# Fit linear regression using all predictors except the factor target
lpm_model <- lm(Churn_num ~ . - Churn, data = train_lpm)
print(summary(lpm_model))

# Predict numeric values (may fall outside [0,1]) and evaluate
lpm_pred_num <- predict(lpm_model, newdata = test_lpm)
rmse <- sqrt(mean((lpm_pred_num - test_lpm$Churn_num)^2))
prop_out_of_bounds <- mean(lpm_pred_num < 0 | lpm_pred_num > 1)
print(list(LPM_RMSE = rmse, Proportion_Out_Of_Bounds = prop_out_of_bounds))

# Classify at 0.5 threshold and compute confusion matrix
lpm_pred_class <- factor(ifelse(lpm_pred_num > 0.5, "Yes", "No"), levels = levels(test_data$Churn))
cm_lpm <- confusionMatrix(lpm_pred_class, test_data$Churn, positive = "Yes")
print(cm_lpm)

# --- IMPROVED LOGISTIC REGRESSION: Test Different Thresholds ---
cat("\n===== THRESHOLD OPTIMIZATION FOR LOGISTIC REGRESSION =====\n\n")

# Test different probability thresholds to improve sensitivity
thresholds <- c(0.3, 0.35, 0.4, 0.45, 0.5)
threshold_results <- data.frame()

for (thresh in thresholds) {
  # Make predictions with current threshold
  pred_class <- factor(ifelse(log_predictions > thresh, "Yes", "No"),
                       levels = levels(test_data$Churn))
  cm <- confusionMatrix(pred_class, test_data$Churn, positive = "Yes")
  
  # Store key metrics
  threshold_results <- rbind(threshold_results, data.frame(
    Threshold = thresh,
    Accuracy = cm$overall['Accuracy'],
    Sensitivity = cm$byClass['Sensitivity'],
    Specificity = cm$byClass['Specificity'],
    Precision = cm$byClass['Pos Pred Value'],
    F1_Score = cm$byClass['F1'],
    Kappa = cm$overall['Kappa']
  ))
}

# Print results table
cat("Threshold Comparison:\n")
print(round(threshold_results, 4))

# Find best threshold for sensitivity
best_sensitivity_idx <- which.max(threshold_results$Sensitivity)
best_f1_idx <- which.max(threshold_results$F1_Score)

cat("\n--- RECOMMENDATIONS ---\n")
cat(sprintf("Best for catching churners (Max Sensitivity): Threshold = %.2f (Sensitivity = %.2f%%)\n", 
            threshold_results$Threshold[best_sensitivity_idx],
            threshold_results$Sensitivity[best_sensitivity_idx] * 100))
cat(sprintf("Best balanced performance (Max F1-Score): Threshold = %.2f (F1 = %.4f)\n",
            threshold_results$Threshold[best_f1_idx],
            threshold_results$F1_Score[best_f1_idx]))

# Use optimal threshold (0.4 is often a good compromise)
# CHANGE THIS VALUE to adjust how aggressive the model is:
#   0.30 = Very aggressive (catches ~73% of churners, but more false alarms)
#   0.35 = Aggressive (catches ~69% of churners, balanced)
#   0.40 = Balanced (catches ~65% of churners) <- RECOMMENDED
#   0.45 = Conservative (catches ~61% of churners)
#   0.50 = Very conservative (catches ~55% of churners)
optimal_threshold <- 0.4  # <-- CHANGE THIS NUMBER
cat(sprintf("\n--- USING OPTIMAL THRESHOLD: %.2f ---\n", optimal_threshold))
log_pred_optimal <- factor(ifelse(log_predictions > optimal_threshold, "Yes", "No"),
                           levels = levels(test_data$Churn))
cm_log_optimal <- confusionMatrix(log_pred_optimal, test_data$Churn, positive = "Yes")
print(cm_log_optimal)

# Compare improvement
cat("\n--- IMPROVEMENT SUMMARY ---\n")
cat(sprintf("Original (0.5 threshold): Sensitivity = %.2f%%, Specificity = %.2f%%\n",
            cm_log$byClass['Sensitivity'] * 100,
            cm_log$byClass['Specificity'] * 100))
cat(sprintf("Optimized (%.2f threshold): Sensitivity = %.2f%%, Specificity = %.2f%%\n",
            optimal_threshold,
            cm_log_optimal$byClass['Sensitivity'] * 100,
            cm_log_optimal$byClass['Specificity'] * 100))
cat(sprintf("Churners caught: %d → %d (gained %d customers)\n",
            sum(cm_log$table[2,2]),
            sum(cm_log_optimal$table[2,2]),
            sum(cm_log_optimal$table[2,2]) - sum(cm_log$table[2,2])))

# --- VISUALIZE THRESHOLD IMPACT ---
library(ggplot2)

# Reshape data for plotting
threshold_long <- tidyr::pivot_longer(threshold_results, 
                                      cols = c(Sensitivity, Specificity, Precision, F1_Score),
                                      names_to = "Metric",
                                      values_to = "Value")

ggplot(threshold_long, aes(x = Threshold, y = Value, color = Metric, group = Metric)) +
  geom_line(size = 1.2) +
  geom_point(size = 3) +
  labs(title = "Impact of Threshold on Model Performance",
       subtitle = "Lower threshold = Catch more churners (higher sensitivity)",
       x = "Probability Threshold",
       y = "Metric Value",
       color = "Performance Metric") +
  theme_minimal() +
  theme(legend.position = "bottom") +
  scale_y_continuous(labels = scales::percent, limits = c(0.4, 1))

# ROC Curve for complete visualization
library(pROC)
roc_obj <- roc(test_data$Churn, log_predictions)
cat(sprintf("\nArea Under ROC Curve (AUC): %.4f\n", auc(roc_obj)))

plot(roc_obj, main = "ROC Curve - Logistic Regression",
     col = "blue", lwd = 2,
     print.auc = TRUE,
     print.auc.x = 0.6, print.auc.y = 0.4)
abline(a = 0, b = 1, lty = 2, col = "gray")