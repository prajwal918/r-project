

library(tidyverse)
library(ggplot2)
library(caret)
library(randomForest)
library(pROC) 
library(gridExtra)  

churn_data <- read.csv("WA_Fn-UseC_-Telco-Customer-Churn.xls")

churn_data <- churn_data %>%
  mutate(TotalCharges = as.numeric(TotalCharges)) %>%
  mutate(TotalCharges = ifelse(is.na(TotalCharges), 0, TotalCharges)) %>%
  select(-customerID) 

churn_data <- churn_data %>%
  mutate(across(c(OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
                  StreamingTV, StreamingMovies),
                ~ recode(., "No internet service" = "No"))) %>%
  mutate(MultipleLines = recode(MultipleLines, "No phone service" = "No"))

churn_data <- churn_data %>%
  mutate(across(where(is.character), as.factor))

# --- Train/Test Split ---
set.seed(123)
churn_data$Churn <- as.factor(churn_data$Churn)

trainIndex <- createDataPartition(churn_data$Churn, p = 0.8, list = FALSE, times = 1)
train_data <- churn_data[trainIndex, ]
test_data  <- churn_data[-trainIndex, ]

cat("=== STEP 1: Training Base Models ===\n")

# --- Train Base Models ---
# 1. Logistic Regression
log_model <- glm(Churn ~ ., data = train_data, family = binomial)
cat("✓ Logistic Regression trained\n")

# 2. Random Forest
rf_model <- randomForest(Churn ~ ., data = train_data, ntree = 100)
cat("✓ Random Forest trained\n")

# --- Get Base Model Predictions on Test Set ---
# Get probability predictions for meta-model
log_pred_prob <- predict(log_model, test_data, type = "response")
rf_pred_prob <- predict(rf_model, test_data, type = "prob")[, "Yes"]

# Get class predictions for base model evaluation
log_pred_class <- factor(ifelse(log_pred_prob > 0.5, "Yes", "No"), 
                         levels = levels(test_data$Churn))
rf_pred_class <- predict(rf_model, test_data)

# Base model performance
cm_log <- confusionMatrix(log_pred_class, test_data$Churn, positive = "Yes")
cm_rf <- confusionMatrix(rf_pred_class, test_data$Churn, positive = "Yes")
print(cm_log)
print(cm_rf)


cat("\n=== Base Model Performance ===\n")
cat("Logistic Regression Accuracy:", round(cm_log$overall['Accuracy'], 4), "\n")
cat("Random Forest Accuracy:", round(cm_rf$overall['Accuracy'], 4), "\n")

# --- Create Meta-Features ---
cat("\n=== STEP 2: Creating Meta-Model ===\n")

# For the meta-model, we need to create meta-features from base model predictions
# We'll use cross-validation on training data to avoid overfitting

# Create folds for cross-validation
set.seed(123)
folds <- createFolds(train_data$Churn, k = 5, list = TRUE)

# Initialize matrices to store out-of-fold predictions
meta_train_log <- numeric(nrow(train_data))
meta_train_rf <- numeric(nrow(train_data))

# Generate out-of-fold predictions for training data
for(i in 1:length(folds)) {
  cat("Processing fold", i, "of", length(folds), "\n")
  
  # Split into train and validation
  val_idx <- folds[[i]]
  fold_train <- train_data[-val_idx, ]
  fold_val <- train_data[val_idx, ]
  
  # Train base models on fold
  fold_log <- glm(Churn ~ ., data = fold_train, family = binomial)
  fold_rf <- randomForest(Churn ~ ., data = fold_train, ntree = 100)
  
  # Predict on validation fold
  meta_train_log[val_idx] <- predict(fold_log, fold_val, type = "response")
  meta_train_rf[val_idx] <- predict(fold_rf, fold_val, type = "prob")[, "Yes"]
}

# Create meta-training dataset
meta_train_data <- data.frame(
  log_pred = meta_train_log,
  rf_pred = meta_train_rf,
  Churn = train_data$Churn
)

# Create meta-test dataset
meta_test_data <- data.frame(
  log_pred = log_pred_prob,
  rf_pred = rf_pred_prob,
  Churn = test_data$Churn
)

# --- Train Meta-Model (Logistic Regression on base predictions) ---
meta_model <- glm(Churn ~ log_pred + rf_pred, 
                  data = meta_train_data, 
                  family = binomial)
print(meta_model)
cat("✓ Meta-model trained\n")
cat("\nMeta-Model Coefficients:\n")
print(summary(meta_model)$coefficients)

# --- Meta-Model Predictions ---
meta_pred_prob <- predict(meta_model, meta_test_data, type = "response")
meta_pred_class <- factor(ifelse(meta_pred_prob > 0.5, "Yes", "No"), 
                          levels = levels(test_data$Churn))

# Meta-model performance
cm_meta <- confusionMatrix(meta_pred_class, test_data$Churn, positive = "Yes")

cat("\n=== STEP 3: Model Performance Comparison ===\n")
performance_summary <- data.frame(
  Model = c("Logistic Regression", "Random Forest", "Meta-Model (Ensemble)"),
  Accuracy = c(cm_log$overall['Accuracy'], 
               cm_rf$overall['Accuracy'], 
               cm_meta$overall['Accuracy']),
  Sensitivity = c(cm_log$byClass['Sensitivity'], 
                  cm_rf$byClass['Sensitivity'], 
                  cm_meta$byClass['Sensitivity']),
  Specificity = c(cm_log$byClass['Specificity'], 
                  cm_rf$byClass['Specificity'], 
                  cm_meta$byClass['Specificity']),
  F1_Score = c(cm_log$byClass['F1'], 
               cm_rf$byClass['F1'], 
               cm_meta$byClass['F1'])
)

print(performance_summary)







# --- STEP 4: ROC Curves ---
cat("\n=== STEP 4: Creating ROC Curves ===\n")

# Create ROC objects
y_test_numeric <- ifelse(test_data$Churn == "Yes", 1, 0)

roc_log <- roc(y_test_numeric, log_pred_prob, quiet = TRUE)
roc_rf <- roc(y_test_numeric, rf_pred_prob, quiet = TRUE)
roc_meta <- roc(y_test_numeric, meta_pred_prob, quiet = TRUE)

# Calculate AUC
auc_log <- auc(roc_log)
auc_rf <- auc(roc_rf)
auc_meta <- auc(roc_meta)

cat("Logistic Regression AUC:", round(auc_log, 4), "\n")
cat("Random Forest AUC:", round(auc_rf, 4), "\n")
cat("Meta-Model AUC:", round(auc_meta, 4), "\n")

# Plot ROC curves
roc_data <- data.frame(
  Model = c(rep("Logistic Regression", length(roc_log$specificities)),
            rep("Random Forest", length(roc_rf$specificities)),
            rep("Meta-Model", length(roc_meta$specificities))),
  Specificity = c(roc_log$specificities, roc_rf$specificities, roc_meta$specificities),
  Sensitivity = c(roc_log$sensitivities, roc_rf$sensitivities, roc_meta$sensitivities),
  AUC = c(rep(paste0("AUC = ", round(auc_log, 3)), length(roc_log$specificities)),
          rep(paste0("AUC = ", round(auc_rf, 3)), length(roc_rf$specificities)),
          rep(paste0("AUC = ", round(auc_meta, 3)), length(roc_meta$specificities)))
)

roc_plot <- ggplot(roc_data, aes(x = 1 - Specificity, y = Sensitivity, color = Model)) +
  geom_line(size = 1.2) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray50") +
  labs(title = "ROC Curves Comparison",
       subtitle = "Logistic Regression, Random Forest, and Meta-Model",
       x = "False Positive Rate (1 - Specificity)",
       y = "True Positive Rate (Sensitivity)") +
  theme_minimal() +
  theme(legend.position = "bottom",
        plot.title = element_text(face = "bold", size = 14),
        legend.text = element_text(size = 10)) +
  scale_color_manual(values = c("Logistic Regression" = "#E74C3C", 
                                 "Random Forest" = "#3498DB",
                                 "Meta-Model" = "#2ECC71"),
                     labels = c(paste0("Logistic Regression (AUC = ", round(auc_log, 3), ")"),
                               paste0("Random Forest (AUC = ", round(auc_rf, 3), ")"),
                               paste0("Meta-Model (AUC = ", round(auc_meta, 3), ")")))

print(roc_plot)

# Save ROC plot
ggsave("ROC_Curves_MetaModel.png", roc_plot, width = 10, height = 7, dpi = 300)
cat("\n✓ ROC curve saved as 'ROC_Curves_MetaModel.png'\n")

# --- STEP 5: Accuracy Curves ---
cat("\n=== STEP 5: Creating Accuracy Curves ===\n")

# Create accuracy vs threshold curves
thresholds <- seq(0, 1, by = 0.01)

accuracy_data <- data.frame()

for(thresh in thresholds) {
  # Logistic Regression
  log_class_temp <- factor(ifelse(log_pred_prob > thresh, "Yes", "No"), 
                           levels = levels(test_data$Churn))
  acc_log <- sum(log_class_temp == test_data$Churn) / length(test_data$Churn)
  
  # Random Forest
  rf_class_temp <- factor(ifelse(rf_pred_prob > thresh, "Yes", "No"), 
                          levels = levels(test_data$Churn))
  acc_rf <- sum(rf_class_temp == test_data$Churn) / length(test_data$Churn)
  
  # Meta-Model
  meta_class_temp <- factor(ifelse(meta_pred_prob > thresh, "Yes", "No"), 
                            levels = levels(test_data$Churn))
  acc_meta <- sum(meta_class_temp == test_data$Churn) / length(test_data$Churn)
  
  accuracy_data <- rbind(accuracy_data, 
                        data.frame(Threshold = thresh, 
                                  Accuracy = c(acc_log, acc_rf, acc_meta),
                                  Model = c("Logistic Regression", "Random Forest", "Meta-Model")))
}




# Plot accuracy curves
accuracy_plot <- ggplot(accuracy_data, aes(x = Threshold, y = Accuracy, color = Model)) +
  geom_line(size = 1.2) +
  geom_vline(xintercept = 0.5, linetype = "dashed", color = "gray50", alpha = 0.7) +
  labs(title = "Accuracy vs Threshold",
       subtitle = "Model Performance Across Different Classification Thresholds",
       x = "Classification Threshold",
       y = "Accuracy") +
  theme_minimal() +
  theme(legend.position = "bottom",
        plot.title = element_text(face = "bold", size = 14),
        legend.text = element_text(size = 10)) +
  scale_color_manual(values = c("Logistic Regression" = "#E74C3C", 
                                 "Random Forest" = "#3498DB",
                                 "Meta-Model" = "#2ECC71")) +
  annotate("text", x = 0.5, y = max(accuracy_data$Accuracy) * 0.95, 
           label = "Default Threshold", angle = 90, vjust = -0.5, size = 3, color = "gray40")

print(accuracy_plot)

# Save accuracy plot
ggsave("Accuracy_Curves_MetaModel.png", accuracy_plot, width = 10, height = 7, dpi = 300)
cat("✓ Accuracy curve saved as 'Accuracy_Curves_MetaModel.png'\n")

# --- STEP 6: Additional Visualizations ---
cat("\n=== STEP 6: Creating Additional Performance Charts ===\n")

# Bar chart of model metrics
metrics_long <- performance_summary %>%
  pivot_longer(cols = -Model, names_to = "Metric", values_to = "Value")

metrics_plot <- ggplot(metrics_long, aes(x = Model, y = Value, fill = Metric)) +
  geom_bar(stat = "identity", position = "dodge", alpha = 0.8) +
  labs(title = "Model Performance Metrics Comparison",
       subtitle = "Accuracy, Sensitivity, Specificity, and F1-Score",
       x = "Model",
       y = "Score") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 15, hjust = 1),
        plot.title = element_text(face = "bold", size = 14),
        legend.position = "bottom") +
  scale_fill_brewer(palette = "Set2") +
  ylim(0, 1)

print(metrics_plot)

ggsave("Performance_Metrics_Comparison.png", metrics_plot, width = 10, height = 7, dpi = 300)
cat("✓ Metrics comparison saved as 'Performance_Metrics_Comparison.png'\n")

# Confusion matrices visualization
cm_data <- data.frame(
  Model = rep(c("Logistic Regression", "Random Forest", "Meta-Model"), each = 4),
  Prediction = rep(rep(c("No", "Yes"), each = 2), 3),
  Actual = rep(c("No", "Yes"), 6),
  Count = c(
    cm_log$table[1,1], cm_log$table[1,2], cm_log$table[2,1], cm_log$table[2,2],
    cm_rf$table[1,1], cm_rf$table[1,2], cm_rf$table[2,1], cm_rf$table[2,2],
    cm_meta$table[1,1], cm_meta$table[1,2], cm_meta$table[2,1], cm_meta$table[2,2]
  )
)

cm_plot <- ggplot(cm_data, aes(x = Actual, y = Prediction, fill = Count)) +
  geom_tile(color = "white", size = 1) +
  geom_text(aes(label = Count), size = 5, fontface = "bold") +
  facet_wrap(~Model, ncol = 3) +
  labs(title = "Confusion Matrices Comparison",
       x = "Actual Class",
       y = "Predicted Class") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 14),
        strip.text = element_text(face = "bold", size = 11)) +
  scale_fill_gradient(low = "#FFFFFF", high = "#3498DB")

print(cm_plot)

ggsave("Confusion_Matrices_Comparison.png", cm_plot, width = 12, height = 5, dpi = 300)
cat("✓ Confusion matrices saved as 'Confusion_Matrices_Comparison.png'\n")

# --- Final Summary ---
cat("\n" , rep("=", 60), "\n", sep = "")
cat("FINAL SUMMARY: META-MODEL ENSEMBLE\n")
cat(rep("=", 60), "\n", sep = "")
cat("\n✓ Base Models Trained: Logistic Regression & Random Forest\n")
cat("✓ Meta-Model Created: Stacking Ensemble\n")
cat("✓ ROC Curves Generated with AUC scores\n")
cat("✓ Accuracy Curves Plotted\n")
cat("\nBest Model:", performance_summary$Model[which.max(performance_summary$Accuracy)], "\n")
cat("Best Accuracy:", round(max(performance_summary$Accuracy), 4), "\n")
cat("Best AUC:", round(max(c(auc_log, auc_rf, auc_meta)), 4), "\n")
cat("\nAll visualizations saved to working directory!\n")
cat(rep("=", 60), "\n", sep = "")
