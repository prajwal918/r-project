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