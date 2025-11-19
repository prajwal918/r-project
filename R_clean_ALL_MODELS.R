library(tidyverse)
library(ggplot2)
library(caret)
library(randomForest)
library(glmnet)
library(class)
library(rpart)
library(rpart.plot)
library(e1071)
library(gbm)
library(xgboost)
library(nnet)
library(pROC)
library(keras)
library(tensorflow)
library(keras3)

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

churn_rate <- churn_data %>%
  count(Churn) %>%
  mutate(Proportion = n / sum(n))
print(churn_rate)

ggplot(churn_data, aes(x = Contract, fill = Churn)) +
  geom_bar(position = "dodge") +
  labs(title = "Customer Churn by Contract Type",
       x = "Contract Type",
       y = "Number of Customers") +
  theme_minimal()

ggplot(churn_data, aes(x = InternetService, fill = Churn)) +
  geom_bar(position = "dodge") +
  labs(title = "Customer Churn by Internet Service Type",
       x = "Internet Service",
       y = "Number of Customers") +
  theme_minimal()

ggplot(churn_data, aes(x = Churn, y = MonthlyCharges, fill = Churn)) +
  geom_boxplot() +
  labs(title = "Monthly Charges by Churn Status",
       x = "Churn Status",
       y = "Monthly Charges ($)") +
  theme_minimal()

numeric_data <- churn_data %>% select(tenure, MonthlyCharges, TotalCharges)
correlation_matrix <- cor(numeric_data)
print(correlation_matrix)

set.seed(123)
churn_data$Churn <- as.factor(churn_data$Churn)

trainIndex <- createDataPartition(churn_data$Churn, p = 0.8, list = FALSE, times = 1)
train_data <- churn_data[trainIndex, ]
test_data  <- churn_data[-trainIndex, ]

log_model <- glm(Churn ~ ., data = train_data, family = binomial)
rf_model <- randomForest(Churn ~ ., data = train_data)

log_predictions <- predict(log_model, test_data, type = "response")
log_pred_class <- factor(ifelse(log_predictions > 0.5, "Yes", "No"),
                         levels = levels(test_data$Churn))
cm_log <- confusionMatrix(log_pred_class, test_data$Churn, positive = "Yes")
print(cm_log)
roc_log <- roc(test_data$Churn, log_predictions)
auc_log <- auc(roc_log)

rf_predictions <- predict(rf_model, test_data)
rf_predictions <- factor(rf_predictions, levels = levels(test_data$Churn))
rf_predictions_prob <- predict(rf_model, test_data, type = "prob")[,2]
cm_rf <- confusionMatrix(rf_predictions, test_data$Churn, positive = "Yes")
print(cm_rf)
roc_rf <- roc(test_data$Churn, rf_predictions_prob)
auc_rf <- auc(roc_rf)

cat("\n=== k-Nearest Neighbors (k-NN) ===\n")
preproc <- preProcess(train_data[, -which(names(train_data) == "Churn")], method = c("center", "scale"))
train_scaled <- predict(preproc, train_data)
test_scaled <- predict(preproc, test_data)

set.seed(123)
knn_model <- train(Churn ~ ., data = train_scaled, method = "knn",
                   tuneGrid = data.frame(k = 5),
                   trControl = trainControl(method = "none"))
knn_predictions <- predict(knn_model, test_scaled)
knn_predictions_prob <- predict(knn_model, test_scaled, type = "prob")[,2]
cm_knn <- confusionMatrix(knn_predictions, test_data$Churn, positive = "Yes")
print(cm_knn)
roc_knn <- roc(test_data$Churn, knn_predictions_prob)
auc_knn <- auc(roc_knn)

cat("\n=== Decision Tree ===\n")
set.seed(123)
tree_model <- rpart(Churn ~ ., data = train_data, method = "class",
                    control = rpart.control(cp = 0.01))
rpart.plot(tree_model, main = "Decision Tree for Churn Prediction")

tree_predictions <- predict(tree_model, test_data, type = "class")
tree_predictions_prob <- predict(tree_model, test_data, type = "prob")[,2]
cm_tree <- confusionMatrix(tree_predictions, test_data$Churn, positive = "Yes")
print(cm_tree)
roc_tree <- roc(test_data$Churn, tree_predictions_prob)
auc_tree <- auc(roc_tree)

cat("\n=== Naive Bayes ===\n")
set.seed(123)
nb_model <- naiveBayes(Churn ~ ., data = train_data)

nb_predictions <- predict(nb_model, test_data)
nb_predictions_prob <- predict(nb_model, test_data, type = "raw")[,2]
cm_nb <- confusionMatrix(nb_predictions, test_data$Churn, positive = "Yes")
print(cm_nb)
roc_nb <- roc(test_data$Churn, nb_predictions_prob)
auc_nb <- auc(roc_nb)

cat("\n=== Gradient Boosting Machine (GBM) ===\n")
set.seed(123)
train_gbm <- train_data %>% mutate(Churn_num = ifelse(Churn == "Yes", 1, 0))
test_gbm <- test_data %>% mutate(Churn_num = ifelse(Churn == "Yes", 1, 0))

gbm_model <- gbm(Churn_num ~ . - Churn, data = train_gbm,
                 distribution = "bernoulli",
                 n.trees = 100,
                 interaction.depth = 3,
                 shrinkage = 0.1,
                 cv.folds = 5,
                 verbose = FALSE)

best_iter <- gbm.perf(gbm_model, method = "cv", plot.it = TRUE)
cat("Optimal number of trees:", best_iter, "\n")

gbm_pred_prob <- predict(gbm_model, test_gbm, n.trees = best_iter, type = "response")
gbm_pred_class <- factor(ifelse(gbm_pred_prob > 0.5, "Yes", "No"), levels = levels(test_data$Churn))
cm_gbm <- confusionMatrix(gbm_pred_class, test_data$Churn, positive = "Yes")
print(cm_gbm)
roc_gbm <- roc(test_data$Churn, gbm_pred_prob)
auc_gbm <- auc(roc_gbm)

cat("\n=== XGBoost ===\n")
set.seed(123)
x_train_xgb <- model.matrix(Churn ~ ., data = train_data)[, -1]
x_test_xgb <- model.matrix(Churn ~ ., data = test_data)[, -1]
y_train_xgb <- ifelse(train_data$Churn == "Yes", 1, 0)
y_test_xgb <- ifelse(test_data$Churn == "Yes", 1, 0)

dtrain <- xgb.DMatrix(data = x_train_xgb, label = y_train_xgb)
dtest <- xgb.DMatrix(data = x_test_xgb, label = y_test_xgb)

xgb_params <- list(
  objective = "binary:logistic",
  eval_metric = "logloss",
  max_depth = 3,
  eta = 0.1,
  subsample = 0.8,
  colsample_bytree = 0.8
)

xgb_cv <- xgb.cv(params = xgb_params, data = dtrain, nrounds = 100,
                 nfold = 5, early_stopping_rounds = 10, verbose = 0)
best_nrounds <- xgb_cv$best_iteration
cat("Optimal number of rounds:", best_nrounds, "\n")

xgb_model <- xgb.train(params = xgb_params, data = dtrain, nrounds = best_nrounds, verbose = 0)

importance_matrix <- xgb.importance(model = xgb_model)
print(importance_matrix)
xgb.plot.importance(importance_matrix, top_n = 10, main = "XGBoost Feature Importance")

xgb_pred_prob <- predict(xgb_model, dtest)
xgb_pred_class <- factor(ifelse(xgb_pred_prob > 0.5, "Yes", "No"), levels = levels(test_data$Churn))
cm_xgb <- confusionMatrix(xgb_pred_class, test_data$Churn, positive = "Yes")
print(cm_xgb)
roc_xgb <- roc(test_data$Churn, xgb_pred_prob)
auc_xgb <- auc(roc_xgb)

cat("\n=== Neural Network (ANN) with Dropout ===\n")
set.seed(123)

x_train_nn <- model.matrix(Churn ~ ., data = train_data)[, -1]
x_test_nn <- model.matrix(Churn ~ ., data = test_data)[, -1]
y_train_nn <- ifelse(train_data$Churn == "Yes", 1, 0)
y_test_nn <- ifelse(test_data$Churn == "Yes", 1, 0)

nn_model <- keras_model_sequential() %>%
  layer_dense(units = 64, activation = 'relu', input_shape = ncol(x_train_nn)) %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 32, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 1, activation = 'sigmoid')

nn_model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = optimizer_adam(learning_rate = 0.001),
  metrics = c('accuracy')
)

history <- nn_model %>% fit(
  x_train_nn, y_train_nn,
  epochs = 50,
  batch_size = 32,
  validation_split = 0.2,
  verbose = 0
)

nn_predictions_prob <- as.vector(nn_model %>% predict(x_test_nn))
nn_pred_class <- factor(ifelse(nn_predictions_prob > 0.5, "Yes", "No"), levels = levels(test_data$Churn))
cm_nn <- confusionMatrix(nn_pred_class, test_data$Churn, positive = "Yes")
print(cm_nn)
roc_nn <- roc(test_data$Churn, nn_predictions_prob)
auc_nn <- auc(roc_nn)

cat("\n=== Linear Probability Model (LPM) ===\n")
set.seed(123)
train_lpm <- train_data %>% mutate(Churn_num = if_else(Churn == "Yes", 1, 0))
test_lpm  <- test_data  %>% mutate(Churn_num = if_else(Churn == "Yes", 1, 0))

lpm_model <- lm(Churn_num ~ . - Churn, data = train_lpm)
print(summary(lpm_model))

lpm_pred_num <- predict(lpm_model, newdata = test_lpm)
lpm_pred_class <- factor(ifelse(lpm_pred_num > 0.5, "Yes", "No"), levels = levels(test_data$Churn))
cm_lpm <- confusionMatrix(lpm_pred_class, test_data$Churn, positive = "Yes")
print(cm_lpm)
roc_lpm <- roc(test_data$Churn, lpm_pred_num)
auc_lpm <- auc(roc_lpm)

cat("\n=== Polynomial Regression ===\n")
set.seed(123)
train_poly <- train_data %>% mutate(Churn_num = if_else(Churn == "Yes", 1, 0))
test_poly  <- test_data  %>% mutate(Churn_num = if_else(Churn == "Yes", 1, 0))

poly_model <- lm(Churn_num ~ poly(tenure, 2) + poly(MonthlyCharges, 2) + poly(TotalCharges, 2) + 
                 Contract + InternetService + PaymentMethod + OnlineSecurity + TechSupport + 
                 gender + SeniorCitizen + Partner + Dependents + PhoneService + MultipleLines + 
                 OnlineBackup + DeviceProtection + StreamingTV + StreamingMovies + PaperlessBilling, 
                 data = train_poly)
print(summary(poly_model))

poly_pred_num <- predict(poly_model, newdata = test_poly)
poly_pred_class <- factor(ifelse(poly_pred_num > 0.5, "Yes", "No"), levels = levels(test_data$Churn))
cm_poly <- confusionMatrix(poly_pred_class, test_data$Churn, positive = "Yes")
print(cm_poly)
roc_poly <- roc(test_data$Churn, poly_pred_num)
auc_poly <- auc(roc_poly)

cat("\n=== ROC CURVES FOR ALL MODELS ===\n")
plot(roc_log, col = "red", main = "ROC Curves - All 11 Models", lwd = 2)
plot(roc_knn, col = "blue", add = TRUE, lwd = 2)
plot(roc_tree, col = "green", add = TRUE, lwd = 2)
plot(roc_nb, col = "purple", add = TRUE, lwd = 2)
plot(roc_svm, col = "orange", add = TRUE, lwd = 2)
plot(roc_rf, col = "brown", add = TRUE, lwd = 2)
plot(roc_gbm, col = "pink", add = TRUE, lwd = 2)
plot(roc_xgb, col = "cyan", add = TRUE, lwd = 2)
plot(roc_nn, col = "darkgreen", add = TRUE, lwd = 2)
plot(roc_lpm, col = "black", add = TRUE, lwd = 2)
plot(roc_poly, col = "gray", add = TRUE, lwd = 2)

legend("bottomright", 
       legend = c(paste0("Logistic Regression (AUC=", round(auc_log, 3), ")"),
                  paste0("k-NN (AUC=", round(auc_knn, 3), ")"),
                  paste0("Decision Tree (AUC=", round(auc_tree, 3), ")"),
                  paste0("Naive Bayes (AUC=", round(auc_nb, 3), ")"),
                  paste0("SVM (AUC=", round(auc_svm, 3), ")"),
                  paste0("Random Forest (AUC=", round(auc_rf, 3), ")"),
                  paste0("GBM (AUC=", round(auc_gbm, 3), ")"),
                  paste0("XGBoost (AUC=", round(auc_xgb, 3), ")"),
                  paste0("Neural Network (AUC=", round(auc_nn, 3), ")"),
                  paste0("LPM (AUC=", round(auc_lpm, 3), ")"),
                  paste0("Polynomial (AUC=", round(auc_poly, 3), ")")),
       col = c("red", "blue", "green", "purple", "orange", "brown", "pink", "cyan", "darkgreen", "black", "gray"),
       lwd = 2, cex = 0.6)
