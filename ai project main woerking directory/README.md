# Telco Customer Churn: How to Run Everything

This project contains:
- `er.R`: End-to-end script for data cleaning, EDA, train/test split, Logistic Regression, Random Forest, and Linear Probability Model (Linear Regression on churn).
- `report.Rmd`: Fully documented R Markdown report with deeper analysis (odds ratios, ROC/AUC, variable importance, and LPM).
- Data file expected in the same folder: `WA_Fn-UseC_-Telco-Customer-Churn.xls`

## 1) Prerequisites

- R (>= 4.2) and RStudio (optional, recommended)
- Internet access to install packages from CRAN

Install required packages once in the R Console:

```r
install.packages(c(
  "tidyverse", "ggplot2", "caret", "randomForest", "e1071", # core
  "rmarkdown",                   # to knit the report
  "pROC", "broom"               # extras used in report.Rmd
))
# If your data file is truly Excel (.xls) format:
# install.packages("readxl")
```

## 2) Confirm the dataset location

Place the dataset file in the same directory as this README:

```
/home/prajwalj/Desktop/ai project main woerking directory/WA_Fn-UseC_-Telco-Customer-Churn.xls
```

The script currently loads via `read.csv`. If your file is actually Excel, switch the loader (see notes below).

## 3) Run the main script (er.R)

### Option A: From RStudio
- Open `er.R` in RStudio.
- Set working directory: Session > Set Working Directory > To Source File Location.
- Click "Source" (or press Ctrl+Shift+Enter) to run all.

### Option B: From a terminal

```bash
Rscript "/home/prajwalj/Desktop/ai project main woerking directory/er.R"
```

### What the script does
- Cleans and prepares the data (recodes, converts to factors).
- EDA: churn rate, bar charts, box plot, simple correlation matrix.
- Splits data into 80/20 train/test.
- Trains:
  - Logistic Regression: `glm(Churn ~ ., family = binomial)`
  - Random Forest: `randomForest(Churn ~ .)`
  - Linear Probability Model (LPM): `lm(Churn_num ~ . - Churn)`
- Evaluates:
  - Logistic & RF: `confusionMatrix` (Accuracy, Sensitivity, Specificity)
  - LPM: RMSE, proportion of predictions outside [0,1], and a 0.5-threshold confusion matrix

### Outputs to expect (Console)
- Churn rate table and correlation matrix.
- `summary(log_model)` and RF model summary.
- Confusion matrices for Logistic and RF.
- LPM summary, RMSE, out-of-bounds rate, and LPM confusion matrix.

## 4) Run specific models manually (optional)

If you want to run individual steps in the R Console (after sourcing the cleaning and split sections):

```r
# Logistic Regression
log_model <- glm(Churn ~ ., data = train_data, family = binomial)
log_predictions <- predict(log_model, test_data, type = "response")
log_pred_class <- factor(ifelse(log_predictions > 0.5, "Yes", "No"),
                         levels = levels(test_data$Churn))
confusionMatrix(log_pred_class, test_data$Churn, positive = "Yes")

# Random Forest
rf_model <- randomForest(Churn ~ ., data = train_data)
rf_predictions <- predict(rf_model, test_data)
rf_predictions <- factor(rf_predictions, levels = levels(test_data$Churn))
confusionMatrix(rf_predictions, test_data$Churn, positive = "Yes")

# Linear Probability Model (LPM)
train_lpm <- train_data %>% mutate(Churn_num = if_else(Churn == "Yes", 1, 0))
test_lpm  <- test_data  %>% mutate(Churn_num = if_else(Churn == "Yes", 1, 0))
lpm_model <- lm(Churn_num ~ . - Churn, data = train_lpm)
lpm_pred_num <- predict(lpm_model, newdata = test_lpm)
rmse <- sqrt(mean((lpm_pred_num - test_lpm$Churn_num)^2))
prop_out_of_bounds <- mean(lpm_pred_num < 0 | lpm_pred_num > 1)
list(RMSE = rmse, Proportion_Out_Of_Bounds = prop_out_of_bounds)
lpm_pred_class <- factor(ifelse(lpm_pred_num > 0.5, "Yes", "No"),
                         levels = levels(test_data$Churn))
confusionMatrix(lpm_pred_class, test_data$Churn, positive = "Yes")
```

## 5) Knit the detailed report (report.Rmd)

### Run from RStudio
- Open `report.Rmd`.
- Set working directory: Session > Set Working Directory > To Source File Location.
- Click Knit → HTML (recommended).

### Or render from Console
```r
setwd("/home/prajwalj/Desktop/ai project main woerking directory")
rmarkdown::render("report.Rmd", output_format = "html_document")
```

The report includes:
- Odds ratios for Logistic Regression.
- ROC curves and AUC for Logistic and RF.
- Variable importance for RF, and coefficient magnitudes for Logistic.
- Linear Probability Model evaluation (RMSE, out-of-bounds rate, confusion matrix).

## 6) Excel vs CSV input (important)

Currently, both `er.R` and `report.Rmd` use:
```r
churn_data <- read.csv("WA_Fn-UseC_-Telco-Customer-Churn.xls")
```
If your file is truly Excel (.xls), switch to:
```r
library(readxl)
churn_data <- readxl::read_excel("WA_Fn-UseC_-Telco-Customer-Churn.xls")
```

## 7) Troubleshooting
- "Package not found": re-run the `install.packages(...)` block in Section 1.
- "File not found": verify the dataset path matches your working directory; use `getwd()`.
- Factor levels: The positive class is set to "Yes" in `confusionMatrix`. If your labels differ, adjust `positive = "Yes"` and factor level ordering.
- Class imbalance: Consider threshold tuning and ROC/AUC (see `report.Rmd`).

---

# How this works (explain to your sir)

## 1) End-to-end pipeline overview

```mermaid
flowchart TD
  A[Raw data: WA_Fn-UseC_-Telco-Customer-Churn.xls] --> B[Load data]
  B --> C[Clean & Recode]
  C --> D[Feature typing (factors)]
  D --> E[EDA: plots + correlations]
  E --> F[Train/Test split (80/20, stratified)]
  F --> G1[Train Logistic Regression]
  F --> G2[Train Random Forest]
  F --> G3[Train Linear Probability Model]
  G1 --> H[Predict on Test]
  G2 --> H
  G3 --> H
  H --> I[Evaluate: confusionMatrix, RMSE (LPM)]
  I --> J[Interpretation & Next steps]
```

## 2) Why we clean and recode
- **TotalCharges to numeric + NA to 0**: Some entries come as strings or blanks; converting ensures models can use the values. NA imputation avoids dropping rows.
- **Drop `customerID`**: It's an identifier with no predictive pattern; leaving it can leak noise.
- **Collapse "No internet/phone service" to "No"**: Reduces sparse categories and clarifies meaning (service not available = not using feature).
- **Convert text columns to factors**: Most ML algorithms in R expect categorical predictors to be factors.

## 3) Train/test split
- **Goal**: Measure generalization on unseen data.
- **Method**: `caret::createDataPartition()` with `p = 0.8` keeps the same churn proportion (stratification) in train and test.

## 4) What each model does

- **Logistic Regression** (`glm(..., family = binomial)`)
  - Estimates log-odds of churn via a linear combination of features.
  - Uses the sigmoid function to convert to probabilities in [0,1].
  - Coefficients indicate direction and strength (we provide Odds Ratios in `report.Rmd`).

- **Random Forest** (`randomForest`)
  - Ensemble of decision trees trained on bootstrapped samples and random feature subsets.
  - Captures non-linearities and interactions automatically.
  - Provides out-of-bag (OOB) error and feature importance (visualized in `report.Rmd`).

- **Linear Probability Model (LPM)** (`lm` on 0/1 target)
  - Simple linear regression treating churn as numeric 0/1.
  - Easy to interpret coefficients, but predictions can fall outside [0,1].
  - We report RMSE and the proportion of out-of-bounds predictions, plus 0.5 threshold classification for comparison.

## 5) How predictions become classes
- **Thresholding**: Probabilities > 0.5 become "Yes" churn; otherwise "No".
- **Note**: 0.5 is a default. If the business needs higher recall (catch more churners), lower the threshold (e.g., 0.35) and re-check precision/recall.

## 6) How we evaluate performance
- **Confusion matrix** (for Logistic and RF):
  - Accuracy = overall correctness.
  - Sensitivity (Recall for "Yes") = how many churners we caught.
  - Specificity = how well we avoid false churn flags.
  - We declare the positive class as `"Yes"` for clarity.
- **LPM**: RMSE (average error on 0/1 target) and out-of-bounds rate; we also show a confusion matrix at a 0.5 cutoff for comparison.
- **Advanced (in report)**: ROC curves and AUC show trade-offs across all thresholds and overall separability.

## 7) Class imbalance and business trade-offs
- Churners are fewer than non-churners (~26% churn). High accuracy can hide low recall.
- Choose a threshold and model that match objectives:
  - Retention teams often prefer higher recall (catch more churners), accepting more false positives.
  - Finance might prefer higher precision (fewer false alarms) to control outreach costs.

## 8) Reproducibility
- We set a random seed (`set.seed(123)`) before splitting to make results repeatable.
- All dependencies are listed in this README and used in the scripts.
- Full, reproducible workflow is also available in `report.Rmd` (knit to HTML/PDF/Word).

## 9) Talking points for your explanation
- "We cleaned the dataset to make it consistent and usable for models, converting text to categories and fixing numeric columns."
- "We explored key patterns: month-to-month contracts and higher monthly charges correlate with churn; fiber customers show higher churn."
- "We split data into train/test to evaluate on unseen customers."
- "We trained two classifiers: Logistic Regression (interpretable) and Random Forest (powerful, non-linear). We also added a Linear Probability Model for comparison."
- "We measured performance with confusion matrices and, in the report, ROC/AUC. Accuracy is ~80%, but sensitivity is lower due to class imbalance—this is a typical trade-off."
- "Depending on the business goal, we can adjust the probability threshold to catch more churners or reduce false alarms."
