# 📊 Evaluation Metrics & Code: Complete Explanation

## Part 1: Confusion Matrix

### **What is a Confusion Matrix?**

A table showing correct and incorrect predictions

### **Your Result:**

```
          ACTUAL REALITY
          No    Yes    (What really happened)
MODEL No  930   167    ← Model said "Won't churn"
SAYS Yes  104   206    ← Model said "Will churn"
```

### **Four Outcomes:**

#### **1. True Negative (TN) = 930** ✅
```
Model said: "Won't churn"
Reality: Didn't churn
Result: CORRECT!
```
**Meaning:** Model correctly identified 930 loyal customers

---

#### **2. False Negative (FN) = 167** ❌ **DANGEROUS**
```
Model said: "Won't churn"
Reality: DID CHURN
Result: ERROR - Missed churner!
```
**Meaning:** 167 customers will leave WITHOUT retention offer
**Cost:** 167 × $1,000 = $167,000 lost revenue
**Most costly error!**

---

#### **3. False Positive (FP) = 104** ⚠️
```
Model said: "Will churn"
Reality: Stayed loyal
Result: ERROR - False alarm
```
**Meaning:** Sent retention offer to loyal customer
**Cost:** 104 × $50 = $5,200 wasted
**Minor compared to missing churners**

---

#### **4. True Positive (TP) = 206** ✅ **VALUABLE**
```
Model said: "Will churn"
Reality: Did churn
Result: CORRECT!
```
**Meaning:** Caught 206 at-risk customers
**Value:** Can send retention offers and save them

---

## Part 2: All Metrics Explained

### **1. Accuracy = 80.74%**

**Formula:**
```
Accuracy = (Correct Predictions) / (Total)
         = (TP + TN) / (TP + TN + FP + FN)
         = (206 + 930) / 1407
         = 80.74%
```

**What it means:** Model got it right 8 out of 10 times

**⚠️ Problem:** Misleading for imbalanced data!

**Example of why accuracy is misleading:**
```
Dumb model: Always predict "No churn"
├─ TP: 0 (caught no churners)
├─ FP: 0
├─ FN: 373 (missed ALL churners)
├─ TN: 1034
└─ Accuracy: 1034/1407 = 73.5%

This terrible model still has 73.5% accuracy!
```

**Lesson:** Need other metrics for imbalanced data

---

### **2. Sensitivity (Recall) = 55.23%** ❌ **YOUR MAIN PROBLEM**

**Formula:**
```
Sensitivity = TP / (TP + FN)
            = TP / (All Actual Churners)
            = 206 / (206 + 167)
            = 206 / 373
            = 55.23%
```

**What it means:** "Of all customers who churned, how many did we catch?"

**Your result:**
```
373 customers churned:
├─ Caught: 206 (55%) ✅
└─ Missed: 167 (45%) ❌

Problem: Missing almost HALF of churners!
```

**Also called:**
- Recall
- True Positive Rate (TPR)
- Hit Rate

**Why it matters:** **MOST IMPORTANT metric for churn prediction!**
- The whole point is catching churners
- Low sensitivity = many churners escape

---

### **3. Specificity = 89.94%** ✅ **GOOD**

**Formula:**
```
Specificity = TN / (TN + FP)
            = TN / (All Actual Non-Churners)
            = 930 / (930 + 104)
            = 930 / 1034
            = 89.94%
```

**What it means:** "Of all loyal customers, how many did we correctly identify?"

**Your result:**
```
1,034 loyal customers:
├─ Correctly identified: 930 (90%) ✅
└─ False alarms: 104 (10%) ❌

Good: Rarely waste resources on loyal customers
```

**Also called:**
- True Negative Rate (TNR)
- Selectivity

---

### **4. Precision (PPV) = 66.45%**

**Formula:**
```
Precision = TP / (TP + FP)
          = TP / (All Predicted Churners)
          = 206 / (206 + 104)
          = 206 / 310
          = 66.45%
```

**What it means:** "When we predict churn, how often are we right?"

**Your result:**
```
Model predicted churn 310 times:
├─ Correct: 206 (66%) ✅
└─ Wrong: 104 (33%) ❌

About 1 in 3 retention campaigns target loyal customers
```

**Sensitivity vs Precision:**
```
Sensitivity: "Of actual churners, how many caught?"
Precision:   "Of predicted churners, how many real?"

Different perspectives!
```

---

### **5. F1-Score = 0.6032**

**Formula:**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
   = 2 × (0.6645 × 0.5523) / (0.6645 + 0.5523)
   = 0.6032
```

**What it means:** 
- Harmonic mean of precision and recall
- Balances both metrics
- Single score for model quality

**Scale:** 0 to 1 (higher is better)

---

### **6. Kappa = 0.4775** (Moderate)

**What it means:** Agreement beyond chance

**Scale:**
- < 0.40: Poor
- 0.40 - 0.60: Moderate ← **Your model**
- 0.60 - 0.80: Substantial
- 0.80 - 1.00: Excellent

**Your result:** Moderately better than random guessing

---

### **7. Balanced Accuracy = 72.58%**

**Formula:**
```
Balanced Accuracy = (Sensitivity + Specificity) / 2
                  = (55.23% + 89.94%) / 2
                  = 72.58%
```

**What it means:** Average performance on both classes

**Better than raw accuracy for imbalanced data**

---

## Part 3: Threshold Explained

### **What is a Threshold?**

**Definition:** The probability cutoff for classifying as "Churn"

### **How It Works:**

**Step 1:** Model calculates probabilities
```
Customer A: 72% chance of churn
Customer B: 45% chance of churn
Customer C: 28% chance of churn
```

**Step 2:** Compare to threshold
```
Threshold = 0.5 (50%)

Customer A: 72% > 50% → Predict "Yes" (Churn)
Customer B: 45% < 50% → Predict "No" (Stay)
Customer C: 28% < 50% → Predict "No" (Stay)
```

### **In Your Code:**

```r
log_pred_class <- factor(ifelse(log_predictions > 0.5, "Yes", "No"))
                                                   ↑
                                              Threshold
```

### **Changing the Threshold:**

```
Lower threshold (0.35):
├─ More aggressive
├─ Catches MORE churners (↑ sensitivity)
└─ More false alarms (↓ specificity)

Higher threshold (0.6):
├─ More conservative
├─ Catches FEWER churners (↓ sensitivity)
└─ Fewer false alarms (↑ specificity)
```

### **Your Three Tests:**

| Threshold | Sensitivity | Specificity | Churners Caught |
|-----------|-------------|-------------|-----------------|
| 0.50 | 55% | 90% | 206 |
| 0.40 | 65% | 83% | 242 |
| 0.35 | 69% | 79% | 257 |

**Pattern:** Lower threshold → More churners caught

---

## Part 4: Complete Code Walkthrough

### **Section 1: Data Loading & Cleaning (Lines 1-36)**

```r
# Line 5-8: Load required packages
library(tidyverse)   # Data manipulation
library(ggplot2)     # Visualization
library(caret)       # Model evaluation
library(randomForest) # Random Forest model
```

**What each package does:**
- `tidyverse`: Tools for data cleaning (mutate, select, filter)
- `ggplot2`: Create charts and graphs
- `caret`: Confusion matrix and data splitting
- `randomForest`: Train Random Forest models

---

```r
# Line 15: Load data
churn_data <- read.csv("WA_Fn-UseC_-Telco-Customer-Churn.xls")
```

**What it does:** Reads CSV file into R
**Result:** Table with 7,043 rows, 21 columns

---

```r
# Lines 20-23: Handle missing values
churn_data <- churn_data %>%
  mutate(TotalCharges = as.numeric(TotalCharges)) %>%
  mutate(TotalCharges = ifelse(is.na(TotalCharges), 0, TotalCharges)) %>%
  select(-customerID)
```

**Line by line:**
1. `mutate(TotalCharges = as.numeric(...))`: Convert to numbers
2. `mutate(TotalCharges = ifelse(is.na(...), 0, ...))`: Replace NA with 0
3. `select(-customerID)`: Remove customer ID column

**Why:** New customers have no total charges yet (tenure=0)

---

```r
# Lines 27-31: Simplify categories
churn_data <- churn_data %>%
  mutate(across(c(OnlineSecurity, OnlineBackup, DeviceProtection, 
                  TechSupport, StreamingTV, StreamingMovies),
                ~ recode(., "No internet service" = "No"))) %>%
  mutate(MultipleLines = recode(MultipleLines, "No phone service" = "No"))
```

**What it does:** Combines categories
- "No internet service" → "No"
- "No phone service" → "No"

**Why:** Simpler model, same meaning

---

```r
# Lines 34-35: Convert text to factors
churn_data <- churn_data %>%
  mutate(across(where(is.character), as.factor))
```

**What it does:** Converts all text columns to categorical factors
**Why:** Models need factors, not text

---

### **Section 2: Data Exploration (Lines 37-71)**

```r
# Lines 39-42: Calculate churn rate
churn_rate <- churn_data %>%
  count(Churn) %>%
  mutate(Proportion = n / sum(n))
print(churn_rate)
```

**What it does:** 
- Counts "Yes" and "No" in Churn column
- Calculates percentage
**Result:** ~73.5% No, ~26.5% Yes

---

```r
# Lines 45-50: Bar chart by contract type
ggplot(churn_data, aes(x = Contract, fill = Churn)) +
  geom_bar(position = "dodge") +
  labs(title = "Customer Churn by Contract Type",
       x = "Contract Type",
       y = "Number of Customers") +
  theme_minimal()
```

**What it creates:** Bar chart showing churn by contract
**Insight:** Month-to-month contracts have higher churn

---

```r
# Lines 69-71: Correlation matrix
numeric_data <- churn_data %>% select(tenure, MonthlyCharges, TotalCharges)
correlation_matrix <- cor(numeric_data)
print(correlation_matrix)
```

**What it does:** Shows relationships between numeric features
**Example:** Tenure and TotalCharges are highly correlated (long customers pay more)

---

### **Section 3: Train/Test Split (Lines 73-80)**

```r
# Line 74: Set random seed
set.seed(123)
```
**What it does:** Ensures reproducibility
**Why:** Same split every time you run code

---

```r
# Line 76: Ensure factor
churn_data$Churn <- as.factor(churn_data$Churn)
```
**What it does:** Makes sure Churn is categorical
**Why:** Required for classification

---

```r
# Line 78: Create split indices
trainIndex <- createDataPartition(churn_data$Churn, p = 0.8, list = FALSE, times = 1)
```

**Breaking it down:**
- `churn_data$Churn`: Stratify by target variable
- `p = 0.8`: 80% for training
- `list = FALSE`: Return vector
- `times = 1`: Create one split

**Result:** Vector of row numbers for training set

---

```r
# Lines 79-80: Create datasets
train_data <- churn_data[trainIndex, ]   # Select training rows
test_data  <- churn_data[-trainIndex, ]  # Select other rows (test)
```

**Result:**
- `train_data`: 5,634 customers (80%)
- `test_data`: 1,409 customers (20%)

---

### **Section 4: Model Training (Lines 82-86)**

```r
# Line 84: Train logistic regression
log_model <- glm(Churn ~ ., data = train_data, family = binomial)
```

**What happens:**
1. Creates logistic regression model
2. Predicts Churn using all features (`.`)
3. Uses training data
4. `family = binomial`: Binary classification

**Result:** Trained model stored in `log_model`

---

```r
# Line 86: Train random forest
rf_model <- randomForest(Churn ~ ., data = train_data)
```

**What happens:**
1. Creates 500 decision trees
2. Each tree trained on random data sample
3. Each split uses random features
4. Saves all trees

**Result:** Trained model with 500 trees

---

### **Section 5: Model Evaluation (Lines 88-100)**

```r
# Lines 90-92: Logistic regression predictions
log_predictions <- predict(log_model, test_data, type = "response")
log_pred_class <- factor(ifelse(log_predictions > 0.5, "Yes", "No"),
                         levels = levels(test_data$Churn))
```

**Line by line:**
1. `predict(...)`: Get probabilities for test data
2. `ifelse(... > 0.5, "Yes", "No")`: Convert to Yes/No (threshold = 0.5)
3. `factor(...)`: Make sure format matches test data

---

```r
# Lines 93-94: Create confusion matrix
cm_log <- confusionMatrix(log_pred_class, test_data$Churn, positive = "Yes")
print(cm_log)
```

**What it does:**
- Compares predictions to actual values
- Calculates all metrics (accuracy, sensitivity, etc.)
- `positive = "Yes"`: Treats "Yes" (churn) as positive class

**Output:** Full confusion matrix with statistics

---

### **Section 6: Threshold Optimization (Lines 122-215)**

```r
# Lines 126-145: Test different thresholds
thresholds <- c(0.3, 0.35, 0.4, 0.45, 0.5)
threshold_results <- data.frame()

for (thresh in thresholds) {
  pred_class <- factor(ifelse(log_predictions > thresh, "Yes", "No"),
                       levels = levels(test_data$Churn))
  cm <- confusionMatrix(pred_class, test_data$Churn, positive = "Yes")
  
  threshold_results <- rbind(threshold_results, data.frame(
    Threshold = thresh,
    Accuracy = cm$overall['Accuracy'],
    Sensitivity = cm$byClass['Sensitivity'],
    Specificity = cm$byClass['Specificity'],
    ...
  ))
}
```

**What it does:**
1. Tests 5 different thresholds
2. For each threshold:
   - Makes predictions
   - Calculates metrics
   - Stores results
3. Creates comparison table

**Result:** Table comparing all thresholds

---

```r
# Lines 170: Set optimal threshold
optimal_threshold <- 0.4  # <-- CHANGE THIS
```

**What you can change:** This number (0.3 to 0.5)
**Effect:** Lower = catch more churners, more false alarms

---

## Part 5: Key Takeaways

### **Your Model Performance:**

**Original (Threshold 0.5):**
- ❌ Only catches 55% of churners
- ✅ Very few false alarms (10%)
- 💡 TOO CONSERVATIVE

**Optimized (Threshold 0.35-0.4):**
- ✅ Catches 65-69% of churners
- ⚠️ More false alarms (15-21%)
- 💰 Much better business value

### **Best Model:** Logistic Regression (not Random Forest)

**Why:**
1. Better sensitivity (catches more churners)
2. Interpretable (can explain predictions)
3. Faster
4. Easy to adjust threshold

### **Recommended Threshold:** 0.35 or 0.4

**Why:**
- Catches 65-69% of churners
- Acceptable false alarm rate
- Highest business value ($90k-117k)

---

## Summary: Every Concept in One Page

1. **Machine Learning:** Teaching computers to predict from examples
2. **Churn:** Customers leaving your service
3. **Classification:** Predicting categories (Yes/No)
4. **Features:** Input variables (gender, charges, tenure, etc.)
5. **Target:** What you predict (Churn: Yes/No)
6. **Train/Test Split:** 80% learn, 20% evaluate
7. **Logistic Regression:** Formula-based probability model ⭐ **BEST**
8. **Random Forest:** 500 trees voting together
9. **Confusion Matrix:** Table of correct/incorrect predictions
10. **Sensitivity:** % of churners caught (MOST IMPORTANT)
11. **Specificity:** % of loyal correctly identified
12. **Precision:** When predict churn, % correct
13. **Threshold:** Probability cutoff for classification
14. **Lower Threshold:** Catch more churners, more false alarms
15. **Your Goal:** Maximize sensitivity while keeping acceptable precision

**Bottom Line:** Use Logistic Regression with threshold 0.35-0.4 to catch 65-69% of churners with good business value!
