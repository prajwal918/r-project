# 📚 Complete Project Concepts: Beginner's Guide

## 🎯 What is This Project?

### **Goal:** Predict which customers will leave (churn) before they do

### **Why?**
- Losing customers costs money ($1,000 per customer)
- Keeping existing customers is cheaper than finding new ones
- Can send retention offers to at-risk customers

### **Project Type:** Binary Classification Machine Learning

---

## 🧠 Key Concepts Explained

### **1. Machine Learning**

**Simple Definition:** Teaching computers to make predictions by learning from examples

**Traditional vs ML:**
```
Traditional Programming:
You write rules → Computer follows → Output

Machine Learning:  
Give examples → Computer learns patterns → Can predict new cases
```

**Example:**
```
Traditional: IF temperature > 38°C THEN fever
ML: Show 1000 examples → Learn pattern → Predict new patient
```

---

### **2. Churn**

**Definition:** When a customer stops using your service

**Examples:** Cancel subscription, close account, switch to competitor

**Why it matters:** Acquiring new customers costs 5-25x more than keeping existing ones

---

### **3. Classification**

**Definition:** Predicting categories/labels

**Your Project:** Binary Classification = 2 categories (Churn: Yes or No)

**Other Examples:**
- Email: Spam or Not Spam
- Transaction: Fraud or Legitimate
- Patient: Sick or Healthy

---

### **4. Supervised Learning**

**Definition:** Learning from labeled examples

**How it works:**
1. Give computer examples with answers (features + labels)
2. Computer learns patterns
3. Computer predicts new examples

**Your Data:**
- Features: Gender, contract type, charges, tenure, etc.
- Labels: Did they churn? (Yes/No)

---

### **5. Features (X) and Target (Y)**

**Features (Input):** Variables used to make predictions
- Gender, SeniorCitizen, Partner, Tenure, MonthlyCharges, Contract, etc.
- Think of these as "clues" to solve the mystery

**Target (Output):** What you're predicting
- Churn: Yes or No

---

### **6. Training vs Testing Data**

**Why split?**
```
If we test on data we trained on:
├─ Model memorized answers
└─ Not a fair test!

Solution:
├─ Train on 80% of data (model learns)
└─ Test on 20% of data (model never saw before)
```

**Analogy:** Like studying from textbook (train) then taking exam with new questions (test)

**Your Split:**
- Training: 5,634 customers (80%)
- Testing: 1,409 customers (20%)

---

## 📊 The Three Models Explained

### **Model 1: Logistic Regression** ⭐ **BEST FOR YOUR PROJECT**

**What it is:** Statistical model that calculates probability using a formula

**How it works:**
```
Probability = 1 / (1 + e^-(weighted sum of features))

Example:
P(Churn) = 1 / (1 + e^-(0.5×MonthToMonth + 0.3×HighCharges - 0.4×LongTenure))
```

**Advantages:**
- ✅ Fast
- ✅ Returns probabilities (0-100%)
- ✅ Interpretable (can see which features matter)
- ✅ Easy to adjust threshold

**Disadvantages:**
- ❌ Assumes linear relationships
- ❌ Can't capture complex patterns

**When to use:** When you need interpretable, fast predictions

---

### **Model 2: Random Forest**

**What it is:** 500 decision trees voting together

**How it works:**
```
1. Create 500 different decision trees
2. Each tree makes a prediction
3. Final prediction = majority vote

Example:
- 320 trees say "Churn"
- 180 trees say "No Churn"
- Result: "Churn" (majority wins)
```

**Decision Tree Example:**
```
Is MonthlyCharges > $70?
├─ Yes → Is Contract Month-to-Month?
│        ├─ Yes → Predict CHURN
│        └─ No → Predict STAY
└─ No → Predict STAY
```

**Advantages:**
- ✅ Handles complex patterns
- ✅ No data preparation needed
- ✅ Feature importance available

**Disadvantages:**
- ❌ Slower
- ❌ Black box (hard to explain why)
- ❌ Doesn't perform better for your data

**When to use:** Complex non-linear relationships

---

### **Model 3: Linear Regression (LPM)**

**What it is:** Straight line formula

**Formula:** Y = a + b₁X₁ + b₂X₂ + ...

**Problem for classification:**
Can predict impossible values:
- Customer A: 1.2 (120% probability?) ❌
- Customer B: -0.3 (-30% probability?) ❌

**Why included:** Educational comparison to show why logistic regression is better

**When to use:** DON'T use for classification! Use logistic regression instead

---

## 🔧 Data Preparation Explained

### **Step 1: Loading Data**

```r
churn_data <- read.csv("WA_Fn-UseC_-Telco-Customer-Churn.xls")
```

Reads CSV file into R as a table (data frame)
- 7,043 rows (customers)
- 21 columns (features + target)

---

### **Step 2: Handling Missing Values**

**What are missing values?** Empty cells (NA, null, blank)

**Your code:**
```r
mutate(TotalCharges = ifelse(is.na(TotalCharges), 0, TotalCharges))
```

**What it does:** Replaces missing TotalCharges with 0
**Reason:** New customers (tenure=0) haven't paid anything yet

**Alternatives:**
- Delete rows (lose data)
- Use average value
- Predict using other features

---

### **Step 3: Simplifying Categories**

```r
mutate(across(c(OnlineSecurity, OnlineBackup, ...), 
              ~ recode(., "No internet service" = "No")))
```

**What it does:**
```
Before: OnlineSecurity = "Yes", "No", "No internet service"
After:  OnlineSecurity = "Yes", "No"

Logic: "No internet" same as "No security"
```

**Why:** Simpler categories = simpler model

---

### **Step 4: Converting to Factors**

```r
mutate(across(where(is.character), as.factor))
```

**What it does:** Converts text to categories (factors)

**Why needed:**
```
Text "Yes", "No":
└─ Model sees: random text ❌

Factor ["Yes", "No"]:
└─ Model sees: Category 1, Category 2 ✅
```

---

### **Step 5: Removing Customer ID**

```r
select(-customerID)
```

**Why:** Each customer has unique ID (no predictive value)
**Problem if kept:** Model would memorize IDs instead of learning patterns

---

### **Step 6: Train/Test Split**

```r
set.seed(123)  # For reproducibility
trainIndex <- createDataPartition(churn_data$Churn, p = 0.8, list = FALSE)
train_data <- churn_data[trainIndex, ]   # 80%
test_data <- churn_data[-trainIndex, ]   # 20%
```

**`set.seed(123)`:** Ensures same split every time you run code

**`createDataPartition`:**
- `p = 0.8`: 80% for training
- Stratifies by Churn (keeps same churn rate in both sets)

**Why stratify?**
```
Without: Train might have 30% churn, Test 20% churn ❌
With:    Both have 26% churn ✅
```

---

## 🎓 Model Training Explained

### **Logistic Regression**

```r
log_model <- glm(Churn ~ ., data = train_data, family = binomial)
```

**Breaking it down:**
- `glm`: Generalized Linear Model
- `Churn ~ .`: Predict Churn using all other features
- `data = train_data`: Learn from training data only
- `family = binomial`: Binary outcome (Yes/No)

**What happens:**
1. Starts with random weights
2. Calculates probabilities for each customer
3. Compares to actual results
4. Adjusts weights to reduce errors
5. Repeats until optimized

---

### **Random Forest**

```r
rf_model <- randomForest(Churn ~ ., data = train_data)
```

**What happens:**
1. Creates 500 decision trees
2. Each tree:
   - Uses random subset of customers
   - Uses random subset of features
   - Makes predictions
3. Final prediction = majority vote

---

### **Linear Regression**

```r
train_lpm <- train_data %>% mutate(Churn_num = if_else(Churn == "Yes", 1, 0))
lpm_model <- lm(Churn_num ~ . - Churn, data = train_lpm)
```

**What it does:**
- Converts Churn to numbers (Yes=1, No=0)
- Fits straight line through data
- Problem: Can predict <0 or >1

---

## 📈 Evaluation Metrics Explained
