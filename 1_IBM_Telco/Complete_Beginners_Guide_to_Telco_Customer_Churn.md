# Complete Beginner's Guide to Telco Customer Churn

## 1. The Business Problem: Why Do Customers Leave?
"Churn" is when a customer cancels their subscription. For a Telco company, losing a customer is expensive—it costs much more to find a new customer than to keep an old one.

**Our Goal**: Use data to predict *who* will leave before they actually do, so the company can make them an offer to stay.

## 2. The Data: Who Are Our Customers?
We analyzed a dataset of **7,043 customers**. We looked at three types of information:
1.  **Who they are**: Age, family status (Partner, Dependents).
2.  **What they buy**: Phone lines, Internet type (DSL, Fiber), Streaming TV.
3.  **How they pay**: Contract type (Month-to-month, 1-year), Payment method, Monthly bill.

**Data Cleaning**:
We found that some new customers had missing "Total Charges" because they hadn't been billed yet. Instead of deleting them, we smartly calculated what their bill should be, keeping 100% of the data.

## 3. Exploratory Analysis: What Drives Churn?
We played "detective" with the data and found several clear patterns.

### 🚩 The "Fiber Optic" Problem
*   **Finding**: Customers with **Fiber Optic** internet leave at a rate of **42%**.
*   **Comparison**: DSL customers only leave at 19%.
*   **Insight**: There is likely a problem with the Fiber service—it might be too expensive, unreliable, or competitors are offering better deals.

### 🚩 The "Electronic Check" Risk
*   **Finding**: Customers who pay by **Electronic Check** are huge churn risks (45% churn rate).
*   **Insight**: These customers might be less digitally integrated than those on automatic credit card payments.

### 🚩 The "Month-to-Month" Trap
*   **Finding**: Customers on **Month-to-Month** contracts are the most likely to leave (43%).
*   **Insight**: Long-term contracts (1 or 2 years) drastically reduce churn to under 3%.

### 🛡️ The "Sticky" Services
*   **Finding**: Customers who buy **Tech Support** or **Online Security** rarely leave.
*   **Insight**: These services make customers feel safe and supported. Bundling these for free might save at-risk customers.

## 4. The Solution: Predicting the Future
We built a "Crystal Ball" (a Machine Learning model) to predict churn.

### How We Built It
1.  **Teaching the Model**: We showed the model thousands of past customers.
2.  **Balancing the Class**: Since most people *don't* churn, the model might get lazy and guess "Stay" for everyone. We used a technique called **SMOTE** to show it an equal number of "Stayers" and "Leavers" so it learned both equally well.
3.  **The Algorithm**: We used **XGBoost**, a smart algorithm that builds hundreds of "decision trees" (flowcharts) to make a final prediction.

## 5. The Results
Our model achieved an **Accuracy of 86.08%**.

### What Does This Mean?
*   If we have 100 customers, the model correctly predicts the future for 86 of them.
*   We can now give the marketing team a list of "High Risk" customers (e.g., Fiber Optic users paying by Check).
*   **Action Plan**: Offer these specific customers a discount on a 1-year contract or free Tech Support. This could save the company millions in lost revenue.
