# Steel Industry Energy Consumption Prediction

## Project Overview
This project involves a deep exploratory data analysis (EDA), feature engineering, and baseline regression modeling for a steel manufacturing plant's energy consumption. The goal is to identify patterns in energy usage and build a predictive model to forecast consumption (kWh) based on power factors and operational variables.

## Dataset Information
- **Source:** Steel Industry Energy Consumption Dataset.
- **Size:** 35,040 entries.
- **Features:**
  - `date`: Timestamp of the reading.
  - `Usage_kWh`: Target variable (Energy Consumption).
  - `Lagging_Current_Reactive.Power_kVarh` & `Leading_Current_Reactive_Power_kVarh`: Reactive power metrics.
  - `CO2(tCO2)`: CO2 emissions.
  - `Power Factors`: Lagging and leading current power factors.
  - `Load_Type`: Categorical status (Light, Medium, Maximum Load).

## Environment Setup
To run this project, you need Python and the following libraries:
- `pandas`, `numpy`: Data manipulation.
- `matplotlib`, `seaborn`: Data visualization.
- `scikit-learn`: Machine learning modeling.
- `pymupdf`: PDF processing (for instruction extraction).
- `openpyxl`: Excel file support.

## Feature Engineering Steps
1. **Temporal Extraction:** Extracted `Hour`, `Day`, `Month`, and `Is_Weekend` from the timestamp.
2. **Calculated Ratios:** Created `Power_Factor_Ratio` (Leading PF / Lagging PF).
3. **Categorical Flagging:** Created `High_Load` binary feature based on the 75th percentile of `Usage_kWh` (used for EDA but dropped for modeling to prevent leakage).
4. **Encoding:** Applied One-Hot Encoding to categorical variables like `Load_Type` and `Day_of_week`.

## EDA Findings
- **Correlations:** Energy usage is almost perfectly correlated with `CO2(tCO2)` (0.99) and strongly correlated with `Lagging_Current_Reactive.Power_kVarh` (0.90).
- **Outliers:** 328 outliers were detected using the IQR method, typically representing massive spikes during peak production.
- **Patterns:** Usage peaks significantly between 9 AM and 6 PM during "Maximum Load" status. Weekends show slightly lower average usage compared to weekdays.

## Model Training Process
Four baseline regression models were trained using an 80/20 train-test split:
1. Linear Regression
2. Ridge Regression
3. Decision Tree Regressor
4. Random Forest Regressor

Performance was validated using **5-fold Cross-Validation** (Mean RMSE).

## Results and Conclusions
| Model | MAE | RMSE | R-squared | Mean CV RMSE |
| :--- | :--- | :--- | :--- | :--- |
| Linear Regression | 2.6284 | 4.1644 | 0.9847 | 4.4938 |
| Ridge Regression | 4.3670 | 6.3373 | 0.9647 | 6.6337 |
| Decision Tree | 0.5678 | 1.6249 | 0.9977 | 2.7840 |
| **Random Forest** | **0.3590** | **1.1195** | **0.9989** | **2.2540** |

**Conclusion:** The **Random Forest Regressor** is the best model for this task, successfully capturing the complex, non-linear relationships in industrial energy data with minimal error.
