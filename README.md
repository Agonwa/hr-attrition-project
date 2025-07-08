### 🧠 Employee Attrition Analysis and Prediction

#### 📌 Project Overview

This project explores employee attrition within an organization using data-driven insights and predictive modeling. It leverages both **Power BI visualizations** for storytelling and **machine learning models** to predict which employees are at risk of leaving.

---

#### 📈 Dataset Description
The dataset consists of 1,470 employee records with 35 features, such as  `Age`, `Department`, `JobRole`, `MonthlyIncome`, `WorkLifeBalance`, and `Attrition` (categorical target with values `Yes` or `No`). Sourced from Kaggle's IBM HR Analytics dataset, it provides insights into factors driving employee turnover.

---

#### 🔍 Use Cases Addressed

1. **Which department has the highest attrition rate?**
2. **Is there a strong relationship between job satisfaction and attrition?**
3. **Does overtime or work-life balance correlate with attrition?**
4. **Are certain roles, age groups, or salary bands more at risk?**

---

#### 🛠️ Tools & Technologies

- Python: Pandas, Scikit-learn, XGBoost, Matplotlib, Numpy, Seaborn for data analysis and modeling. 
- Jupyter Notebooks: for exploratory data analysis and modeling.
- Power BI: for interactive dashboards visualizing attrition insights. 
- Joblib: for model serialization.

---

#### 📊 Exploratory Data Analysis (EDA)

EDA was performed to understand the key drivers of attrition. Insights such as:

* The **Sales** department had the highest attrition rate.
* **Overtime** is strongly correlated with higher attrition (30.5% vs 10.4%).
* Employees with **poor work-life balance** or in **lower salary bands** are more likely to leave.

EDA was documented in the [`01_exploratory_data_analysis.ipynb`](./notebooks/01_exploratory_data_analysis.ipynb) notebook and visualized in **Power BI**.

---

#### 🧠 Machine Learning Model

The categorical `Attrition` target (`Yes`/`No`) was encoded as binary (`1`/`0`) for model training. I trained multiple classification models, including:

* **Logistic Regression**
* **Random Forest**
* ✅ **XGBoost** (Best performer)

📈 **Best Model Performance (XGBoost)**

* Accuracy: `86.7%`
* Recall (Attrition class): `40%`
* Precision (Attrition class): `63%`

The final model was saved in the [`modeling/xgboost_model.pkl`](./modeling/xgboost_model.pkl).

---

#### 📁 Project Structure

```
├── data/
│   ├── cleaned_data/
│   │   ├── cleaned_hr_data.csv                # Cleaned dataset for machine learning model building
│   │   └── hr_data_for_powerbi.csv            # Cleaned and aggregated dataset for Power BI visualizations
│   ├── raw_data/
│   │   └── hr_data_raw.csv                    # Original raw dataset
├── modeling/
│   ├── scaler.pkl                             # Standard scaler for feature preprocessing
│   └── xgboost_model.pkl                      # Trained XGBoost model for attrition prediction      
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb     # EDA and initial insights
│   ├── 02_data_cleaning_for_modeling.ipynb    # Data cleaning for ML modeling
│   ├── 03_model_building_and_evaluation.ipynb # Model training and performance evaluation
│   └── 04_powerbi_preparation.ipynb           # Data preparation for Power BI dashboard
├── screenshots/                               # PNGs of key dashboard visuals
├── visuals/
│   └── hr_attrition_dashboard.pbix            # Power BI dashboard with attrition visualizations
└── README.md                                  # Project overview and documentation

```

---

#### 💡 Key Insights

* **Work-life balance and overtime** are critical predictors of attrition.
* **Age group 18–25** shows the highest risk.
* **Job role and salary band** significantly affect turnover probability.

---

#### 🚀 Future Improvements

* Improve model recall for attrition class.
* Deploy a **Streamlit app** for HR teams to upload employee data and get attrition predictions.
* Automate report generation with scheduled Power BI refresh.

---

#### 🚀 Getting Started

1. **Clone the Repository**:
   - Clone the project and navigate to the folder: 
   ```bash
    git clone https://github.com/agonwa/hr-attrition-project.git
    cd "HR ATTRITION PROJECT"
    ```

2. **Set Up Your Environment**:
   - Install **Python 3.8+** from [python.org](https://www.python.org/downloads/).
   - Install required Python libraries using pip:
     ```bash
     pip install pandas scikit-learn xgboost matplotlib jupyter joblib numpy seaborn
     ```
   - Install **Power BI Desktop** (free) from [Microsoft’s website](https://powerbi.microsoft.com/desktop/) to view the dashboard.

3. **Run the Notebooks**:
   - Start Jupyter Notebook: 
   ```bash
   jupyter notebook
   ```
   - Open the notebooks in `notebooks/` to explore:
     - `01_exploratory_data_analysis.ipynb`: Perform exploratory data analysis.
     - `02_data_cleaning_for_modeling.ipynb`: Clean data for machine learning modeling.
     - `03_model_building_&_evaluation.ipynb`: Train and evaluate the XGBoost model.
     - `04_powerbi_preparation.ipynb`: Prepare data for Power BI visualizations.

4. **View the Power BI Dashboard**:
   - Open `visuals/hr_attrition_dashboard.pbix` in Power BI Desktop to explore attrition visualizations.

5. **Use the Model**:
   - Load `modeling/xgboost_model.pkl` and `modeling/scaler.pkl` in Python using `joblib`. Expected input features match columns in `cleaned_hr_data.csv` (e.g., `Age`, `MonthlyIncome`, `Overtime`):
```python
import joblib
model = joblib.load('modeling/xgboost_model.pkl')
scaler = joblib.load('modeling/scaler.pkl')
```

---

#### 🙌 Author

**NWOKIKE CHIAGOZIE**  
*Data Scientist | Analyst | ML Enthusiast | AI Explorer*

📧 [chiagozienwokike@gmail.com](mailto:chiagozienwokike@gmail.com)
