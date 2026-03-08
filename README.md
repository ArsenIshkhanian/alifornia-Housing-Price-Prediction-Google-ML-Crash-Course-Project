# California Housing Price Prediction

This project is based on exercises from **Google's Machine Learning Crash Course (MLCC)** and focuses on building a machine learning pipeline to predict housing prices using the **California Housing dataset**.

The project was implemented using **Google Colaboratory (Colab)** and demonstrates key machine learning concepts such as exploratory data analysis, feature engineering, model training, and hyperparameter tuning.

---

## Project Workflow

The project follows a typical machine learning pipeline:

### 1. Exploratory Data Analysis (EDA)
- Visualized feature distributions using histograms
- Detected outliers using boxplots
- Explored feature relationships using scatter plots
- Analyzed correlations using a heatmap

### 2. Feature Engineering
Created new features to better represent housing characteristics:

- `rooms_per_household`
- `bedrooms_per_room`
- `people_per_house`

Added geographic features:

- `distance_to_coast`
- `dist_to_sf` - Distance from each location to **San Francisco (SF)**.
- `dist_to_la` – Distance from each location to **Los Angeles (LA)**.
- `north_california`

### 3. Feature Selection
Removed redundant raw features after creating engineered features.

### 4. Data Splitting
The dataset was split into training and test sets for model evaluation.

### 5. Model Training
Three models were trained and compared:

- **Linear Regression** (baseline model)
- **Random Forest Regressor**
- **XGBoost Regressor**

### 6. Hyperparameter Tuning
Used **RandomizedSearchCV** to optimize XGBoost parameters.

### 7. Model Evaluation
Models were evaluated using **Root Mean Squared Error (RMSE)**.

---

## Model Performance

| Model | RMSE |
|------|------|
| Linear Regression | ~73,439 |
| Random Forest | ~51,189 |
| XGBoost | ~48,197 |
| XGBoost (Tuned) | **~45,605** |

The tuned **XGBoost model** achieved the best performance.

---

## Technologies Used

- Python
- Google Colab
- Pandas
- NumPy
- Matplotlib
- plotly
- Seaborn
- Scikit-learn
- XGBoost

---

## Dataset

The project uses the **California Housing dataset**, commonly used for regression tasks in machine learning.

The dataset includes features such as:

- population statistics
- housing characteristics
- geographic location
- median income levels

The target variable is:

- median_house_value
