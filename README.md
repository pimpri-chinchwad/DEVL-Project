# DEVL-E-Commerce-Task

https://www.kaggle.com/datasets/prachi13/customer-analytics?resource=download
Here are two practical, real-world problem statements tailored for a Second Year Computer Engineering student taking **Data Exploration and Visualization Laboratory (DEnVL)**.

While Pandas is typically used for loading tabular CSV files, NumPy is the underlying engine used for numerical heavy lifting, mathematical transformations, filtering, and statistical analysis (which aligns perfectly with your syllabus objectives of *data cleaning, preprocessing, and building a statistical foundation*).

---

## Problem Statement 1: E-Commerce Customer Analytics

**Core NumPy Concepts Tested:** Vectorized boolean indexing, statistical aggregations (`mean`, `std`), data cleaning, and masking.

### 1. Dataset Context (Kaggle Reference)

* **Dataset Name:** *E-Commerce Shipping Data* (or any standard retail transactional dataset).
* **What it represents:** A structured matrix where rows represent individual customer orders, and columns represent numeric features like **Order Cost (USD)**, **Discount Offered ($)**, and **Customer Rating (1 to 5)**.

### 2. The Problem Statement

You are given a raw numerical dataset representing 1,000 transactions. The matrix has 3 columns: `[Order_Cost, Discount_Offered, Customer_Rating]`.
Your tasks are:

1. **Data Cleaning:** Identify and replace any invalid customer ratings (ratings outside the 1–5 range, e.g., `0` or missing values represented as `-1`) with the average rating of the valid transactions.
2. **Feature Engineering:** Calculate a new array representing the net amount paid by each customer (`Order_Cost - Discount_Offered`).
3. **Statistical Filtering:** Extract and display the net amount paid only for the "High-Value Disappointed Customers"—defined as customers who paid *above the average net cost* but rated the experience *below 3*.

### 3. Solution Implementation

```python
import numpy as np

# Simulating the Kaggle dataset structure (1000 rows, 3 columns)
# Columns: [Order_Cost, Discount_Offered, Customer_Rating]
np.random.seed(42)
raw_data = np.hstack([
    np.random.uniform(20, 300, (1000, 1)),  # Order Cost
    np.random.uniform(0, 50, (1000, 1)),    # Discount
    np.random.choice([1, 2, 3, 4, 5, -1], (1000, 1)) # Rating (-1 simulates bad/missing data)
])

# --- Step 1: Data Cleaning ---
ratings = raw_data[:, 2]
valid_ratings_mask = (ratings >= 1) & (ratings <= 5)
mean_valid_rating = np.mean(ratings[valid_ratings_mask])

# Replace -1 (invalid) with the calculated mean
ratings[~valid_ratings_mask] = mean_valid_rating
raw_data[:, 2] = ratings

# --- Step 2: Feature Engineering (Vectorized Operations) ---
order_cost = raw_data[:, 0]
discount = raw_data[:, 1]
net_paid = order_cost - discount

# --- Step 3: Statistical Filtering using Boolean Masks ---
mean_net_paid = np.mean(net_paid)

# Condition: Paid > mean AND rating < 3
target_mask = (net_paid > mean_net_paid) & (raw_data[:, 2] < 3)
high_value_disappointed = net_paid[target_mask]

print(f"Average Net Paid: ${mean_net_paid:.2f}")
print(f"Number of high-value disappointed customers found: {len(high_value_disappointed)}")
print(f"First 5 segments net amounts: {high_value_disappointed[:5]}")

```

---

## Problem Statement 2: Weather Sensor Data Standardization & Outliers

**Core NumPy Concepts Tested:** Matrix reshaping, axis-based operations (`axis=0`, `axis=1`), Broadcasting, and Z-score outlier detection.

### 1. Dataset Context (Kaggle Reference)

* **Dataset Name:** *Daily Climate time-series Data*.
* **What it represents:** A time-series matrix of weather readings where columns represent daily metrics like **Mean Temperature**, **Humidity**, and **Wind Speed**.

### 2. The Problem Statement

You receive a 2D NumPy array containing climate metrics for 365 days across 3 features: `[Temperature, Humidity, Wind_Speed]`. Because these features are on completely different scales, they cannot be used directly in machine learning algorithms.
Your tasks are:

1. **Feature Scaling (Standardization):** Center and scale the dataset so that each feature has a mean ($\mu$) of `0` and a standard deviation ($\sigma$) of `1`. Formula: $Z = \frac{X - \mu}{\sigma}$
2. **Outlier Detection:** Find the indices of days where *any* of the three weather metrics deviated by more than 2.5 standard deviations from the seasonal mean (anomalous weather days).

### 3. Solution Implementation

```python
import numpy as np

# Simulating 365 days of weather data for 3 features
# Columns: [Temperature (°C), Humidity (%), Wind_Speed (km/h)]
np.random.seed(7)
weather_data = np.hstack([
    np.random.normal(25, 5, (365, 1)),   # Temp
    np.random.normal(60, 15, (365, 1)),  # Humidity
    np.random.normal(15, 8, (365, 1))    # Wind Speed
])

# Injecting a few extreme weather anomaly days manually
weather_data[50] = [45.0, 98.0, 55.0]  # Severe storm day

# --- Step 1: Standardization using Broadcasting ---
# Compute mean and std dev across rows (axis=0) for each column
means = np.mean(weather_data, axis=0)
stds = np.std(weather_data, axis=0)

# Broadcast subtraction and division across the entire matrix
standardized_weather = (weather_data - means) / stds

# --- Step 2: Outlier Detection ---
# Identify absolute values greater than 2.5
outlier_mask = np.abs(standardized_weather) > 2.5

# Find row indices where ANY column is true
anomalous_days = np.any(outlier_mask, axis=1)
anomalous_indices = np.where(anomalous_days)[0]

print(f"Feature Means (Before): {means}")
print(f"Feature Std Devs (Before): {stds}")
print(f"\nAnomalous weather day indices found: {anomalous_indices}")
print(f"Raw data on day 50 anomaly: {weather_data[50]}")

```

### Why these match your SY BTech Syllabus:

* **Assignment 1 Core:** Demonstrates data manipulation without relying on slow Python loops (`for` loops), showing why NumPy's C-backend vectorization is efficient.
* **Assignment 2 & 3 Prep:** Directly handles data cleaning (outliers/null values) and establishes the statistical framework (means, distributions) required before you start plotting with Matplotlib/Seaborn.
