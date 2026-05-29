# Laptop Price Prediction using Multiple Preprocessing and ML Pipelines  
2026-1 Data Science (14455_005) Team 1 Open-source Contribution
## Overview

This project was created as a term assignment for a Data Science course. The objective of the project is to compare different preprocessing pipelines, encoding methods, scaling methods, classification models, and regression models in order to find the best-performing combination for laptop price prediction.

This GitHub repository was created to share knowledge and findings from the project.

The project uses a custom machine learning pipeline with:

1. Data preprocessing
2. Feature encoding
3. Feature scaling
4. Classification
5. Regression
6. Custom K-Fold cross validation
7. Comparison of all combinations


# Dataset

The dataset used in this project is:

- `laptops.csv` from kaggle by JUAN MERINO
https://www.kaggle.com/datasets/juanmerinobermejo/laptops-price-dataset

The dataset itself follows Apache 2.0 license.   
However, this project does not follow a specific license because I do not own the dataset.   
The code used in this project consists mostly of general machine learning techniques, so I may not be in a position to grant a license. I'm simply sharing what I have done and learned.   

Back to project,   
The dataset contains laptop specifications such as:

- CPU
- GPU
- RAM
- Storage
- Screen size
- Brand
- Storage type
- Touch support
- Laptop price

Target variable:
- Final Price



# Project Structure

Main functions:

| Function             | Description                          |
| -------------------- | ------------------------------------ |
| `fixed_preprocess()` | Data cleaning and preprocessing      |
| `encode()`           | Feature encoding experiments         |
| `scale()`            | Feature scaling experiments          |
| `regression()`       | Regression model testing             |
| `model()`            | Classification + regression pipeline |
| `test_combination()` | Tests all combinations               |
| `cross_val_custom()` | Custom K-Fold cross validation       |
| `show_top5()`        | Displays top 5 results               |

- fixed_preprocess() : Data cleaning and preprocessing 
  - data: input data
  - is_train: whether the input is train or not
  - train_stats: dictionary used to store preprocessing status values
  - return: preprocessed data
  
- encode(): Feature encoding experiments
  - df: input dataframe
  - is_train: whether the input is train or not
  - train_stats: dictionary used to store preprocessing status values
  - encoding_method: encoding method to use
  - return: encoded data
  
- scale(): Feature scaling experiments
  - train_pre: train set to scale
  - test_pre: test set to scale
  - scaling_method: scaling method to use
  - return: scaled data

- regression(): Regression model testing
  - X_train: training X
  - y_train_class: training class y
  - y_train_price: training price y
  - X_test: testing X
  - y_test_class: testing class y
  - y_test_price: testing price y
  - test_pred: prediction result of testset
  - regression_method: regression method to use
  - return: result dictionary includes various scores
 
- model(): Classification + regression pipeline
  - X_train: training X
  - y_train_class: training class y
  - Y_train_price: training price y
  - X_test: testing X
  - y_test_class: testing class y
  - y_test_price: testing price y
  - classification_method: classification method to use
  - regression_method: regression method to use
  - return: result dictionary includes various scores
 
- test_combination(): test all combinations
  - train_df: training dataset
  - test_df: testing dataset
  - return: result dataframe
 
- cross_val_custom(): Custom K-Fold cross validation
  - df: Original dataset
  - n_splits: number of folds
  - return: cv_result dataframe
 
- show_top5(): Displays top 5 results
  - cv_result: cross validation result dataframe
  - return: no return. use terminal

Most of the functions are self-explanatory. However, model() may require additional explanation.
The original name of the function was classification(). However, since our approach combined classification and regression, there was a need for a function that could call regression internally.

That's the reason why its name has been changed to model().


# Preprocessing

## 1. Missing Value Handling

### GPU Missing Values

Missing GPU values were replaced using a manually designed CPU-to-GPU mapping table.   
Based on domain knowledge, we thought that most of the missing values were actually integrated graphics, because without a graphics processor, the laptop would not be able to display a GUI.       
(and that generally does not happen in real life — nobody would buy a laptop that cannot display a GUI.)

Example:

```
'Intel Core i5' -> 'Intel Iris Xe Graphics'
'AMD Ryzen 5' -> 'AMD Radeon Graphics'
```

### Storage Type and Screen

- Storage type:
  - Filled using mode   
    mean or median values may become skewed or even unrealistic for storage capacity. (There are no storage like 12.34gb! Memory chips with unusual storage capacities are generally not manufactured.)
- Screen:
  - Filled using median   
    Same as above, mean or mode values are highly likely to be skewed or unrealistic.

The statistics were learned only from training data and reused for test data. This is important in most data science projects since statistics of test set can be a really good hint to model, but we can't get that when prediction on real life.


## 2. Data Cleaning
### Invalid Storage Removal
Rows with impossible storage values, such as values below zero, were removed.

### RAM Filtering

Only realistic RAM values were kept:
```python
[4, 8, 16, 24, 32, 64, 128]
```

As I mentioned above, because of memory chip limitations, values such as 9GB or 12GB RAM are unusual. (Technically possible, however highly likely just a wrong value.)

I think this is where intuition and domain knowledge become important. Nobody but you may need to decide what to keep and what to change. However, you must be careful because removing rows may introduce additional bias or skewness into the dataset.

### Brand Normalization
After that, we merged some brands that were actually the same brand but written differently.

Example:
```python
'Dynabook Toshiba' -> 'Toshiba'
```

### Feature Removal

Unused text-heavy features were removed:
- Laptop
- Model

These two columns were somewhat problematic. They have too many possible values, which is hard to encode. Also, at prediction phase, what would users actually enter as model names?      
"MACBOOK PRO"? "MBP"? "MB pro 14'"?

These features did not appear to be very useful for price prediction.

## 3. Log Transformation
The target variable was highly skewed. For example, some high-end laptops were significantly more expensive compared to low-end laptops. Therefore logarithmic transformation was applied.

Storage size was also log transformed.
Let's think about the following scenarios.
- 64GB? Unusable
- 128GB? Maybe
- 256GB? Bare minimum
- 512GB? Fine
- 1TB? Nice
- 2TB? More than enough for most people.
- 4TB? More than enough for most users

What I want to say is that when the capacity exceeds a certain value, like 512GB, then it may not significantly affect purchasing decisions or price.

# Encoding Methods
Three encoding methods were tested.

## 1. Default Encoding
A manually designed encoding method using domain knowledge. We put most of our effort into this part.

### CPU/GPU Score Mapping
CPU and GPU models were converted into approximate performance scores using PassMark scores as a guideline.

Example:
```
Intel Core i7 -> 25000
RTX 4060 -> 17000
```

### One-Hot Encoding
- Status
- Brand

### Storage Tier Mapping (Ordinal encoding)
```python
eMMC -> 0
HDD  -> 1
SSD  -> 2
```

Because SSDs are clearly better than HDDs and eMMC storage, at least in terms of general price and performance.

### Rare Brand Grouping
Low-frequency and low-importance brands were merged into 'Brand_others' using F-score analysis.

This was done to avoid feature explosion (the curse of dimensionality).


## 2. Full One-Hot Encoding
One-hot encoding was applied to all categorical features using
Scikit-learn's OneHotEncoder.

## 3. Ordinal Encoding
Ordinal encoding was applied to all categorical features using Scikit-learn's OrdinalEncoder.


# Scaling Methods
Three scaling methods were tested.
- 1. Standard Scaling
- 2. Min-Max Scaling
- 3. Robust Scaling

# Classification Models
The dataset was first divided into:

- Low-price laptops
- High-price laptops

using the median price. The reason we used the median was to preserve as much data as possible as the number of rows was only around 2000. When divided into two groups, it drops to around 1000. Fewer than 1000 samples may not be sufficient for regression.

The classification target, Price Class was generated.

## 1. Decision Tree Classifier

Tested parameters:
- `max_depth` : Maximum depth of tree
- `min_samples_leaf` : minimum number of samples required in a leaf node.

## 2. Logistic Regression
Tested parameter:
- `C`: Regularization strength.

Generally,   
Small C -> underfitting  
Big C -> overfitting

## 3. K-Nearest Neighbors
Tested parameter:
- `n_neighbors`: number of neighbors - K.


# Regression Models

After classification, separate regression models were trained for each group.

## 1. Multiple Linear Regression
Standard linear regression model.

## 2. Polynomial Regression
Polynomial features were generated and then used with linear regression.

Degrees 2 and 3 were used. Higher degrees were impractical since it takes too much time to calculate.

Also, high-complexity combinations such as  
OneHot Encoding + Polynomial Regression were also skipped.

These combinations were computationally too expensive and were highly likely to perform worse because of overfitting.


# Evaluation Metrics
The following metrics were used.

- RMSE(Root Mean Squared Error): Lower is better.
- MAE(Mean Absolute Error): Lower is better.
- R2 Score(): Higher is better.

# Cross Validation

A custom 10-Fold Stratified Cross Validation pipeline was implemented.
The folds were stratified using the price class because this was the target of the classifier.

For each fold:

1. Preprocessing (fixed)
2. Encoding
3. Scaling
4. Classification
5. Regression
6. Evaluation

were repeated (all processes explained above).

Final scores were calculated using
the mean and standard deviation of each fold's results.


# Experimental Pipeline

The project tested combinations of:

- 3 encoding methods
- 3 scaling methods
- 3 classification models with multiple parameters
- 2 regression methods with multiple parameters

This resulted in a large number of combinations being evaluated automatically.

# Results
## Best Performing Combinations
The program automatically displays:

- Top 5 RMSE results
- Top 5 MAE results
- Top 5 R2 results

# Observations / Result
<img width="1400" height="410" alt="Screenshot 2026-05-16 at 13 39 52" src="https://github.com/user-attachments/assets/271a4d34-0dfb-4816-885d-bd1ef812b1e6" />
You can also find the output in output.txt in this repository.

The best combination varied depending on the evaluation metric. However, the top-performing models achieved very similar results overall.

If I had to select a single overall best combination, I would choose:
- `onehot + robust + Logistic + multiple_linear`

We expected that our manually created pipeline would outperform the others. However, this was not true.

Full One-Hot Encoding frequently achieved better performance than the manually designed encoding method. This suggests that preserving detailed categorical information can sometimes outperform manually compressed domain-specific representations.

I highly recommend trying basic encoding methods before attempting more advanced domain-specific approaches.

# How to Run

Please install all required packages.   
After that, you can run the code with:
```bash
python Laptop_find_best.py
```
However, I do not recommend running this code on a battery-powered, low performance computer as the code is computationally heavy.

# Thank you.
