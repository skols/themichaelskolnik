---
{
  "title": "My code from the exercises of Level 1 of Kaggle's Learn Machine Learning series",
  "subtitle": "",
  "date": "2018-03-25",
  "slug": "home-prices-python",
  "tags": ["Python", "machine learning"]
}
---
<!--more-->

## Level 1 *Learn Maching Learning* series on Kaggle
I went through the level 1 *Learn Machine Learning* series on Kaggle using Python (https://www.kaggle.com/learn/machine-learning). The data used is from the [*Home Prices: Advanced Regression Techniques*](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) competition.

This post will show the section name, my code from the corresponding section for the instructions under **Your Turn**, and some brief notes on what is taught in each section. You should go to the links to learn and also do yourself as I found this very helpful. Even if you've taken other machine learning courses as I have, this is a good refresher.

### Section 2
[Starting Your ML Project](https://www.kaggle.com/dansbecker/starting-your-ml-project)

This section has you load the data and set up the computing environment for the project. You also view summary statistics and columns.


```python
# Import the pandas library
import pandas as pd
```

```python
# Save filepath to variable
training_data_filepath = "C:/Development/Kaggle/House Prices - Advanced \
Regression Techniques/train.csv"

# Read the data and store in a dataframe called training_set
training_set = pd.read_csv(training_data_filepath)

# Print a summary of the data in training_set
print(training_set.describe())
```

                    Id   MSSubClass  LotFrontage        LotArea  OverallQual  \
    count  1460.000000  1460.000000  1201.000000    1460.000000  1460.000000   
    mean    730.500000    56.897260    70.049958   10516.828082     6.099315   
    std     421.610009    42.300571    24.284752    9981.264932     1.382997   
    min       1.000000    20.000000    21.000000    1300.000000     1.000000   
    25%     365.750000    20.000000    59.000000    7553.500000     5.000000   
    50%     730.500000    50.000000    69.000000    9478.500000     6.000000   
    75%    1095.250000    70.000000    80.000000   11601.500000     7.000000   
    max    1460.000000   190.000000   313.000000  215245.000000    10.000000   
    
           OverallCond    YearBuilt  YearRemodAdd   MasVnrArea   BsmtFinSF1  \
    count  1460.000000  1460.000000   1460.000000  1452.000000  1460.000000   
    mean      5.575342  1971.267808   1984.865753   103.685262   443.639726   
    std       1.112799    30.202904     20.645407   181.066207   456.098091   
    min       1.000000  1872.000000   1950.000000     0.000000     0.000000   
    25%       5.000000  1954.000000   1967.000000     0.000000     0.000000   
    50%       5.000000  1973.000000   1994.000000     0.000000   383.500000   
    75%       6.000000  2000.000000   2004.000000   166.000000   712.250000   
    max       9.000000  2010.000000   2010.000000  1600.000000  5644.000000   
    
               ...         WoodDeckSF  OpenPorchSF  EnclosedPorch    3SsnPorch  \
    count      ...        1460.000000  1460.000000    1460.000000  1460.000000   
    mean       ...          94.244521    46.660274      21.954110     3.409589   
    std        ...         125.338794    66.256028      61.119149    29.317331   
    min        ...           0.000000     0.000000       0.000000     0.000000   
    25%        ...           0.000000     0.000000       0.000000     0.000000   
    50%        ...           0.000000    25.000000       0.000000     0.000000   
    75%        ...         168.000000    68.000000       0.000000     0.000000   
    max        ...         857.000000   547.000000     552.000000   508.000000   
    
           ScreenPorch     PoolArea       MiscVal       MoSold       YrSold  \
    count  1460.000000  1460.000000   1460.000000  1460.000000  1460.000000   
    mean     15.060959     2.758904     43.489041     6.321918  2007.815753   
    std      55.757415    40.177307    496.123024     2.703626     1.328095   
    min       0.000000     0.000000      0.000000     1.000000  2006.000000   
    25%       0.000000     0.000000      0.000000     5.000000  2007.000000   
    50%       0.000000     0.000000      0.000000     6.000000  2008.000000   
    75%       0.000000     0.000000      0.000000     8.000000  2009.000000   
    max     480.000000   738.000000  15500.000000    12.000000  2010.000000   
    
               SalePrice  
    count    1460.000000  
    mean   180921.195890  
    std     79442.502883  
    min     34900.000000  
    25%    129975.000000  
    50%    163000.000000  
    75%    214000.000000  
    max    755000.000000  
    
    [8 rows x 38 columns]
    


```python
# Print the columns in training_set
print(training_set.columns)
```

    Index(['Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
           'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
           'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
           'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
           'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
           'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
           'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
           'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
           'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
           'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
           'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
           'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
           'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
           'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
           'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',
           'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',
           'SaleCondition', 'SalePrice'],
          dtype='object')
    

### Section 3
[Selecting and Filtering in Pandas](https://www.kaggle.com/dansbecker/selecting-and-filtering-in-pandas)

This section has you use pandas to select the data you want to use, which allows you to get the data ready for modeling.


```python
# Store the series of prices separately as training_price_data
training_price_data = training_set.SalePrice

# Print the first 5 records
print(training_price_data.head())
```

    0    208500
    1    181500
    2    223500
    3    140000
    4    250000
    Name: SalePrice, dtype: int64
    


```python
# Create a list with the columns I am interested in
columns_of_interest = ["LotArea", "YearBuilt"]

# Create a dataframe with just those columns
training_two_columns = training_set[columns_of_interest]

# Print a summary of the training_two_columns dataframe
print(training_two_columns.describe())
```

                 LotArea    YearBuilt
    count    1460.000000  1460.000000
    mean    10516.828082  1971.267808
    std      9981.264932    30.202904
    min      1300.000000  1872.000000
    25%      7553.500000  1954.000000
    50%      9478.500000  1973.000000
    75%     11601.500000  2000.000000
    max    215245.000000  2010.000000
    

### Section 4
[Your First Scikit-Learn Model](https://www.kaggle.com/dansbecker/your-first-scikit-learn-model)

Building your first model! Spoiler: it's a Decision Tree model. :-)


```python
# Select the target variable and call it y
y = training_set.SalePrice
```

```python
# Create a list of the predictor variables
predictors = ["LotArea", "YearBuilt", "1stFlrSF", "2ndFlrSF", "FullBath",
              "BedroomAbvGr", "TotRmsAbvGrd"]

# Create a new dataframe with the predictors list
X = training_set[predictors]
```

```python
# Import DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor

# Define the first model
tree_model = DecisionTreeRegressor()

# Fit model
tree_model.fit(X, y)
```

    DecisionTreeRegressor(criterion='mse', max_depth=None, max_features=None,
               max_leaf_nodes=None, min_impurity_decrease=0.0,
               min_impurity_split=None, min_samples_leaf=1,
               min_samples_split=2, min_weight_fraction_leaf=0.0,
               presort=False, random_state=None, splitter='best')




```python
# Make some predictions
print("Making predictions for the first 10 houses")
print(X.head(n=10))
print("The predictions are:")
print(tree_model.predict(X.head(n=10)))
```

    Making predictions for the first 10 houses
       LotArea  YearBuilt  1stFlrSF  2ndFlrSF  FullBath  BedroomAbvGr  \
    0     8450       2003       856       854         2             3   
    1     9600       1976      1262         0         2             3   
    2    11250       2001       920       866         2             3   
    3     9550       1915       961       756         1             3   
    4    14260       2000      1145      1053         2             4   
    5    14115       1993       796       566         1             1   
    6    10084       2004      1694         0         2             3   
    7    10382       1973      1107       983         2             3   
    8     6120       1931      1022       752         2             2   
    9     7420       1939      1077         0         1             2   
    
       TotRmsAbvGrd  
    0             8  
    1             6  
    2             6  
    3             7  
    4             9  
    5             5  
    6             7  
    7             7  
    8             8  
    9             5  
    The predictions are:
    [208500. 181500. 223500. 140000. 250000. 143000. 307000. 200000. 129900.
     118000.]
    

### Section 5
[Model Validation](https://www.kaggle.com/dansbecker/model-validation)

This section introduces model validation to measure the performance of the model. You also learn about "in-sample" scores and why you should split your data into training and test sets.


```python
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Split data into training and validation data, for both predictors and
# target.
# The split is based on a random number generator. Supplying a numeric value
# to the random_state argument guarantees we get the same split every time we
# run this script. It can be any number; I'm choosing 42.
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=42)

# Define the model
tree_model = DecisionTreeRegressor()

# Fit model
tree_model.fit(X_train, y_train)

# Get predicted prices on validation data
predictions_val = tree_model.predict(X_val)
print(mean_absolute_error(y_val, predictions_val))
```

    30855.94794520548
    

### Section 6
[Underfitting, Overfitting, and Model Optimization](https://www.kaggle.com/dansbecker/underfitting-overfitting-and-model-optimization)

In this section you learn about underfitting, overfitting, and optimizing your model. The *max_leaf_nodes* argument is used to provide a very sensible way to control overfitting vs underfitting in Decision Tree models.


```python
# Create a utility function to help compare MAE scores from different values
# for *max_leaf_nodes*.
def get_mae(max_leaf_nodes, predictors_train, val_predictors, targ_train,
            targ_val):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes,
                                  random_state=42)
    model.fit(predictors_train, targ_train)
    preds_val = model.predict(val_predictors)
    mae = mean_absolute_error(targ_val, preds_val)
    return(mae)
```

```python
# Loop through a list of max leaf nodes and print the MAE of each
for max_leaf_nodes in [5, 10, 25, 50, 100, 200, 500, 1000, 5000]:
    my_mae = get_mae(max_leaf_nodes, X_train, X_val, y_train, y_val)
    print("Max leaf nodes: {0} \t\t Mean Absolute Error: {1}".
          format(max_leaf_nodes, my_mae))
```

    Max leaf nodes: 5 		 Mean Absolute Error: 35244.94032482636
    Max leaf nodes: 10 		 Mean Absolute Error: 31256.15612179911
    Max leaf nodes: 25 		 Mean Absolute Error: 29611.012298361497
    Max leaf nodes: 50 		 Mean Absolute Error: 27232.09960472095
    Max leaf nodes: 100 		 Mean Absolute Error: 27021.244092878136
    Max leaf nodes: 200 		 Mean Absolute Error: 29015.822642629737
    Max leaf nodes: 500 		 Mean Absolute Error: 31450.856430996708
    Max leaf nodes: 1000 		 Mean Absolute Error: 31717.233789954334
    Max leaf nodes: 5000 		 Mean Absolute Error: 31724.594520547944
    

### Section 7
[Random Forests](https://www.kaggle.com/dansbecker/random-forests)

This section has use a Random Forest model and you can compare the results to the Decision Tree one.


```python
from sklearn.ensemble import RandomForestRegressor

# Create the second model, a Random Forest
forest_model = RandomForestRegressor()
forest_model.fit(X_train, y_train)
forest_preds = forest_model.predict(X_val)
print(mean_absolute_error(y_val, forest_preds))
```

    22458.350528375733
    


### Putting all the code in one place


```python
# Import the necessary libraries
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


# Save filepath to variable
training_data_filepath = "C:/Development/Kaggle/House Prices - Advanced \
Regression Techniques/train.csv"

# Read the data and store in a dataframe called training_set
training_set = pd.read_csv(training_data_filepath)

# Print a summary of the data in training_set
print(training_set.describe())

# Print the columns in training_set
print(training_set.columns)

# Store the series of prices separately as training_price_data
training_price_data = training_set.SalePrice

# Print the first 5 records
print(training_price_data.head())

# Create a list with the columns I am interested in
columns_of_interest = ["LotArea", "YearBuilt"]

# Create a dataframe with just those columns
training_two_columns = training_set[columns_of_interest]

# Print a summary of the training_two_columns dataframe
print(training_two_columns.describe())

# Select the target variable and call it y
y = training_set.SalePrice

# Create a list of the predictor variables
predictors = ["LotArea", "YearBuilt", "1stFlrSF", "2ndFlrSF", "FullBath",
              "BedroomAbvGr", "TotRmsAbvGrd"]

# Create a new dataframe with the predictors list
X = training_set[predictors]

# Define the first model, a Decision Tree
tree_model = DecisionTreeRegressor()

# Fit model
tree_model.fit(X, y)

# Make some predictions
print("Making predictions for the first 10 houses")
print(X.head(n=10))
print("The predictions are:")
print(tree_model.predict(X.head(n=10)))

# Split data into training and validation data, for both predictors and
# target.
# The split is based on a random number generator. Supplying a numeric value
# to the random_state argument guarantees we get the same split every time we
# run this script. It can be any number; I'm choosing 42.
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=42)

# Define the model
tree_model = DecisionTreeRegressor()

# Fit model
tree_model.fit(X_train, y_train)

# Get predicted prices on validation data
predictions_val = tree_model.predict(X_val)
print(mean_absolute_error(y_val, predictions_val))

# Create a utility function to help compare MAE scores from differevalues for
# *max_leaf_nodes*.
def get_mae(max_leaf_nodes, predictors_train, val_predictors, targ_train,
            targ_val):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes,
                                  random_state=42)
    model.fit(predictors_train, targ_train)
    preds_val = model.predict(val_predictors)
    mae = mean_absolute_error(targ_val, preds_val)
    return(mae)


# Loop through a list of max leaf nodes and print the MAE of each
for max_leaf_nodes in [5, 10, 25, 50, 100, 200, 500, 1000, 5000]:
    my_mae = get_mae(max_leaf_nodes, X_train, X_val, y_train, y_val)
    print("Max leaf nodes: {0} \t\t Mean Absolute Error: {1}".
          format(max_leaf_nodes, my_mae))

# Create the second model, a Random Forest
forest_model = RandomForestRegressor()
forest_model.fit(X_train, y_train)
forest_preds = forest_model.predict(X_val)
print(mean_absolute_error(y_val, forest_preds))
```

My next post will be the level 2 part of the series and after that I'm going to do it in R. I'm hoping to have level 2 up in a few days, a week at most.
