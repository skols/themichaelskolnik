---
{
  "title": "My R code from two sections of the Level 2 exercises of Kaggle's Learn Machine Learning series",
  "subtitle": "",
  "date": "2018-05-08",
  "slug": "home-prices-r2",
  "tags": ["R", "machine learning"]
}
---
<!--more-->

## *Learn Maching Learning* series on Kaggle in R

This is my R code for the first two sections of the level 2 part of the *Learn Machine Learning* series on Kaggle. I've already done the Python one, which is on Kaggle located [here](https://www.kaggle.com/learn/machine-learning). The data used is from the [*Home Prices: Advanced Regression Techniques*](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) competition.

I had planned on doing all of level 2 but that was more difficult than I expected. One reason for that is the Python series told me what packages to use and gave me an outline of steps to follow. Since I'm doing this in R with no tutorial to follow, I had to research and decide which packages to use and steps to follow. That slowed me down but I definitely learned a lot. I also had problems with Jupyter and the R kernel, so I worked on this in RStudio while fixing those issues.

### Load and install packages and load the data
I learned since the last time that I need to do an if-else statement, not just if, when checking for packages before loading them.


```R
# Install and load packages
if (!require("randomForest")) {
  install.packages("randomForest", repos="http://cran.rstudio.com/")
  library(randomForest)
} else {
  library(randomForest)
}

if (!require("dplyr")) {
  install.packages("dplyr", repos="http://cran.rstudio.com/")
  library(dplyr)
} else {
  library(dplyr)
}

if (!require("caTools")) {
  install.packages("caTools", repos="http://cran.rstudio.com/")
  library(caTools)
} else {
  library(caTools)
}

if (!require("rpart")) {
  install.packages("rpart", repos="http://cran.rstudio.com/")
  library(rpart)
} else {
  library(rpart)
}

# Save filepath to variable
training_data_filepath <- "C:/Development/Kaggle/House Prices - Advanced Regression Techniques/train.csv"

# Import data
dataset <- read.csv(training_data_filepath)
```

    Loading required package: randomForest
    randomForest 4.6-12
    Type rfNews() to see new features/changes/bug fixes.
    Loading required package: dplyr
    
    Attaching package: 'dplyr'
    
    The following object is masked from 'package:randomForest':
    
        combine
    
    The following objects are masked from 'package:stats':
    
        filter, lag
    
    The following objects are masked from 'package:base':
    
        intersect, setdiff, setequal, union
    
    Loading required package: caTools
    Warning message:
    "package 'caTools' was built under R version 3.4.4"Loading required package: rpart
    

### Split the data set into training and test

This is the same as before.


```R
# Split data into training and validation data, for both predictors and target.
set.seed(42)
split <- sample.split(dataset, SplitRatio=0.7)  # for training data
training_set <- subset(dataset, split==TRUE)
test_set <- subset(dataset, split==FALSE)
```

### Select only the numeric predictors and then impute missing data


```R
# Create the training and tests dataframe with only numeric predictors
nums <- unlist(lapply(training_set, is.numeric))
training_set_nums <- training_set[, nums]
test_set_nums <- test_set[, nums]

# Show the number of NAs in each field for the training set
apply(training_set_nums, 2, function(x) {sum(is.na(x))})
```

<dl class=dl-horizontal>
	<dt>Id</dt>
		<dd>0</dd>
	<dt>MSSubClass</dt>
		<dd>0</dd>
	<dt>LotFrontage</dt>
		<dd>176</dd>
	<dt>LotArea</dt>
		<dd>0</dd>
	<dt>OverallQual</dt>
		<dd>0</dd>
	<dt>OverallCond</dt>
		<dd>0</dd>
	<dt>YearBuilt</dt>
		<dd>0</dd>
	<dt>YearRemodAdd</dt>
		<dd>0</dd>
	<dt>MasVnrArea</dt>
		<dd>5</dd>
	<dt>BsmtFinSF1</dt>
		<dd>0</dd>
	<dt>BsmtFinSF2</dt>
		<dd>0</dd>
	<dt>BsmtUnfSF</dt>
		<dd>0</dd>
	<dt>TotalBsmtSF</dt>
		<dd>0</dd>
	<dt>X1stFlrSF</dt>
		<dd>0</dd>
	<dt>X2ndFlrSF</dt>
		<dd>0</dd>
	<dt>LowQualFinSF</dt>
		<dd>0</dd>
	<dt>GrLivArea</dt>
		<dd>0</dd>
	<dt>BsmtFullBath</dt>
		<dd>0</dd>
	<dt>BsmtHalfBath</dt>
		<dd>0</dd>
	<dt>FullBath</dt>
		<dd>0</dd>
	<dt>HalfBath</dt>
		<dd>0</dd>
	<dt>BedroomAbvGr</dt>
		<dd>0</dd>
	<dt>KitchenAbvGr</dt>
		<dd>0</dd>
	<dt>TotRmsAbvGrd</dt>
		<dd>0</dd>
	<dt>Fireplaces</dt>
		<dd>0</dd>
	<dt>GarageYrBlt</dt>
		<dd>53</dd>
	<dt>GarageCars</dt>
		<dd>0</dd>
	<dt>GarageArea</dt>
		<dd>0</dd>
	<dt>WoodDeckSF</dt>
		<dd>0</dd>
	<dt>OpenPorchSF</dt>
		<dd>0</dd>
	<dt>EnclosedPorch</dt>
		<dd>0</dd>
	<dt>X3SsnPorch</dt>
		<dd>0</dd>
	<dt>ScreenPorch</dt>
		<dd>0</dd>
	<dt>PoolArea</dt>
		<dd>0</dd>
	<dt>MiscVal</dt>
		<dd>0</dd>
	<dt>MoSold</dt>
		<dd>0</dd>
	<dt>YrSold</dt>
		<dd>0</dd>
	<dt>SalePrice</dt>
		<dd>0</dd>
</dl>




```R
# Show the number of NAs in each field for the test set
apply(test_set_nums, 2, function(x) {sum(is.na(x))})
```

<dl class=dl-horizontal>
	<dt>Id</dt>
		<dd>0</dd>
	<dt>MSSubClass</dt>
		<dd>0</dd>
	<dt>LotFrontage</dt>
		<dd>83</dd>
	<dt>LotArea</dt>
		<dd>0</dd>
	<dt>OverallQual</dt>
		<dd>0</dd>
	<dt>OverallCond</dt>
		<dd>0</dd>
	<dt>YearBuilt</dt>
		<dd>0</dd>
	<dt>YearRemodAdd</dt>
		<dd>0</dd>
	<dt>MasVnrArea</dt>
		<dd>3</dd>
	<dt>BsmtFinSF1</dt>
		<dd>0</dd>
	<dt>BsmtFinSF2</dt>
		<dd>0</dd>
	<dt>BsmtUnfSF</dt>
		<dd>0</dd>
	<dt>TotalBsmtSF</dt>
		<dd>0</dd>
	<dt>X1stFlrSF</dt>
		<dd>0</dd>
	<dt>X2ndFlrSF</dt>
		<dd>0</dd>
	<dt>LowQualFinSF</dt>
		<dd>0</dd>
	<dt>GrLivArea</dt>
		<dd>0</dd>
	<dt>BsmtFullBath</dt>
		<dd>0</dd>
	<dt>BsmtHalfBath</dt>
		<dd>0</dd>
	<dt>FullBath</dt>
		<dd>0</dd>
	<dt>HalfBath</dt>
		<dd>0</dd>
	<dt>BedroomAbvGr</dt>
		<dd>0</dd>
	<dt>KitchenAbvGr</dt>
		<dd>0</dd>
	<dt>TotRmsAbvGrd</dt>
		<dd>0</dd>
	<dt>Fireplaces</dt>
		<dd>0</dd>
	<dt>GarageYrBlt</dt>
		<dd>28</dd>
	<dt>GarageCars</dt>
		<dd>0</dd>
	<dt>GarageArea</dt>
		<dd>0</dd>
	<dt>WoodDeckSF</dt>
		<dd>0</dd>
	<dt>OpenPorchSF</dt>
		<dd>0</dd>
	<dt>EnclosedPorch</dt>
		<dd>0</dd>
	<dt>X3SsnPorch</dt>
		<dd>0</dd>
	<dt>ScreenPorch</dt>
		<dd>0</dd>
	<dt>PoolArea</dt>
		<dd>0</dd>
	<dt>MiscVal</dt>
		<dd>0</dd>
	<dt>MoSold</dt>
		<dd>0</dd>
	<dt>YrSold</dt>
		<dd>0</dd>
	<dt>SalePrice</dt>
		<dd>0</dd>
</dl>




```R
# Impute missing data using rfImpute
training_set_nums_impute <- rfImpute(SalePrice ~ ., training_set_nums)
test_set_nums_impute <- rfImpute(SalePrice ~ ., test_set_nums)
```

         |      Out-of-bag   |
    Tree |      MSE  %Var(y) |
     300 | 8.511e+08    14.40 |
         |      Out-of-bag   |
    Tree |      MSE  %Var(y) |
     300 | 8.585e+08    14.53 |
         |      Out-of-bag   |
    Tree |      MSE  %Var(y) |
     300 | 8.603e+08    14.56 |
         |      Out-of-bag   |
    Tree |      MSE  %Var(y) |
     300 | 8.545e+08    14.46 |
         |      Out-of-bag   |
    Tree |      MSE  %Var(y) |
     300 | 8.515e+08    14.41 |
         |      Out-of-bag   |
    Tree |      MSE  %Var(y) |
     300 | 7.563e+08    10.52 |
         |      Out-of-bag   |
    Tree |      MSE  %Var(y) |
     300 | 7.763e+08    10.80 |
         |      Out-of-bag   |
    Tree |      MSE  %Var(y) |
     300 | 8.117e+08    11.29 |
         |      Out-of-bag   |
    Tree |      MSE  %Var(y) |
     300 | 8.224e+08    11.44 |
         |      Out-of-bag   |
    Tree |      MSE  %Var(y) |
     300 | 8.131e+08    11.31 |
    

### Create the predictor and target variables from the imputed data

Then get the MAEs for multiple ntree values.


```R
# Create the predictor variable
X <- subset(training_set_nums_impute, select = -c(Id, SalePrice))

# Select the target variable and call it y
y <- training_set_nums_impute$SalePrice

# Create a function to calculate the Mean Absolute Error
mae <- function(error)
{
  mean(abs(error))
}

# Create a function to get the MAE of a random forest
getMae_forest <- function(X, y, test_data, n) {
  set.seed(42)
  regressor <- randomForest(x=X, y=y, ntree=n)
  y_prediction <- predict(regressor, newdata=test_data)
  y_test <- test_data$SalePrice
  error <- (y_prediction - y_test)
  print(paste("ntree of ", n, " has an MAE of ", mae(error), sep=""))
}

# Loop through multiple ntree values
ntrees = c(1, 5, 10, 30, 50, 100, 500, 1000, 5000)

for (i in ntrees) {
  getMae_forest(X, y, test_set_nums_impute, i)
}
```

    [1] "ntree of 1 has an MAE of 28209.0257374631"
    [1] "ntree of 5 has an MAE of 20668.9521828909"
    [1] "ntree of 10 has an MAE of 19227.5275073746"
    [1] "ntree of 30 has an MAE of 17640.3471914946"
    [1] "ntree of 50 has an MAE of 17380.0999144543"
    [1] "ntree of 100 has an MAE of 17154.4681511799"
    [1] "ntree of 500 has an MAE of 16985.3050430152"
    [1] "ntree of 1000 has an MAE of 17151.1359503793"
    [1] "ntree of 5000 has an MAE of 17120.9017568099"
    

ntree of 500 has lowest MAE

### Encoding categorical variables

Random forest in R converts categorical variables to dummy for you.


```R
# Split data into training and validation data, for both predictors and target.
set.seed(42)
split <- sample.split(dataset, SplitRatio=0.7)  # for training data
training_set <- subset(dataset, split==TRUE)
test_set <- subset(dataset, split==FALSE)

# Impute missing data using rfImpute
training_set_impute <- rfImpute(SalePrice ~ ., training_set)
test_set_impute <- rfImpute(SalePrice ~ ., test_set)

# Create the predictor variable
X <- subset(training_set_impute, select = -c(Id, SalePrice))

# Select the target variable and call it y
y <- training_set_impute$SalePrice

# Loop through multiple ntree values
ntrees = c(1, 5, 10, 30, 50, 100, 500, 1000, 5000)

for (i in ntrees) {
  getMae_forest(X, y, test_set_impute, i)
}
```

         |      Out-of-bag   |
    Tree |      MSE  %Var(y) |
     300 | 7.747e+08    13.11 |
         |      Out-of-bag   |
    Tree |      MSE  %Var(y) |
     300 | 8.007e+08    13.55 |
         |      Out-of-bag   |
    Tree |      MSE  %Var(y) |
     300 | 7.933e+08    13.42 |
         |      Out-of-bag   |
    Tree |      MSE  %Var(y) |
     300 | 7.677e+08    12.99 |
         |      Out-of-bag   |
    Tree |      MSE  %Var(y) |
     300 | 8.034e+08    13.59 |
         |      Out-of-bag   |
    Tree |      MSE  %Var(y) |
     300 | 8.134e+08    11.31 |
         |      Out-of-bag   |
    Tree |      MSE  %Var(y) |
     300 | 8.524e+08    11.85 |
         |      Out-of-bag   |
    Tree |      MSE  %Var(y) |
     300 | 8.846e+08    12.30 |
         |      Out-of-bag   |
    Tree |      MSE  %Var(y) |
     300 | 8.484e+08    11.80 |
         |      Out-of-bag   |
    Tree |      MSE  %Var(y) |
     300 | 9.14e+08    12.71 |
    [1] "ntree of 1 has an MAE of 32758.8626474926"
    [1] "ntree of 5 has an MAE of 21188.8914823009"
    [1] "ntree of 10 has an MAE of 19193.7760803835"
    [1] "ntree of 30 has an MAE of 18162.4315843166"
    [1] "ntree of 50 has an MAE of 18186.106070059"
    [1] "ntree of 100 has an MAE of 17762.6606873156"
    [1] "ntree of 500 has an MAE of 17872.5035027286"
    [1] "ntree of 1000 has an MAE of 17851.2130004056"
    [1] "ntree of 5000 has an MAE of 17890.0051326422"
    

ntree of 1000 has the lowest MAE but that could be because of a bias with random forests. A quote from Wikipedia’s article on random forests (in the [variable importance section](https://en.wikipedia.org/wiki/Random_forest#Variable_importance)):
  
  "For data including categorical variables with different number of levels, random forests are biased in favor of those attributes with more levels."

That tells me we are including too many and going forward I'll need to limit them by using numeric variables only or some mix of the two. I'll have to do some research to determine the best way to determine a mix then.

### Next steps

I'm going to work on the next few sections in R but I'm not sure how some of it will translate to R. I know I can do XGBoost but I'm not entirely sure about Partial Dependence Plots. I can also do Cross Validation and probably Data Leakage, but I'm unsure about Pipelines. I know I can do pipelines with %>% and I think I can do similar to pipelines in Python with that. I'll have to play around with it.

Before doing all that I think I'm going to do a post on how I set up my R environment. I like what I have set up now and would like to share it.
