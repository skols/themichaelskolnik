# Install and load packages
if (!require("randomForest")) {
  install.packages("randomForest", repos="http://cran.rstudio.com/")
  library(randomForest, lib.loc="~/R/win-library/3.5")
} else {
  library(randomForest, lib.loc="~/R/win-library/3.5")
}

if (!require("dplyr")) {
  install.packages("dplyr", repos="http://cran.rstudio.com/")
  library(dplyr, lib.loc="~/R/win-library/3.5")
} else {
  library(dplyr, lib.loc="~/R/win-library/3.5")
}

if (!require("caTools")) {
  install.packages("caTools", repos="http://cran.rstudio.com/")
  library(caTools, lib.loc="~/R/win-library/3.5")
} else {
  library(caTools, lib.loc="~/R/win-library/3.5")
}

if (!require("rpart")) {
  install.packages("rpart", repos="http://cran.rstudio.com/")
  library(rpart, lib.loc="~/R/win-library/3.5")
} else {
  library(rpart, lib.loc="~/R/win-library/3.5")
}

# Save filepath to variable
training_data_filepath <- "C:/Development/Kaggle/House Prices - Advanced Regression Techniques/train.csv"

# Import data
dataset <- read.csv(training_data_filepath)

### View some stats about the data

# View some stats and information about the data
summary(dataset)

### Split the data set into training and test, then create the predictor and target variables

# Split data into training and validation data, for both predictors and target.
set.seed(42)
split <- sample.split(dataset, SplitRatio=0.7)  # for training data
training_set <- subset(dataset, split==TRUE)
test_set <- subset(dataset, split==FALSE)

# Create the training and tests dataframe with the initial predictors
predictors <- c("LotArea", "YearBuilt", "X1stFlrSF", "X2ndFlrSF",
                "FullBath", "BedroomAbvGr", "TotRmsAbvGrd", "SalePrice")
training_set <- training_set %>%
  select(predictors)
test_set <- test_set %>%
  select(predictors)

# Create the predictor variable
X <- training_set %>%
  select(-SalePrice)

# Select the target variable and call it y
y <- training_set$SalePrice

### Predict values with a Decision Tree using rpart

# Fitting Decision Tree to the training data
formula=SalePrice ~ .

regressor <- rpart(formula=formula, data=training_set,
                   control=rpart.control(cp=.01))

y_pred <- predict(regressor, test_set)

# View a summary of the predicted values
summary(y_pred)

### Create a function to get the Mean Absolute Error (or MAE)

# Calculating the Mean Absolute Error
mae <- function(error)
{
  mean(abs(error))
}

# Get the MAE
y_test <- test_set$SalePrice
error <- (y_test - y_pred)
mae(error)

### Create a function to compare the MAE for different cp values

# Create the function
getMae_rpart <- function(formula, training_data, test_data, n) {
  set.seed(42)
  regressor_rpart <- rpart(formula=formula, data=training_data,
                           control=rpart.control(cp=n))
  y_prediction <- predict(regressor_rpart, newdata=test_data)
  y_test <- test_data$SalePrice
  error <- (y_test - y_prediction)
  print(paste("cp of ", n, " has an MAE of ", mae(error), sep=""))
}

#### Set up the formula variable and splits, then loop through the values and call the function

# Set the formula variable
formula <- SalePrice ~ .

# Loop through multiple ntree values
cps <- c(.5, .1, .05, .02, .01, .005, .003, .001, .0005, .0001)

for (i in cps) {
  getMae_rpart(formula, training_set, test_set, i)
}

#### MAE continues to decrease as the cp decreases.

### Predict values with a Random Forest

# Fitting Random Forest Regression to the dataset
regressor <- randomForest(x=X, y=y, ntree=100)

# Predicting a new result
y_pred <- predict(regressor, newdata=test_set)

# Get the MAE
y_test <- test_set$SalePrice
error <- (y_pred - y_test)
mae(error)

### Create a function to compare the MAE for different ntree values

# Create the function
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
  getMae_forest(X, y, test_set, i)
}

# ntree of 1000 has the lowest MAE.
# 
# That's all for this post. The more I use R, the more I like it. Python and R both have their advantages though.
# 
# Hopefully the second part doesn't take me nearly as long. Until then!


# Level 2
# Split data into training and validation data, for both predictors and target.
set.seed(42)
split <- sample.split(dataset, SplitRatio=0.7)  # for training data
training_set <- subset(dataset, split==TRUE)
test_set <- subset(dataset, split==FALSE)

# Create the training and tests dataframe with only numeric predictors
nums <- unlist(lapply(training_set, is.numeric))
training_set_nums <- training_set[, nums]
test_set_nums <- test_set[, nums]

# Show the number of NAs in each field for the training set
apply(training_set_nums, 2, function(x) {sum(is.na(x))})

# Show the number of NAs in each field for the test set
apply(test_set_nums, 2, function(x) {sum(is.na(x))})

# Drop NAs
# training_set <- training_set[complete.cases(training_set), ]
# test_set <- test_set[complete.cases(test_set), ]

# Replace NA with 0
# training_set[is.na(training_set)] <- 0

# Impute missing data using rfImpute
training_set_nums_impute <- rfImpute(SalePrice ~ ., training_set_nums)
test_set_nums_impute <- rfImpute(SalePrice ~ ., test_set_nums)

# Create the predictor variable
X <- subset(training_set_nums_impute, select = -c(Id, SalePrice))

# Select the target variable and call it y
y <- training_set_nums_impute$SalePrice

# Loop through multiple ntree values
ntrees = c(1, 5, 10, 30, 50, 100, 500, 1000, 5000)

for (i in ntrees) {
  getMae_forest(X, y, test_set_nums_impute, i)
}

# ntree of 500 has lowest MAE

### Encoding categorical variables

# Random forest converts categorical variables to dummy for you
# Split data into training and validation data, for both predictors and target.
set.seed(42)
split <- sample.split(dataset, SplitRatio=0.7)  # for training data
training_set <- subset(dataset, split==TRUE)
test_set <- subset(dataset, split==FALSE)

# Using model.matrix
# test_matrix <- model.matrix(SalePrice ~ . -1, training_set)

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

# ntree of 5000 has the lowest MAE but that could be because of a bias with random forests.
# A quote from Wikipedia’s article on random forests (in the variable selection section):
# "For data including categorical variables with different number of levels, random forests
# are biased in favor of those attributes with more levels."
# That tells me we are including too many and going forward I'll use numeric variables only.
