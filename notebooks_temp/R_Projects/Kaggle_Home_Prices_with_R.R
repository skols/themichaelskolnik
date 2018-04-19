# Install and load packages
if (!require("randomForest")) {
  install.packages("randomForest", repos="http://cran.rstudio.com/")
  library(randomForest)
}

if (!require("dplyr")) {
  install.packages("dplyr", repos="http://cran.rstudio.com/")
  library(dplyr)
}

if (!require("caTools")) {
  install.packages("caTools", repos="http://cran.rstudio.com/")
  library(caTools)
}

if (!require("rpart")) {
  install.packages("rpart", repos="http://cran.rstudio.com/")
  library(rpart)
}

# Save filepath to variable
training_data_filepath <- "C:/Development/Kaggle/House Prices - Advanced Regression Techniques/train.csv"

# Import data
dataset <- read.csv(training_data_filepath)

# View some stats and information about the data
summary(dataset)

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

# Fitting Random Forest Regression to the dataset
regressor <- randomForest(x=X, y=y, ntree=100)

# Predicting a new result
y_pred <- predict(regressor, newdata=test_set)

# Calculating the Mean Absolute Error
mae <- function(error)
{
  mean(abs(error))
}

# Get the MAE
y_test <- test_set$SalePrice
error <- (y_pred - y_test)
mae(error)

# Create a function to get the MAE for multiple ntree values
getMae <- function(X, y, test_data, n) {
  set.seed(42)
  regressor <- randomForest(x=X, y=y, ntree=n)
  y_prediction <- predict(regressor, newdata=test_data)
  y_test <- test_data$SalePrice
  error <- (y_prediction - y_test)
  print(paste("ntree equals ", n, ": ", mae(error), sep=""))
}

# Loop through multiple ntree values
ntrees = c(10, 50, 100, 500, 1000, 5000)

for (i in ntrees) {
  getMae(X, y, test_set, i)
}

# 1000 has the lowest MAE


# Level 2
# Split data into training and validation data, for both predictors and target.
set.seed(42)
split <- sample.split(dataset, SplitRatio=0.7)  # for training data
training_set <- subset(dataset, split==TRUE)
test_set <- subset(dataset, split==FALSE)

# Create the training and tests dataframe with only numeric predictors
nums <- unlist(lapply(training_set, is.numeric))
training_set <- training_set[, nums]
test_set <- test_set[, nums]

# Show the number of NAs in each field
apply(training_set, 2, function(x) {sum(is.na(x))})

# Drop NAs
# training_set <- training_set[complete.cases(training_set), ]
# test_set <- test_set[complete.cases(test_set), ]

# Replace NA with 0
# training_set[is.na(training_set)] <- 0


# Create the predictor variable
X <- subset(training_set, select = -c(Id, SalePrice))

# Select the target variable and call it y
y <- training_set$SalePrice

# Predicting a new result
y_pred <- predict(regressor, newdata=test_set)
