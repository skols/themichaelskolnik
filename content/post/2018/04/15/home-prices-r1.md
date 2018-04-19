---
{
  "title": "My R code from the Level 1 exercises of Kaggle's Learn Machine Learning series",
  "subtitle": "",
  "date": "2018-04-19",
  "slug": "home-prices-r1",
  "tags": ["R", "machine learning"]
}
---
<!--more-->

## *Learn Maching Learning* series on Kaggle in R

This is my R code for the level 1 part of the *Learn Machine Learning* series on Kaggle. I've already done the Python one, which is on Kaggle located [here](https://www.kaggle.com/learn/machine-learning). The data used is from the [*Home Prices: Advanced Regression Techniques*](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) competition.

Originally I had planned on doing both level 1 and level 2 at the same time, but I encountered some issues with my R install and I got busier than expected. I'm publishing level 1 now since it's done and while I've already started the level 2 part, I'll just publish it a little later.

### Load and install packages and load the data


```R
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
```

    Loading required package: randomForest
    randomForest 4.6-14
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
    Loading required package: rpart
    

### View some stats about the data


```R
# View some stats and information about the data
summary(dataset)
```

           Id           MSSubClass       MSZoning     LotFrontage    
     Min.   :   1.0   Min.   : 20.0   C (all):  10   Min.   : 21.00  
     1st Qu.: 365.8   1st Qu.: 20.0   FV     :  65   1st Qu.: 59.00  
     Median : 730.5   Median : 50.0   RH     :  16   Median : 69.00  
     Mean   : 730.5   Mean   : 56.9   RL     :1151   Mean   : 70.05  
     3rd Qu.:1095.2   3rd Qu.: 70.0   RM     : 218   3rd Qu.: 80.00  
     Max.   :1460.0   Max.   :190.0                  Max.   :313.00  
                                                     NA's   :259     
        LotArea        Street      Alley      LotShape  LandContour  Utilities   
     Min.   :  1300   Grvl:   6   Grvl:  50   IR1:484   Bnk:  63    AllPub:1459  
     1st Qu.:  7554   Pave:1454   Pave:  41   IR2: 41   HLS:  50    NoSeWa:   1  
     Median :  9478               NA's:1369   IR3: 10   Low:  36                 
     Mean   : 10517                           Reg:925   Lvl:1311                 
     3rd Qu.: 11602                                                              
     Max.   :215245                                                              
                                                                                 
       LotConfig    LandSlope   Neighborhood   Condition1     Condition2  
     Corner : 263   Gtl:1382   NAmes  :225   Norm   :1260   Norm   :1445  
     CulDSac:  94   Mod:  65   CollgCr:150   Feedr  :  81   Feedr  :   6  
     FR2    :  47   Sev:  13   OldTown:113   Artery :  48   Artery :   2  
     FR3    :   4              Edwards:100   RRAn   :  26   PosN   :   2  
     Inside :1052              Somerst: 86   PosN   :  19   RRNn   :   2  
                               Gilbert: 79   RRAe   :  11   PosA   :   1  
                               (Other):707   (Other):  15   (Other):   2  
       BldgType      HouseStyle   OverallQual      OverallCond      YearBuilt   
     1Fam  :1220   1Story :726   Min.   : 1.000   Min.   :1.000   Min.   :1872  
     2fmCon:  31   2Story :445   1st Qu.: 5.000   1st Qu.:5.000   1st Qu.:1954  
     Duplex:  52   1.5Fin :154   Median : 6.000   Median :5.000   Median :1973  
     Twnhs :  43   SLvl   : 65   Mean   : 6.099   Mean   :5.575   Mean   :1971  
     TwnhsE: 114   SFoyer : 37   3rd Qu.: 7.000   3rd Qu.:6.000   3rd Qu.:2000  
                   1.5Unf : 14   Max.   :10.000   Max.   :9.000   Max.   :2010  
                   (Other): 19                                                  
      YearRemodAdd    RoofStyle       RoofMatl     Exterior1st   Exterior2nd 
     Min.   :1950   Flat   :  13   CompShg:1434   VinylSd:515   VinylSd:504  
     1st Qu.:1967   Gable  :1141   Tar&Grv:  11   HdBoard:222   MetalSd:214  
     Median :1994   Gambrel:  11   WdShngl:   6   MetalSd:220   HdBoard:207  
     Mean   :1985   Hip    : 286   WdShake:   5   Wd Sdng:206   Wd Sdng:197  
     3rd Qu.:2004   Mansard:   7   ClyTile:   1   Plywood:108   Plywood:142  
     Max.   :2010   Shed   :   2   Membran:   1   CemntBd: 61   CmentBd: 60  
                                   (Other):   2   (Other):128   (Other):136  
       MasVnrType    MasVnrArea     ExterQual ExterCond  Foundation  BsmtQual  
     BrkCmn : 15   Min.   :   0.0   Ex: 52    Ex:   3   BrkTil:146   Ex  :121  
     BrkFace:445   1st Qu.:   0.0   Fa: 14    Fa:  28   CBlock:634   Fa  : 35  
     None   :864   Median :   0.0   Gd:488    Gd: 146   PConc :647   Gd  :618  
     Stone  :128   Mean   : 103.7   TA:906    Po:   1   Slab  : 24   TA  :649  
     NA's   :  8   3rd Qu.: 166.0             TA:1282   Stone :  6   NA's: 37  
                   Max.   :1600.0                       Wood  :  3             
                   NA's   :8                                                   
     BsmtCond    BsmtExposure BsmtFinType1   BsmtFinSF1     BsmtFinType2
     Fa  :  45   Av  :221     ALQ :220     Min.   :   0.0   ALQ :  19   
     Gd  :  65   Gd  :134     BLQ :148     1st Qu.:   0.0   BLQ :  33   
     Po  :   2   Mn  :114     GLQ :418     Median : 383.5   GLQ :  14   
     TA  :1311   No  :953     LwQ : 74     Mean   : 443.6   LwQ :  46   
     NA's:  37   NA's: 38     Rec :133     3rd Qu.: 712.2   Rec :  54   
                              Unf :430     Max.   :5644.0   Unf :1256   
                              NA's: 37                      NA's:  38   
       BsmtFinSF2        BsmtUnfSF       TotalBsmtSF      Heating     HeatingQC
     Min.   :   0.00   Min.   :   0.0   Min.   :   0.0   Floor:   1   Ex:741   
     1st Qu.:   0.00   1st Qu.: 223.0   1st Qu.: 795.8   GasA :1428   Fa: 49   
     Median :   0.00   Median : 477.5   Median : 991.5   GasW :  18   Gd:241   
     Mean   :  46.55   Mean   : 567.2   Mean   :1057.4   Grav :   7   Po:  1   
     3rd Qu.:   0.00   3rd Qu.: 808.0   3rd Qu.:1298.2   OthW :   2   TA:428   
     Max.   :1474.00   Max.   :2336.0   Max.   :6110.0   Wall :   4            
                                                                               
     CentralAir Electrical     X1stFlrSF      X2ndFlrSF     LowQualFinSF    
     N:  95     FuseA:  94   Min.   : 334   Min.   :   0   Min.   :  0.000  
     Y:1365     FuseF:  27   1st Qu.: 882   1st Qu.:   0   1st Qu.:  0.000  
                FuseP:   3   Median :1087   Median :   0   Median :  0.000  
                Mix  :   1   Mean   :1163   Mean   : 347   Mean   :  5.845  
                SBrkr:1334   3rd Qu.:1391   3rd Qu.: 728   3rd Qu.:  0.000  
                NA's :   1   Max.   :4692   Max.   :2065   Max.   :572.000  
                                                                            
       GrLivArea     BsmtFullBath     BsmtHalfBath        FullBath    
     Min.   : 334   Min.   :0.0000   Min.   :0.00000   Min.   :0.000  
     1st Qu.:1130   1st Qu.:0.0000   1st Qu.:0.00000   1st Qu.:1.000  
     Median :1464   Median :0.0000   Median :0.00000   Median :2.000  
     Mean   :1515   Mean   :0.4253   Mean   :0.05753   Mean   :1.565  
     3rd Qu.:1777   3rd Qu.:1.0000   3rd Qu.:0.00000   3rd Qu.:2.000  
     Max.   :5642   Max.   :3.0000   Max.   :2.00000   Max.   :3.000  
                                                                      
        HalfBath       BedroomAbvGr    KitchenAbvGr   KitchenQual  TotRmsAbvGrd   
     Min.   :0.0000   Min.   :0.000   Min.   :0.000   Ex:100      Min.   : 2.000  
     1st Qu.:0.0000   1st Qu.:2.000   1st Qu.:1.000   Fa: 39      1st Qu.: 5.000  
     Median :0.0000   Median :3.000   Median :1.000   Gd:586      Median : 6.000  
     Mean   :0.3829   Mean   :2.866   Mean   :1.047   TA:735      Mean   : 6.518  
     3rd Qu.:1.0000   3rd Qu.:3.000   3rd Qu.:1.000               3rd Qu.: 7.000  
     Max.   :2.0000   Max.   :8.000   Max.   :3.000               Max.   :14.000  
                                                                                  
     Functional    Fireplaces    FireplaceQu   GarageType   GarageYrBlt  
     Maj1:  14   Min.   :0.000   Ex  : 24    2Types :  6   Min.   :1900  
     Maj2:   5   1st Qu.:0.000   Fa  : 33    Attchd :870   1st Qu.:1961  
     Min1:  31   Median :1.000   Gd  :380    Basment: 19   Median :1980  
     Min2:  34   Mean   :0.613   Po  : 20    BuiltIn: 88   Mean   :1979  
     Mod :  15   3rd Qu.:1.000   TA  :313    CarPort:  9   3rd Qu.:2002  
     Sev :   1   Max.   :3.000   NA's:690    Detchd :387   Max.   :2010  
     Typ :1360                               NA's   : 81   NA's   :81    
     GarageFinish   GarageCars      GarageArea     GarageQual  GarageCond 
     Fin :352     Min.   :0.000   Min.   :   0.0   Ex  :   3   Ex  :   2  
     RFn :422     1st Qu.:1.000   1st Qu.: 334.5   Fa  :  48   Fa  :  35  
     Unf :605     Median :2.000   Median : 480.0   Gd  :  14   Gd  :   9  
     NA's: 81     Mean   :1.767   Mean   : 473.0   Po  :   3   Po  :   7  
                  3rd Qu.:2.000   3rd Qu.: 576.0   TA  :1311   TA  :1326  
                  Max.   :4.000   Max.   :1418.0   NA's:  81   NA's:  81  
                                                                          
     PavedDrive   WoodDeckSF      OpenPorchSF     EnclosedPorch      X3SsnPorch    
     N:  90     Min.   :  0.00   Min.   :  0.00   Min.   :  0.00   Min.   :  0.00  
     P:  30     1st Qu.:  0.00   1st Qu.:  0.00   1st Qu.:  0.00   1st Qu.:  0.00  
     Y:1340     Median :  0.00   Median : 25.00   Median :  0.00   Median :  0.00  
                Mean   : 94.24   Mean   : 46.66   Mean   : 21.95   Mean   :  3.41  
                3rd Qu.:168.00   3rd Qu.: 68.00   3rd Qu.:  0.00   3rd Qu.:  0.00  
                Max.   :857.00   Max.   :547.00   Max.   :552.00   Max.   :508.00  
                                                                                   
      ScreenPorch        PoolArea        PoolQC       Fence      MiscFeature
     Min.   :  0.00   Min.   :  0.000   Ex  :   2   GdPrv:  59   Gar2:   2  
     1st Qu.:  0.00   1st Qu.:  0.000   Fa  :   2   GdWo :  54   Othr:   2  
     Median :  0.00   Median :  0.000   Gd  :   3   MnPrv: 157   Shed:  49  
     Mean   : 15.06   Mean   :  2.759   NA's:1453   MnWw :  11   TenC:   1  
     3rd Qu.:  0.00   3rd Qu.:  0.000               NA's :1179   NA's:1406  
     Max.   :480.00   Max.   :738.000                                       
                                                                            
        MiscVal             MoSold           YrSold        SaleType   
     Min.   :    0.00   Min.   : 1.000   Min.   :2006   WD     :1267  
     1st Qu.:    0.00   1st Qu.: 5.000   1st Qu.:2007   New    : 122  
     Median :    0.00   Median : 6.000   Median :2008   COD    :  43  
     Mean   :   43.49   Mean   : 6.322   Mean   :2008   ConLD  :   9  
     3rd Qu.:    0.00   3rd Qu.: 8.000   3rd Qu.:2009   ConLI  :   5  
     Max.   :15500.00   Max.   :12.000   Max.   :2010   ConLw  :   5  
                                                        (Other):   9  
     SaleCondition    SalePrice     
     Abnorml: 101   Min.   : 34900  
     AdjLand:   4   1st Qu.:129975  
     Alloca :  12   Median :163000  
     Family :  20   Mean   :180921  
     Normal :1198   3rd Qu.:214000  
     Partial: 125   Max.   :755000  
                                    


### Split the data set into training and test, then create the predictor and target variables


```R
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
```

### Predict values with a Decision Tree using rpart


```R
# Fitting Decision Tree to the training data
formula=SalePrice ~ .

regressor <- rpart(formula=formula, data=training_set,
                   control=rpart.control(cp=.01))

# Get predicted prices
y_pred <- predict(regressor, test_set)

# View a summary of the predicted values
summary(y_pred)
```

       Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
     115718  115718  149822  175554  200484  480209 


### Create a function to get the Mean Absolute Error (or MAE)


```R
# Calculating the Mean Absolute Error
mae <- function(error)
{
  mean(abs(error))
}

# Get the MAE
y_test <- test_set$SalePrice
error <- (y_test - y_pred)
mae(error)
```

29589.8455005301


### Create a function to compare the MAE for different cp values


```R
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
```

Set up the formula variable and cp values, then loop through the values and call the function.


```R
# Set the formula variable
formula <- SalePrice ~ .

# Loop through multiple ntree values
cps <- c(.5, .1, .05, .02, .01, .005, .003, .001, .0005, .0001)

for (i in cps) {
  getMae_rpart(formula, training_set, test_set, i)
}
```

    [1] "cp of 0.5 has an MAE of 57536.8983354404"
    [1] "cp of 0.1 has an MAE of 40654.9088557541"
    [1] "cp of 0.05 has an MAE of 36460.7134426164"
    [1] "cp of 0.02 has an MAE of 33492.3580079057"
    [1] "cp of 0.01 has an MAE of 29589.8455005301"
    [1] "cp of 0.005 has an MAE of 29136.0138171344"
    [1] "cp of 0.003 has an MAE of 29583.0145339228"
    [1] "cp of 0.001 has an MAE of 27909.4547519322"
    [1] "cp of 5e-04 has an MAE of 27597.8067312116"
    [1] "cp of 1e-04 has an MAE of 27419.4284590988"
    

MAE continues to decrease as the cp decreases.

### Predict values with a Random Forest


```R
# Fitting Random Forest Regression to the dataset
regressor <- randomForest(x=X, y=y, ntree=100)

# Predicting a new result
y_pred <- predict(regressor, newdata=test_set)

# Get the MAE
y_test <- test_set$SalePrice
error <- (y_pred - y_test)
mae(error)
```

23217.1818323031


### Create a function to compare the MAE for different ntree values


```R
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
```

    [1] "ntree of 1 has an MAE of 35761.9752775473"
    [1] "ntree of 5 has an MAE of 25399.3227531454"
    [1] "ntree of 10 has an MAE of 24226.9883834123"
    [1] "ntree of 30 has an MAE of 23401.1638509278"
    [1] "ntree of 50 has an MAE of 23610.084126271"
    [1] "ntree of 100 has an MAE of 23260.3606851458"
    [1] "ntree of 500 has an MAE of 23166.618382558"
    [1] "ntree of 1000 has an MAE of 23113.7696443243"
    [1] "ntree of 5000 has an MAE of 23172.7757985064"
    

ntree of 1000 has the lowest MAE.

That's all for this post. The more I use R, the more I like it. Python and R both have their advantages though.

Hopefully the second part doesn't take me nearly as long. Until then!
