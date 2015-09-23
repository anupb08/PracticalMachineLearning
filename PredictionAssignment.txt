
# Practical Machine Learning - Course Project

## Overview
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 

The goal of this project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. 

## Data cleaning and Feature selection

```r
set.seed(32)
pmlData <- read.csv("pml-training.csv", header=TRUE)
# Removing near zero variance predictors
nzv <- nearZeroVar(pmlData)
pmlData <- pmlData[, -nzv] 
dim(pmlData)
```

```
## [1] 19622   100
```

```r
# Remove columns which has more than 80% NAs 
NAratio <- colSums(is.na(pmlData))/nrow(pmlData)
NAColumns <- which(NAratio > 0.80)
pmlData <- pmlData[, -NAColumns] # remove columns which has 80% NAs
dim(pmlData)
```

```
## [1] 19622    59
```

### Create two separate dataset one for all numeric columns and another for factors

```r
numericData <- pmlData[sapply(pmlData, is.numeric)] 
factorData <- pmlData[sapply(pmlData, is.factor)]
```

### Identifying Correlated Predictors

```r
dataCor <-  cor(numericData) # create correlation matrix
summary(dataCor[upper.tri(dataCor)])
```

```
##      Min.   1st Qu.    Median      Mean   3rd Qu.      Max. 
## -0.992000 -0.102000  0.001729  0.001405  0.084720  0.980900
```

```r
# calculate predictors which has correlation value more than 0.9
highlyCorData <- findCorrelation(dataCor, cutoff = 0.90) 
filteredData <- numericData[,-highlyCorData] # remove highly correleated predictors
dataCor2 <- cor(filteredData)
summary(dataCor2[upper.tri(dataCor2)])
```

```
##      Min.   1st Qu.    Median      Mean   3rd Qu.      Max. 
## -0.884200 -0.108600  0.001377  0.001198  0.089530  0.849100
```

```r
dim(filteredData)
```

```
## [1] 19622    49
```

Around 49 variables are highly correlated, so we removed them. 

### Remove some more unrelevant variables 
 As first 4 variables ("X", "raw_timestamp_part_1", "raw_timestamp_part_2", "num_window") has no relevance, we remove these from the dataset.


```r
names(filteredData)
```

```
##  [1] "X"                    "raw_timestamp_part_1" "raw_timestamp_part_2"
##  [4] "num_window"           "pitch_belt"           "yaw_belt"            
##  [7] "total_accel_belt"     "gyros_belt_x"         "gyros_belt_y"        
## [10] "gyros_belt_z"         "magnet_belt_x"        "magnet_belt_y"       
## [13] "magnet_belt_z"        "roll_arm"             "pitch_arm"           
## [16] "yaw_arm"              "total_accel_arm"      "gyros_arm_y"         
## [19] "gyros_arm_z"          "accel_arm_x"          "accel_arm_y"         
## [22] "accel_arm_z"          "magnet_arm_x"         "magnet_arm_y"        
## [25] "magnet_arm_z"         "roll_dumbbell"        "pitch_dumbbell"      
## [28] "yaw_dumbbell"         "total_accel_dumbbell" "gyros_dumbbell_y"    
## [31] "accel_dumbbell_x"     "accel_dumbbell_y"     "accel_dumbbell_z"    
## [34] "magnet_dumbbell_x"    "magnet_dumbbell_y"    "magnet_dumbbell_z"   
## [37] "roll_forearm"         "pitch_forearm"        "yaw_forearm"         
## [40] "total_accel_forearm"  "gyros_forearm_x"      "gyros_forearm_y"     
## [43] "gyros_forearm_z"      "accel_forearm_x"      "accel_forearm_y"     
## [46] "accel_forearm_z"      "magnet_forearm_x"     "magnet_forearm_y"    
## [49] "magnet_forearm_z"
```

```r
finalData <- filteredData[,c(-1,-2,-3,-4)]
# Also in factorData, only relevant feature is "classe", so we merge this variable to finalData 
names(factorData)
```

```
## [1] "user_name"      "cvtd_timestamp" "classe"
```

```r
finalData <- data.frame(cbind(finalData,classe=factorData$classe))
dim(finalData)
```

```
## [1] 19622    46
```

```r
filteredData <- NULL
numericData <- NULL
factorData <- NULL
pmlData <- NULL
```

## Spliting dataset for training and cross-validation data 

```r
inTrain <- createDataPartition(y = finalData$classe, list = FALSE, p=0.7)
trainData <- finalData[inTrain,]
testData <- finalData[-inTrain,]
```

## Build a Predictive Model
We use *train()* function from **caret** package for **random forest** model generation. 

For training control, we use Bootstrap resampling method for 2 iterations 

```r
control <- trainControl(method = "boot", number = 2) # Here k =2
```
### We use Random Forest algorithm for Model creation

```r
library(caret)
library(randomForest)
# Train the data by random forest classification technique.
model.forest <- train(classe~., trainData, method = "rf",  trControl = control) 
model.forest
```

```
## Random Forest 
## 
## 13737 samples
##    45 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (2 reps) 
## Summary of sample sizes: 13737, 13737 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD   Kappa SD    
##    2    0.9841896  0.9800019  0.0003107278  0.0003768634
##   23    0.9874778  0.9841627  0.0021620885  0.0027229237
##   45    0.9751554  0.9685796  0.0046068500  0.0057999029
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 23.
```

### Error estimation with cross validation
First we do error estimate for training data.

```r
predTrain <- predict(model.forest,newdata=trainData)
confusionMatrix(predTrain,trainData$classe)$table
```

```
##           Reference
## Prediction    A    B    C    D    E
##          A 3906    0    0    0    0
##          B    0 2658    0    0    0
##          C    0    0 2396    0    0
##          D    0    0    0 2252    0
##          E    0    0    0    0 2525
```

Accuracy rate is 100% for training data.

Now we check, out of sample error on the "testData" which was created from the original data by splitiing 70:30 ratio 

```r
predTest <- predict(model.forest, newdata = testData[,names(finalData)])
confusionMatrix(predTest,testData$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1673   10    0    0    0
##          B    0 1126    7    0    0
##          C    1    3 1011    6    2
##          D    0    0    8  957    1
##          E    0    0    0    1 1079
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9934         
##                  95% CI : (0.991, 0.9953)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9916         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9994   0.9886   0.9854   0.9927   0.9972
## Specificity            0.9976   0.9985   0.9975   0.9982   0.9998
## Pos Pred Value         0.9941   0.9938   0.9883   0.9907   0.9991
## Neg Pred Value         0.9998   0.9973   0.9969   0.9986   0.9994
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2843   0.1913   0.1718   0.1626   0.1833
## Detection Prevalence   0.2860   0.1925   0.1738   0.1641   0.1835
## Balanced Accuracy      0.9985   0.9936   0.9915   0.9955   0.9985
```
Based on the confusion matrix summary above, the model has out of sample accuracy rate is : 99%

Total 37 has been misclassified out of 5885.  
### Now we will predict on the given testing data

```r
test <- read.csv("pml-testing.csv", header=TRUE)
answers <- predict(model.forest, newdata = test)
answers
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
