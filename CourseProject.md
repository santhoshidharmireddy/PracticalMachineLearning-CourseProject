# CourseProject
Santhoshi Dharmireddi  
##Overview

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

##Data

The training data for this project are available here:

(https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv)

The test data are available here:

(https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv)

The data for this project come from this source: (http://groupware.les.inf.puc-rio.br/har). If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment.

##What you should submit

The goal of your project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. You may use any of the other variables to predict with. You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases.

##Loading the required packages

```r
library(knitr)
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
library(rattle)
```

```
## Rattle: A free graphical interface for data mining with R.
## Version 4.1.0 Copyright (c) 2006-2015 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.
```

```r
library(randomForest)
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
library(rpart.plot)
```

```
## Loading required package: rpart
```

```r
opts_chunk$set(echo=TRUE)
```

##Loading the required data

```r
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", destfile = "pmltraining.csv")

download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", destfile = "pmltesting.csv")

pmltrain<- read.csv("pmltraining.csv", sep=",", na.strings = c("","NA"))
test<- read.csv("pmltesting.csv", sep= ",", na.strings = c("","NA"))
dim(pmltrain)
```

```
## [1] 19622   160
```

```r
dim(test)
```

```
## [1]  20 160
```

##Cleaning the Data

Now I am going to reduce the number of features by removing the variables that are almost always NA, variables with nearly zero variance, and variables that don't make any intuitive sense for prediction. 

```r
#remove variables which are almost always NAs
NAs<- sapply(pmltrain, function(x) mean(is.na(x))) > 0.95
pmltrain1<- pmltrain[,NAs == F]

#remove the values with nearly zero variance
nzv<- nearZeroVar(pmltrain1)
pmltrain1<- pmltrain1[,-nzv]

#remove the variabels which don't make any intuitive sense for prediction. That means remove the first 5 variables
pmltrain1<- pmltrain1[,-(1:5)]
```

Now I want to randomly split the full pmltrain1 data into a smaller training set (traindata) and a test set (testdata).


```r
set.seed(123)
intrain<- createDataPartition(pmltrain1$classe, p=0.6, list = F)
traindata<- pmltrain1[intrain, ]
testdata<- pmltrain1[-intrain, ]
```

##Algorithm Selection

First, I would like to use 3-fold cross validation process to select the optimal tuning parameters for the model. Second, I would like to go with random Forest model to see whether it is an acceptable model.

```r
#appply 3-fold cross validation method on traindata to select the optimal tuning parameters
fitCVmethod<- trainControl(method = "cv", number = 3, verboseIter = F)
#fit model on traindata set
fitmodel<- train(classe ~ ., method = "rf", data = traindata, trControl = fitCVmethod)

#Print final model to see which tuning parameters it chose
fitmodel$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 0.33%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3346    1    0    0    1 0.0005973716
## B    5 2271    2    1    0 0.0035103115
## C    0    7 2045    2    0 0.0043816943
## D    0    0   14 1915    1 0.0077720207
## E    0    1    0    4 2160 0.0023094688
```

From the above output, we can say that it decided to use 500 trees and try 27 variables at each split.

##Algorithm Evaluation

Inorder to evaluate the model, now we have to use fitted model to predict the "classe" label in testdata.
Also, show the confusion matrix to compare the predicted versus actual labels.


```r
#Now use fitted moel to predict the classe label in testdata set.
preds<- predict(fitmodel, newdata = testdata)

#Show the confusion matrix to compare the predicted versus actual labels
confusionMatrix(testdata$classe, preds)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2232    0    0    0    0
##          B    2 1514    2    0    0
##          C    0    3 1365    0    0
##          D    0    0    5 1281    0
##          E    0    0    0    1 1441
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9983          
##                  95% CI : (0.9972, 0.9991)
##     No Information Rate : 0.2847          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9979          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9991   0.9980   0.9949   0.9992   1.0000
## Specificity            1.0000   0.9994   0.9995   0.9992   0.9998
## Pos Pred Value         1.0000   0.9974   0.9978   0.9961   0.9993
## Neg Pred Value         0.9996   0.9995   0.9989   0.9998   1.0000
## Prevalence             0.2847   0.1933   0.1749   0.1634   0.1837
## Detection Rate         0.2845   0.1930   0.1740   0.1633   0.1837
## Detection Prevalence   0.2845   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9996   0.9987   0.9972   0.9992   0.9999
```

From the above confusion matrix output, we can say that the model accuracy is 99.83% and my predicted accuracy for the out-of-sample error is 0.2%.
Since this model has an excellent result, I will use Random Forests model to predict on the test set.

## Re-fitting algorithm on full training data

In order to produce the most accurate predictions, it is good to train the model on full training data instead of using a model that trained on small amount of training data. So, now I am going to fit this model on full training data.
Before training the model on both full training data and test data, we have to clean up the test data as well.


```r
#remove variables which are almost always NAs
NAs<- sapply(test, function(x) mean(is.na(x))) > 0.95
pmltest<- test[,NAs == F]

#remove the values with nearly zero variance
nzv<- nearZeroVar(pmltest)
pmltest<- pmltest[,-nzv]

#remove the variabels which don't make any intuitive sense for prediction. That means remove the first 5 variables
pmltest<- pmltest[,-(1:5)]

#re-fit model using the full traing set (pmltrain1)
fitCVmethod1<- trainControl(method = "cv", number = 3, verboseIter = F)
fitmodel1<- train(classe ~ ., method = "rf", data = pmltrain1, trControl = fitCVmethod1)
preds1<- predict(fitmodel1, newdata=pmltest)
```

## Test Set Predictions - Generating files to submit as answers for the assignment:

Finally, use the above algorithm to predict the labels for the observations in pmltest. Also, write those predictions to individual files.


```r
#prediction on test data set
preds1<- predict(fitmodel1, newdata=pmltest)
#create a function to write predictions to files

pml_write_files<- function(x){
  n<- length(x)
  for(i in 1:n){
   filename<- paste0("problem_id_",i,".txt")
   write.table(x[i],file=filename,row.names = F, col.names = F, quote = F)
  
  }
}

pml_write_files(preds1)
```






