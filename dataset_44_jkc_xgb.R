# Advanced Data Analytics
# dataset_44_spambase
# Campbell

# loading
rm(list=ls())
spam<-read.csv("/Users/kent/Library/Mobile Documents/com~apple~CloudDocs/Leeds/Spring/MSBX5415-002AdvancedDataAnalytics/Project/dataset_44_spambase.csv",header=T,na.strings=c("","NA"))

# review
head(spam)
tail(spam)
dim(spam)
str(spam)
summary(spam)
names<-names(spam)
print(names)
column_names<-colnames(spam)

#
as.factor(spam$class)

# Look at amount of zero's
colSums(is.na(spam))
for (i in names(spam)){
  print(c('Percentage of zeros in ',i,sum(spam[,i]==0)/nrow(spam)))}

# Look at skewness
library(e1071)  #library for skewness
for (i in names(spam)){
  print(c('Skewness in ',i,skewness(spam[,i])))}

for (i in names(spam)){
  hist(spam[,i],main=i,breaks=50)}

# Boxplots
for (i in names(spam)){
  boxplot(spam[,i],main=i)}

# Correlations
#pairs(spam)   # margins issue
M <- cor(spam)
library(corrplot)
corrplot(M,method="color",order="FPC")

## XGBOOST
require(xgboost)
# Convert the class factor to an integer class starting at 0
# This is picky, but it's a requirement for XGBoost
class<-spam$class
label<-as.integer(spam$class)
spam$class<-NULL

# Split the data for training and testing (70/20/10 train/test/holdout split)
n<-nrow(spam)                                 # 4601 total rows
set.seed(123)
train_index<-sample(n,floor(0.70*n))          # 70% -> 3220 rows
train_data<-as.matrix(spam[train_index,])
train_label<-label[train_index]

testholdout_data<-spam[-train_index,]         # 30%
testholdout_label<-label[-train_index]
nth<-nrow(testholdout_data)
th_index<-sample(nth,floor(0.67*nth))         # split to 20%, 10% -> 925, 456 rows
test_data<-as.matrix(testholdout_data[th_index,])
test_label<-testholdout_label[th_index]
holdout_data<-as.matrix(testholdout_data[-th_index,])
holdout_label<-testholdout_label[-th_index]
#test_data<-as.matrix(spam[-train_index,])
#test_label<-label[-train_index]

# Transform the two data sets into xgb.Matrix
xgb_train<-xgb.DMatrix(data=train_data,label=train_label)
xgb_test<-xgb.DMatrix(data=test_data,label=test_label)

# get the number of negative & positive cases in our data
negative_cases <- sum(train_label==0)
postive_cases <- sum(train_label==1)

# Define the DEFAULT parameters for classification
params<-list(booster="gbtree",                       # default
             objective="binary:logistic",            # the objective function
             eta=0.3,                                # 0.1
             gamma=0,
             max_depth=6,                            # the maximum depth of each decision tree
             min_child_weight=1,
#             scale_pos_weight = negative_cases/postive_cases,   # NOT DEFAULT control for imbalanced classes
             subsample=1,
             colsample_bytree=1)
set.seed(123)
xgbcv<-xgb.cv(params=params,
              data=xgb_train,
              nrounds=100,                           # 10, 2
              nfold=5,
 #             metrics = list("rmse","auc"),
              showsd=T,
              stratified=T,
              print_every_n=10,
              early_stopping_rounds=10,             # number of boosting rounds, 3
              maximize=F)

# Review the model and results
xgbcv
#### Best iteration:
#### iter train_logloss_mean train_logloss_std test_logloss_mean test_logloss_std
#### 42          0.0444324        0.00299472         0.1390574       0.02348673
min(xgbcv$evaluation_log$test_logloss_mean)

#Train the XGBoost classifer
set.seed(123)
xgb1<-xgb.train(params=params,
                data=xgb_train,                      # the data
                nrounds=42,
                watchlist=list(val=xgb_test,train=xgb_train),
                print_every_n=10,
                early_stopping_rounds=10,
                maximize=F,
                eval_metric="error")                  # 'logloss'

# Predict outcomes with the test data
xgb1_pred<-predict(xgb1,test_data)
library(pROC)
my_roc1<-roc(test_label,xgb1_pred)
plot(my_roc1,print.thres="best",print.thres.best.method="youden",
     print.thres.best.weights=c(50,0.2))
coords(my_roc1,"best",ret=c("threshold","specificity","sensitivity","accuracy",
                           "precision","recall"),transpose=FALSE)
####           threshold specificity sensitivity  accuracy precision    recall
#### threshold  0.420929   0.9575221   0.9527778 0.9556757 0.9346049 0.9527778


# Importance Matrix
importance_matrix1<-xgb.importance(model=xgb1)
print(importance_matrix1)
xgb.plot.importance(importance_matrix=importance_matrix1)
xgb.dump(xgb1,with_stats=TRUE)
xgb.plot.tree(model=xgb1)


# hyperparameter grid
ntrees<-42
hyper_grid<-expand.grid(
  eta=2/ntrees,
  max_depth=c(3,4,5,6,8,10),
  min_child_weight=c(1,2,3),
  subsample=c(0.5,0.75,1),
  colsample_bytree=c(0.4,0.6,0.8,1),
  gamma=c(0,1,3,10,100,1000),
  error=0,          # a place to dump error results
  trees=0)          # a place to dump required number of trees

# grid search
for(i in seq_len(nrow(hyper_grid))) {
  set.seed(123)
  m<-xgb.cv(
    data=train_data,
    label=train_label,
    nrounds=ntrees,
    objective="binary:logistic",
    early_stopping_rounds=10,
    nfold=5,
    verbose=1,
    eval_metric="error",
    params=list(
      eta=hyper_grid$eta[i],
      max_depth=hyper_grid$max_depth[i],
      min_child_weight=hyper_grid$min_child_weight[i],
      subsample=hyper_grid$subsample[i],
      colsample_bytree=hyper_grid$colsample_bytree[i],
      gamma=hyper_grid$gamma[i])
  )
  #hyper_grid$logloss[i]<-min(m$evaluation_log$test_logloss_mean)
  hyper_grid$error[i]<-min(m$evaluation_log$test_error_mean)
  hyper_grid$trees[i]<-m$best_iteration}
####  Stopping. Best iteration:
####  [1]	train-error:0.391926+0.002729	test-error:0.391929+0.010908

# results
library(magrittr)
library(dplyr)
hyper_grid %>%
  filter(error>0) %>%
  arrange(error) %>%
  glimpse()
#### $ eta              <dbl> 0.04761905
#### $ max_depth        <dbl> 10
#### $ min_child_weight <dbl> 1
#### $ subsample        <dbl> 0.75
#### $ colsample_bytree <dbl> 0.6
#### $ gamma            <dbl> 0
#### $ error            <dbl> 0.0562088
#### $ trees            <dbl> 42

# optimal parameter list from results of grid search
params<-list(
  eta=0.04761905,
  max_depth=10,
  min_child_weight=1,
  subsample=0.75,
  colsample_bytree=0.6,
  gamma=0)

# train final model
set.seed(123)
xgb_final<-xgboost(
  params=params,
  data=train_data,
  label=train_label,
  nrounds=31,
  objective="binary:logistic",
  early_stopping_rounds=10,
  nfold=5,
  verbose = 0,
  eval_metric="error")

xgb_pred_final<-predict(xgb_final,test_data)
library(pROC)
my_roc_final<-roc(test_label,xgb_pred_final)
coords(my_roc_final,"best",ret=c("threshold","specificity","sensitivity","accuracy",
                           "precision","recall"),transpose=FALSE)
####           threshold specificity sensitivity  accuracy precision    recall
#### threshold 0.3437382   0.9380531   0.9444444 0.9405405 0.9066667 0.9444444

# Importance Matrix
importance_matrix<-xgb.importance(model=xgb_final)
print(importance_matrix)
xgb.plot.importance(importance_matrix=importance_matrix)
xgb.dump(xgb_final,with_stats=TRUE)
xgb.plot.tree(model=xgb_final)

# Final model against holdout
xgb_pred_finalvsholdout<-predict(xgb_final,holdout_data)
library(pROC)
my_roc_finalvsholdout<-roc(holdout_label,xgb_pred_finalvsholdout)
coords(my_roc_finalvsholdout,"best",ret=c("threshold","specificity","sensitivity","accuracy",
                                 "precision","recall"),transpose=FALSE)
####           threshold specificity sensitivity  accuracy precision    recall
#### threshold 0.3290529   0.9396226   0.9790576 0.9561404 0.9211823 0.9790576

plot.roc(holdout_label,xgb_pred_finalvsholdout)
plot(my_roc_finalvsholdout,print.thres="best",       # not sure why different...
     print.thres.best.weights=c(50,0.2))

# confusion matrix if want to create for pic
xgb_pred <- ifelse (xgb_pred_finalvsholdout>=0.3290529,1,0)
table(holdout_label, xgb_pred)             #
####              xgb_pred
#### holdout_label   0   1
####             0 249  16
####             1   4 187
#        pred0   pred1
#actual0 TN      FP
#actual1 FN      TP

#Accuracy = (TP+TN)/GrandTotal
#True positive rate, recall, sensitivity = TP/(TP+FN)
#True negative rate, specificity = TN/(TN+FP)
#Precision = TP/(TP+FP)
