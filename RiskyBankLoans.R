##---------------------- Decision Trees-------------------------
## Entropy
-0.60 * log2(0.60) - 0.40 * log2(0.40)

##Entropy curve for all possible values of x

curve(-x * log2(x) - (1 - x) * log2(1 - x),
      col="red", xlab 
      = "x", ylab = "Entropy", lwd=4)

credit <- read.csv("D:/Way to go as DS/ML/Data/german_credit_data_dataset.csv", header = T)
str(credit)
credit <- credit[c(-18:-21)]
str(credit)
table(credit$checking_balance)
table(credit$savings_balance)
summary(credit$months_loan_duration)
summary(credit$amount)

## as it is integer, we should convert into a factor as we are interested 
## to classify it
class(credit$default)
credit$default <- as.factor(credit$default)
table(credit$default)
## since we are interested to classify the default class, we find the 
## table values for default to check the split also.

set.seed(12345)
credit_rand <- credit[order(runif(1000)),]
head(credit_rand$amount)
credit_train <- credit_rand[1:900,]
credit_test <- credit_rand[901:1000,]

prop.table(table(credit_train$default))
prop.table(table(credit_test$default))

## this apperas to be fairly equal split by comparing the table value
## of default

## 1) Training the model on the data

install.packages("C50")
library(C50)
credit_model <- C5.0(x=credit_train[-17], y=credit_train$default)
credit_model
summary(credit_model)

## 2) Evaluating model performance

credit_pred <- predict(credit_model, credit_test)

## comparing Predicted class values with the actual class values

library(gmodels)
CrossTable(credit_test$default, credit_pred)
CrossTable(credit_test$default, credit_pred,
           prop.chisq = F, prop.r = F, prop.c = F,
           dnn = c("Actual Default", "Predicted Default" ))
## boosting by addidng the trials
## here 10 trials, indicates the no of separate decision trees to use in 
## forming the boosted team

## 3) Improving model performance

credit_boost10 <- C5.0(credit_train[-17], credit_train$default, trials = 10)
credit_boost10
credit_model
## 80 trees reduced to 61.4 trees
# summary(a)   (b)    <-classified as
# 631     1    (a): class 1
#  18   250    (b): class 2
summary(credit_model) 
# (a)   (b)    <-classified as
# 605    27    (a): class 1
#  87   181    (b): class 2

## now the test error is reduced from 12.7% to 2.1%
#(19/900)*100 = 2.1
#((87+27)/900)*100 = 12.7

credit_boost10_pred <- predict(credit_boost10, credit_test)
CrossTable(credit_test$default, credit_boost10_pred, 
           prop.chisq = F, prop.r = F, prop.c = F,
           dnn = c("Actual default", "Predicted default"))

## making some mistakes more costly than others

cost_matrix <- matrix(c(0,1,4,0), nrow = 2)
credit_cost <- C5.0(credit_train[-17], credit_train$default,
                    costs = cost_matrix)
credit_cost
credit_cost_pred <- predict(credit_cost, credit_test)
CrossTable(credit_test$default, credit_cost_pred,
           prop.chisq = F, prop.c = F, prop.r = F, prop.t = F,
           dnn = c("actual", "predicted"))
## here, 67% accuracy with 33% error
## and 4/4+28 = 12.5% of defaults are wrongly predicted as defaults

save.image()

