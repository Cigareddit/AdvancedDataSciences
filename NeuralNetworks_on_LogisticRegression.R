
# Multimonimal logistic regression on Titanic Data set

install.packages("Amelia")
install.packages("ROCR")
install.packages("boot")
library(XLConnect)
library(boot)
library(Amelia)
data_frame_out = read.csv("/Users/sriman/Desktop/data sets/Titanic data set/train.csv",header=TRUE,na.strings = c(""))
data_frame_out

# getting the number of null values for each column, Sapply applies the function to each column of data_frame_out
sapply(data_frame_out,function(x) sum(is.na(x)))

# getting the length of unique values of each column
sapply(data_frame_out,function(x) length(unique(x)))

# to see the plot of missing values and observed values(mismap in amelia's package)
missmap(data_frame_out, main = "Missing values vs observed")


# The variable cabin has too many missing values, we will not use it. We will also drop PassengerId since it is only an index and Ticket.
# Eliminating Name column
# Using the subset() function we subset the original dataset selecting the relevant columns only.
data_frame_out <- subset(data_frame_out,select=c(2,3,5,6,7,8,10,12))

# since Age and Embarked have null values , we have to fill null values.
# all the null values are assigned to the mean of the column without considering null values
data_frame_out$Age[is.na(data_frame_out$Age)] <- mean(data_frame_out$Age,na.rm=T)

# since embarked is a categorical atribute and has null values, we assign the mode of the column
# R doesnot support mode function, so we write a function to calculate mode
calcmode = function(x)
{
  val = unique(x)
  val[which.max(tabulate(match(x, val)))]
  
}
data_frame_out$Embarked[is.na(data_frame_out$Embarked)] = calcmode(data_frame_out$Embarked);
sum(is.na(data_frame_out$Embarked))

# As far as categorical variables are concerned, using the read.table() or read.csv() by default will encode the categorical variables as factors. A factor is how R deals categorical variables.
# By default , contrasts function helps to see how R dummifies categorical columns
contrasts(data_frame_out$Sex)
is.factor(data_frame_out$Sex)
contrasts(data_frame_out$Embarked)
is.factor(data_frame_out$Embarked)

#----------#End of Data Preparation Phase


# Now Splitting the data 80% training and 20% test data and checking the Accuracy
set.seed(2)
train = sample(1:nrow(data_frame_out),nrow(data_frame_out)*4/5,replace = FALSE)
test =- train
data_frame_out_train=data_frame_out[train, ]
data_frame_out_test=data_frame_out[test, ]


#------------#Training the Model
model = glm(Survived~., family = binomial(link = "logit"),data = data_frame_out_train)
summary(model)

# Removing the insignificant columns Embarked, Parch, Sibsp
data_frame_out_train
data_frame_out_train = data_frame_out_train[ ,-c(8)]
data_frame_out_train = data_frame_out_train[ ,-c(6)]
data_frame_out_train = data_frame_out_train[ ,-c(6)]

# Predicting the model on test data

fitted.results <- predict(model,data_frame_out_test,type='response')
fitted.results
fitted.results <- ifelse(fitted.results > 0.5,1,0)
fitted.results

# Calculating the Miclassification Error
misClasificError <- mean(fitted.results != data_frame_out_test$Survived)
print(paste('Accuracy',1-misClasificError))

#  As a rule of thumb, a model with good predictive ability should have an AUC closer to 1 (1 is ideal) than to 0.5.
library(ROCR)
pr <- prediction(fitted.results, data_frame_out_test$Survived)
pr
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf)

# Area under Curve
# As a rule of thumb, a model with good predictive ability should have an AUC closer to 1 (1 is ideal) than to 0.5.
auc <- performance(pr, measure = "auc")
auc
auc <- auc@y.values[[1]]
auc

# Now performing Neural Networks on the reduced data
# input layer has 4 inputs and output layer has two outputs since it is a classification
# model.matrix converts the data_frame containing the factor attributes to dummy attributes
library(neuralnet)
n = names(data_frame_out_train)
m = model.matrix(~Survived + SibSp + Sex + Age + Pclass , data = data_frame_out_train)
nn = neuralnet(Survived ~ SibSp + Age + Sexmale + Pclass ,data = m, hidden = 3, linear.output = FALSE)
plot(nn)


# Removing the insignificant columns Embarked, Parch on test data
data_frame_out_test
data_frame_out_test = data_frame_out_test[ ,-c(8)]
data_frame_out_test = data_frame_out_test[ ,-c(6)]
data_frame_out_test = data_frame_out_test[ ,-c(6)]
data_frame_out_test
m1 = model.matrix(~Survived + Pclass + Sex + Age + SibSp, data_frame_out_test)

# Predicting for test data
help(compute)
#data_frame_out_test$Sex = factor(data_frame_out_test$Sex,labels = c(0,1))
m2 = model.matrix(~Survived + SibSp + Sex + Age + Pclass , data = data_frame_out_test)
pr_nn = compute(nn,m2)




