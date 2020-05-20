rm(list=ls())
library(tm)
library(e1071)
library(randomForest)

csvdata1 <- read.csv("complaint1700.csv", header=TRUE, sep=',')
csvdata2 <- read.csv("noncomplaint1700.csv", header=TRUE, sep=',')
csvdata1['complaint'] = 1
csvdata2['complaint'] = -1
csvdata = rbind(csvdata1,csvdata2)
y <- csvdata$complaint

docs <- Corpus(VectorSource(csvdata$tweet))
dtm.control = list(tolower=T, removePunctuation=T, removeNumbers=T, 
                   stripWhitespace=T, stemming=T)
dtm.full <- DocumentTermMatrix(docs, control=dtm.control)

dtm <- removeSparseTerms(dtm.full,0.99)
X <- as.matrix(dtm)
Y <- as.factor(y)

set.seed(1) # fixing the seed value for the random selection guarantees the same results in repeated runs
n=length(y)
n1=round(n*0.8)
n2=n-n1
train=sample(1:n,n1)

###########################################
#############   Evaluation   ##############
###########################################

Evaluation <- function(pred, true, class)
{
  
  tp <- sum( pred==class & true==class)
  fp <- sum( pred==class & true!=class)
  tn <- sum( pred!=class & true!=class)
  fn <- sum( pred!=class & true==class)
  precision <- tp/(tp+fp)
  recall <- tp/(tp+fn)
  F1 <- 2/(1/precision + 1/recall)
  F1
}

###########################################
##########   Naive Bayesion   #############
###########################################

nb.model <- naiveBayes(X[train,], Y[train], nfold = 10, laplace = 0)
pred <- predict(nb.model, X[-train,])
table(pred, Y[-train])
Evaluation(pred, Y[-train], 1)
Evaluation(pred, Y[-train], -1)


###########################################
#######   Support Vector Machine   ########
###########################################

svm.model1 <- svm(Y[train] ~ ., data = X[train,], kernel='linear', type = 'C', degree = 3)
pred1 <- predict(svm.model1, X[-train,])
table(pred1, Y[-train])
Evaluation(pred1, Y[-train], 1)
Evaluation(pred1, Y[-train], -1)

###########################################
#######   Support Vector Machine 2  ########
###########################################
svm.model2 <- svm(Y[train] ~ ., data = X[train,], kernel='linear', type = 'nu-classification', degree = 3)
pred2 <- predict(svm.model2, X[-train,])
table(pred2, Y[-train])
Evaluation(pred2, Y[-train], 1)
Evaluation(pred2, Y[-train], -1)

###########################################
#######   Support Vector Machine 3  ########
###########################################
svm.model3 <- svm(Y[train] ~ ., data = X[train,], kernel='polynomial', type = 'nu-classification', degree = 3)
pred3 <- predict(svm.model3, X[-train,])
table(pred3, Y[-train])
Evaluation(pred3, Y[-train], 1)
Evaluation(pred3, Y[-train], -1)

###########################################
#######   Random Forest  ########
###########################################
rf.model <- randomForest(X[train,], Y[train], ntree=800, sampsize=200, threshold=0.7)
predrf <- predict(rf.model, X[-train,])
table(predrf, Y[-train])
Evaluation(predrf, Y[-train], 1)
Evaluation(predrf, Y[-train], -1)


library(DBI)
library(RMySQL)

driver <- dbDriver("MySQL")
myhost <- "localhost"
mydb   <- "studb"
myacct <- "cis434"
mypwd  <- "LLhtFPbdwiJans8F@S207" 

conn <- dbConnect(driver, host=myhost, dbname=mydb, myacct, mypwd)

# hE'p0o41<Kur
temp <- dbGetQuery(conn, "SELECT * FROM proj4final WHERE tag=\"hE'p0o41<Kur\"")

dbDisconnect(conn)


temp_docs <- Corpus(VectorSource(temp$tweet))

dtm.control = list(tolower=T, removePunctuation=T, removeNumbers=T, 
                   stripWhitespace=T, stemming=T)
temp_dtm.full <- DocumentTermMatrix(temp_docs, control=dtm.control)
temp_dtm <- removeSparseTerms(temp_dtm.full,0.99)
temp_X <- as.matrix(temp_dtm)
temp_pred = predict(nb.model, temp_X)
output = temp[temp_pred==-1,c(1,5)]
write.csv(output,"output_final.csv", row.names = FALSE)

#calculate precision
evaluation_result <- read.csv("project.csv", header=TRUE, sep=',')
sum(evaluation_result$evaluation)/nrow(evaluation_result)


