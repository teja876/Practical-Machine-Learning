library(caret)
library(rattle)
pmlTrain <- read.csv('pml-training.csv')
pmlTest <- read.csv('pml-testing.csv')


incolTrain <- which(colSums(is.na(pmlTrain)  | pmlTrain == "") > 
                      dim(pmlTrain)[1]*.9)
incolTest <- which(colSums(is.na(pmlTest)  | pmlTest == "") > 
                     dim(pmlTest)[1]*.9)

pmlTrain2 <- pmlTrain[, -incolTrain]
pmlTrain2 <- pmlTrain2[, -c(1:7)]
pmlTest2 <- pmlTest[, -incolTest]
pmlTest2 <- pmlTest2[, -c(1:7)]
dim(pmlTrain2)
dim(pmlTest2)

set.seed(12345)
intrain <- createDataPartition(pmlTrain2$classe , p=0.75, list = FALSE)
trainData <- pmlTrain2[intrain, ]
testData <- pmlTrain2[-intrain, ]


dim(trainData)
dim(testData)

modelTree <- train(classe~., data = trainData, method = "rpart",
                   trControl = trainControl(method = "cv", number = 5))

fancyRpartPlot(modelTree$finalModel)

prediction <- predict(modelTree, newdata = testData)
conMat <- confusionMatrix(testData$classe, prediction)
conMat$table
conMat$overall


modelRF <- train(classe~., data = trainData, method = "rf", 
                 trControl = trainControl(method = "cv", number = 5), 
                 verbose=FALSE)


prediction <- predict(modelRF, newdata = testData)
conMat <- confusionMatrix(testData$classe, prediction)
conMat$table
conMat$overall
plot(modelRF)



modelGbm <- train(classe~., data = trainData, method = "gbm",
                  trControl = trainControl(method = "cv", number = 5),
                  verbose=FALSE)

prediction <- predict(modelGbm, newdata = testData)
conMat <- confusionMatrix(testData$classe, prediction)
conMat$table
conMat$overall



prediction <- predict(modelRF, newdata = pmlTest2)
prediction

















