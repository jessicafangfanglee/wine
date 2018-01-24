# Classification problem with Wine dataset using support vector machine and Naive Bayes
# , random forest, k-nearest neighbor classifier

head(wine)
str(wine)
summary(wine)

# The wine dataset should be standardized since they are all in different measurement
wine.stdz <- wine
wine.stdz <- as.data.frame(scale(wine.stdz[,2:14]))
wine.stdz$Wine <- wine$Wine
head(wine.stdz)

plot(wine.stdz$Flavanoids, wine.stdz$Proline,col=(2:4)[wine$Wine])
plot(wine.stdz[['Total phenols']], wine.stdz[['OD280/OD315 of diluted wines']],col=(2:4)[wine$Wine])

cor(wine.stdz)



# examine the principal components
wine.pc <- princomp(wine.stdz)
wine.pc
summary(wine.pc)
wine.pc$loadings
pairs(princomp(wine.stdz)$scores[,1:4],col=(2:4)[wine$Wine])
plot(wine.pc)
as.factor(wine.stdz$Wine)

# plant a tree here to examine the variance and compare to princomp
library(rpart)
wine.tree <- rpart(Wine~., data=wine.stdz, method ='class')
summary(wine.tree)
wine.tree
# seems like flavanoids and proline are the most important variables
# so when plotting the svm, i will use these two dimensions 

# build an svm model using caret
# create partitions of training and testing
library(caret)
set.seed(3456)
trainIndex <- createDataPartition(wine.stdz$Wine, p = .7, 
                                  list = FALSE, 
                                  times = 1)
head(trainIndex)

wineTrain <- wine.stdz[trainIndex,]
wineTest  <- wine.stdz[-trainIndex,]

# examine the property of the training and testing data, as well as 
# the original scaled wine dataset
sapply(wineTrain, summary)
sapply(wineTest, summary)
sapply(wine.stdz, summary)

# train an svm model and use builtin cross validation
set.seed(2024)
wine.svm <- train(
  Wine ~ ., wineTrain,
  method = "svmRadialWeights",
  trControl = trainControl(
    method = "cv", number = 10,
    verboseIter = TRUE
  )
)

plot(wine.svm)

pred <- predict(wine.svm, wineTest)
tab <- confusionMatrix(pred, wineTest$Wine)
tab

data(iris)
m2 <- svm(Species~., data = iris)
plot(m2, iris, Petal.Width ~ Petal.Length,
     slice = list(Sepal.Width = 3, Sepal.Length = 4))

# the plot is not working so i'm creating a new dataframe with only three
# variables

wine_3v <- as.data.frame(wine.stdz[c("Wine", "Flavanoids","Proline")])
wine.svm.svm <- svm(Wine~., data=wine_3v)
plot(wine.svm.svm, wine.stdz, Flavanoids ~ Proline, plotType = 'scatter')

# train a naive bayes model with cross validation
set.seed(2028)
wine.nb <- train(
  Wine~., wineTrain,
  method = 'nb',
  trControl = trainControl(
    method = 'cv', number = 10, verboseIter = TRUE
  )
)

pred.nb <- predict(wine.nb, wineTest)
tab.nb <- confusionMatrix(pred.nb, wineTest$Wine)
tab.nb

# plot roc and auc curve to see how well the model has performed
install.packages("caTools")
library('caTools')
colAUC(wine.stdz$Flavanoids, wine$Wine, plotROC = TRUE)

# use a knn model to evaluate how well it has performed
set.seed(2019)
wine.knn <- train(
  Wine~., wineTrain,
  method = 'knn',
  trControl = trainControl(
    method = 'cv', number = 10, verboseIter = TRUE
  )
)

pred.knn <- predict(wine.knn, wineTest)
tab.knn <- confusionMatrix(pred.knn, wineTest$Wine)
tab.knn


# split the data into half and half
library(caret)
set.seed(3456)
trainIndex <- createDataPartition(wine.stdz$Wine, p = .5, 
                                  list = FALSE, 
                                  times = 1)
head(trainIndex)
wine_Train <- wine.stdz[ trainIndex,]
wine_Test  <- wine.stdz[-trainIndex,]
# fit the same models into the data
set.seed(2024)
wine.svm <- train(
  as.factor(Wine) ~ ., wine_Train,
  method = "svmRadialWeights"
)

plot(wine.svm)

pred <- predict(wine.svm, wine_Test)
tab <- confusionMatrix(pred, wine_Test$Wine)
tab

# train a random forrest model with builtin cross validation
set.seed(2020)
wine.rf <- train(
  as.factor(Wine)~., wine_Train,
  method = 'ranger')

wine.rf

pred.rf <- predict(wine.rf, wine_Test)
tab.rf <- confusionMatrix(pred.rf, wine_Test$Wine)
tab.rf

# train a naive bayes model with manual split
set.seed(2028)
wine.nb <- train(
  as.factor(Wine)~., wine_Train,
  method = 'nb'
)

pred.nb <- predict(wine.nb, wine_Test)
tab.nb <- confusionMatrix(pred.nb, wine_Test$Wine)
tab.nb

# train a knn model with manual split
set.seed(2019)
wine.knn <- train(
  as.factor(Wine)~., wine_Train,
  method = 'knn')

pred.knn <- predict(wine.nb, wine_Test)
tab.knn <- confusionMatrix(pred.nb, wine_Test$Wine)
tab.knn

# conclusion: svm performs the best, nb and knn performs the worst, random forrest
# did ok. Now, examine the 3-d plot of the three factors to see if the variables 
# can be separated perfectly
install.packages("pca3d")

library('pca3d')

cor(wine)
library('rgl')
plot3d(wine$Flavanoids, wine$Proline, wine$`Nonflavanoid phenols`, col=(2:4)[wine$Wine])
plot3d(wine$Flavanoids, wine$Proline, wine$`OD280/OD315 of diluted wines`, col=(2:4)[wine$Wine])