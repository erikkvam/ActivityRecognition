
library(MASS)
library(nnet)
library(caret)
library(e1071)
set.seed (4567)

setLocalDir <- function() {
  actualDir <- dirname(rstudioapi::getSourceEditorContext()$path)
  setwd(actualDir)
}

setLocalDir()

activity_labels <- read.table(file = "UCI_HAR_Dataset/activity_labels.txt", header=FALSE)
features <- read.table(file = "UCI_HAR_Dataset/features.txt", header=FALSE)

#0) Construcció dels datasets
#Construcció del dataset de train
x_train <- read.table(file = "UCI_HAR_Dataset/train/X_train.txt", header=FALSE)
y_train <- read.table(file = "UCI_HAR_Dataset/train/y_train.txt", header=FALSE)
subjects <- read.table(file = "UCI_HAR_Dataset/train/subject_train.txt", header=FALSE)

colnames(x_train) <- features$V2 #donem noms a les columnes dels features
raw.train <- x_train

#Construcció del dataset de test
x_test <- read.table(file = "UCI_HAR_Dataset/test/X_test.txt", header=FALSE)
y_test <- read.table(file = "UCI_HAR_Dataset/test/y_test.txt", header=FALSE)
subjects <- read.table(file = "UCI_HAR_Dataset/test/subject_test.txt", header=FALSE)

colnames(x_test) <- features$V2 #definim quina 
raw.test <- x_test



#1) --Selecció de variables a utilitzar--

#Busquem les components principals de les nostres dades de train
pca.train <- prcomp(raw.train,center = T,retx = T) 

#predim les components principals de les nostres dades de test
pca.test <- predict(pca.train, newdata = raw.test)

#les potejem per veure com triar quants components principals 
#per explicat la major part de les nostres dades 
plot(pca.train, type = "l")

vars <- apply(pca.train$x, 2, var)  
props <- vars / sum(vars)
acumul <- cumsum(props)
plot(acumul[1:200], type = "l", xlab = "Nº Components Principals", ylab = "% Explicat", log = "x", panel.first = grid())
plot(acumul[1:200], type = "l", xlab = "Nº Components Principals", ylab = "% Explicat", panel.first = grid())
#podem observar que:
#amb 34 expliquem el 90% de les dades
#amb 67 expliquem el 95% de les dades
#amb 155 expliquem el 99% de les dades


#2) --Perprocessem les dades-- 
npcs <- 67
data.train <- pca.train$x[,1:npcs]
data.train <- as.data.frame(cbind(data.train, activity=y_train$V1)) #afegim la columa de l'activitat que esta realitzant
data.train$activity <- factor(data.train$activity, labels=c("WALKING","WALKING_UPSTAIRS","WALKING_DOWNSTAIRS","SITTING","STANDING","LAYING"))

data.test <- pca.test[,1:npcs]
data.test <- as.data.frame(cbind(data.test, activity=y_test$V1)) #afegim la columa de l'activitat que esta realitzant
data.test[,npcs+1] <- factor(data.test$activity, labels=c("WALKING","WALKING_UPSTAIRS","WALKING_DOWNSTAIRS","SITTING","STANDING","LAYING"))

#3) --Metodes lineals--



#3.3) SVM Lineal
(C <- 2^(3:5))
model.CV.SVMLIN <- tune(svm, activity ~., data=data.train, kernel="linear", ranges=list(cost=C))

####################################
save(model.CV.SVMLIN, file = "CV-SVMLIN.mod")
load ("CV-SVMLIN.mod")
####################################

(model.CV.SVMLIN)

(model.svmLine <- svm (activity ~., data=data.train, type="C-classification", cost=16, kernel="linear", scale = FALSE))
summary(model.svmLine)

p1 <- as.factor(predict (model.svmLine, type="class"))
(encerts <- sum(p1 == data.train$activity))
(error_rate.learn <- 100*(1-(encerts)/nrow(data.train)))

p2 <- as.factor(predict (model.svmLine, newdata=data.test, type="class"))
(encerts <- sum(p2 == data.test$activity))
(error_rate.valid <- 100*(1-(encerts)/nrow(data.test)))

#3.4) SVM Quadratic
(C <- 2^(4:6))
model.CV.SVMQUAD <- tune(svm, activity ~., data=data.train, kernel="polynomial", degree=2, coef0=1, ranges=list(cost=C))

####################################
save(model.CV.SVMQUAD, file = "CV-SVMQUAD.mod")
load ("CV-SVMQUAD.mod")
####################################

(model.CV.SVMQUAD)

(model.svmQuad <- svm(activity ~., data=data.train, type="C-classification", cost=16, kernel="polynomial", degree=2, coef0=1, scale = FALSE))
summary(model.svmQuad)

p1 <- as.factor(predict (model.svmQuad, type="class"))
(encerts <- sum(p1 == data.train$activity))
(error_rate.learn <- 100*(1-(encerts)/nrow(data.train)))

p2 <- as.factor(predict (model.svmQuad, newdata=data.test, type="class"))
(encerts <- sum(p2 == data.test$activity))
(error_rate.valid <- 100*(1-(encerts)/nrow(data.test)))


#4) --Metodes no lieals--
#4.1) MLP
#Primer intentem buscar un nombre de neurones ocultes que sobreajustin. 
HiddenUnits <- 12

model.nnet <- nnet(activity ~., data = data.train, size=HiddenUnits, maxit=500, MaxNWts = 1200)

p1 <- as.factor(predict (model.nnet, type="class"))
(encerts <- sum(p1 == data.train$activity))
(error_rate.learn <- 100*(1-(encerts)/nrow(data.train)))

p2 <- as.factor(predict (model.nnet, newdata=data.test, type="class"))

(encerts <- sum(p2 == data.test$activity))
(error_rate.valid <- 100*(1-(encerts)/nrow(data.test)))


#es pot observar que amb 12 neurones ocultes, el model esta sobreajustant, ja que el error de train es 0, 
#i el error de test augmenta si el comparem amb models on hi ha menys neurones ocultes.

#Ara, a traves de cross validation, buscarem quin es el millor valor de decay, per tal de regularitzar el model.
(decays <- 10^seq(-0.4,-0.3,by=0.025))

trc <- trainControl(method="repeatedcv", number=10, repeats=10)

model.10x10CV <- train(activity ~., data=data.train, method='nnet', maxit = 500, trace = FALSE, 
                        tuneGrid = expand.grid(.size=12,.decay=decays),trainControl = trc)

####################################
save(model.10x10CV, file = "10x10CV-MLP.mod")
load ("10x10CV-MLP.mod")
####################################

#Mirem quin es el valor de regularitzacio que millor ajusta el nostre model 
model.10x10CV$bestTune
  
#El model final de MLP es el seguent:
model.nnet <- nnet(activity ~., data = data.train, size=HiddenUnits, maxit=700, decay = 0.4466836)

p1 <- as.factor(predict (model.nnet, type="class"))
(encerts <- sum(p1 == data.train$activity))
(error_rate.learn <- 100*(1-(encerts)/nrow(data.train)))

p2 <- as.factor(predict (model.nnet, newdata=data.test, type="class"))

(encerts <- sum(p2 == data.test$activity))
(error_rate.valid <- 100*(1-(encerts)/nrow(data.test)))



#4.2) SVM amb el kernel RBF

#Fem una primera aproximació sense tocar els hiper-parametres
(model.svm <- svm(activity ~., data=data.train, type="C-classification", cost=1, kernel="radial", scale = FALSE))

p1 <- as.factor(predict (model.svm, type="class"))
(encerts <- sum(p1 == data.train$activity))
(error_rate.learn <- 100*(1-(encerts)/nrow(data.train)))

p2 <- as.factor(predict (model.svm, newdata=data.test, type="class"))
(encerts <- sum(p2 == data.test$activity))
(error_rate.valid <- 100*(1-(encerts)/nrow(data.test)))

(G <- 2^(-7:-5))
(C <- 2^(2:4))
#Un cop vist que els resultats son bons, pero poden ser millors si apliquem CV al model per trobar els hiper-parametres
(model.CV.SVMRBF <- tune(svm, activity ~ ., data = data.train, ranges = list(gamma = G, cost = C),
            tunecontrol = tune.control(sampling = "fix")))
####################################
save(model.CV.SVMRBF, file = "CV-SVMRBF.mod")
load ("CV-SVMRBF.mod")
####################################

model.CV.SVMRBF
#Tal com es pot veure ens els resultat del CV, els millors valors per els hiper-parametres gamma i cost són,  0.015625 i 8 respectivament.

(model.svm <- svm(activity ~., data=data.train, type="C-classification", cost=8, gamma = 0.015625, kernel="radial", scale = FALSE))

p1 <- as.factor(predict (model.svm, type="class"))
(encerts <- sum(p1 == data.train$activity))
(error_rate.learn <- 100*(1-(encerts)/nrow(data.train)))

p2 <- as.factor(predict (model.svm, newdata=data.test, type="class"))
(encerts <- sum(p2 == data.test$activity))
(error_rate.valid <- 100*(1-(encerts)/nrow(data.test)))
