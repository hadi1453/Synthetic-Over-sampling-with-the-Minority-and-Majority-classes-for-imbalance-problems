library(ggplot2)
library(sp)
library(pROC)
library(e1071)
library(rpart)
library(randomForest)
library(smotefamily)
library(mvtnorm)
library(splitstackshape)
library(MASS)
library(expm)
library(FNN)
library(raster)
library(PRROC)
library(dplyr)
library(nnet)
library(mlbench)
library(fastDummies)

memory.limit(size = 56000)
options(scipen=10000)

Path_Imb_Div <- "C:/Users/hkhorshidi/Documents/CIS Unimelb/Research topics/Imbalanced data/Oversampling SWIM SMOTE/Data sets"

# Poker
poker <- read.csv(file.path(Path_Imb_Div,"poker-hand-testing.csv"))
table(poker$class)
poker1 <- poker[poker$class>=6,]
poker1$class[poker1$class==6] <- 0
poker1$class[poker1$class>=7] <- 1
table(poker1$class)

# Poker Multi-class, 0/1/2/3/4/5/6/7,8,9
poker1 <- poker[sample(nrow(poker), 5000),]
pokerMC <- poker1
pokerMC$class[pokerMC$class>=5] <- 5
table(pokerMC$class)/nrow(pokerMC)

# Segment
Segment <- read.csv(file.path(Path_Imb_Div,"segmentation.csv"))
table(Segment$class)

# sonar
data("Sonar")
names(Sonar)[61] <- "class"
levels(Sonar$class) <- c(0,1)
table(Sonar$class)

# Coil2000
Coil <- read.csv(file.path(Path_Imb_Div,"ticdata2000.csv"))
table(Coil$class)
Coil$class <- as.factor(Coil$class)
Coil1 <- Coil[sample(nrow(Coil), 1500),]
table(Coil1$class)

# Diabetes
Diabetes <- read.csv(file.path(Path_Imb_Div,"diabetes.csv"))
levels(Diabetes$class) <- c(0,1)

# Oil-spill
Oil <- read.csv(file.path(Path_Imb_Div,"oil-spill.csv"))
Oil$class <- as.factor(Oil$class)
Oil <- Oil[,-c(23,33)]

# Hepatitis
Hepatitis <- read.csv(file.path(Path_Imb_Div,"hepatitis_update.csv"), row.names = 1)
Hepatitis1 <- cbind.data.frame(Hepatitis[,-1], class=Hepatitis[,1])

# Breast Cancer Wisconsin
breast <- read.csv(file.path(Path_Imb_Div,"breast-cancer-wisconsin.csv"))
breast$class[breast$class==4] <- 1
breast$class[breast$class==2] <- 0
breast$class <- as.factor(breast$class)

# Waveform
Waveform <- read.csv(file.path(Path_Imb_Div,"waveform.csv"))
Waveform$class[Waveform$class==0] <- 3
Waveform$class[Waveform$class==1] <- 0
Waveform$class[Waveform$class==2] <- 0
Waveform$class[Waveform$class==3] <- 1
Waveform$class <- as.factor(Waveform$class)
Waveform1 <- Waveform[sample(nrow(Waveform), 1800),]

# Vowel 
data(Vowel)
table(Vowel$Class)
Vowel$V1 <- as.numeric(Vowel$V1)

# Vowel0
Vowel0 <- Vowel[,-11]
Vowel0$class[Vowel$Class!="hid"] <- 0
Vowel0$class[Vowel$Class=="hid"] <- 1
Vowel0$class <- as.factor(Vowel0$class)
levels(Vowel0$class) 
table(Vowel0$class)
Vowel0$V1 <- as.numeric(Vowel0$V1)

# Vowel1
Vowel1 <- Vowel[,-11]
Vowel1$class[Vowel$Class!="hId"] <- 0
Vowel1$class[Vowel$Class=="hId"] <- 1
Vowel1$class <- as.factor(Vowel1$class)
levels(Vowel1$class) 
table(Vowel1$class)

# Vowel2
Vowel2 <- Vowel[,-11]
Vowel2$class[Vowel$Class!="hEd"] <- 0
Vowel2$class[Vowel$Class=="hEd"] <- 1
Vowel2$class <- as.factor(Vowel2$class)
levels(Vowel2$class) 
table(Vowel2$class)

# Vowel10
Vowel10 <- Vowel[,-11]
Vowel10$class[Vowel$Class!="hed"] <- 0
Vowel10$class[Vowel$Class=="hed"] <- 1
Vowel10$class <- as.factor(Vowel10$class)
levels(Vowel10$class) 
table(Vowel10$class)

# Vowel Multi-class, 3 classes
VowelMC <- Vowel[,-11]
VowelMC$class[Vowel$Class!="hed"] <- 0
VowelMC$class[Vowel$Class=="hid"] <- 1
VowelMC$class[Vowel$Class=="hed"] <- 2
VowelMC$class <- as.factor(VowelMC$class)
#levels(VowelMC$class) 
table(VowelMC$class)/nrow(VowelMC)

# Vowel Multi-class, 5 classes
VowelMC5 <- Vowel[,-11]
VowelMC5$class[Vowel$Class!="hed"] <- 0
VowelMC5$class[Vowel$Class=="hid"] <- 1
VowelMC5$class[Vowel$Class=="hId"] <- 2
VowelMC5$class[Vowel$Class=="hud"] <- 3
VowelMC5$class[Vowel$Class=="hed"] <- 4
VowelMC5$class <- as.factor(VowelMC5$class)
#levels(VowelMC$class) 
table(VowelMC5$class)/nrow(VowelMC5)

# KDD Synthetic Control
KDD <- read.csv(file.path(Path_Imb_Div,"synthetic_control.csv"))
KDD$class <- as.factor(KDD$class)

# Skin of Orange
Orange <- read.csv(file.path(Path_Imb_Div,"orange10.csv"))
Orange$class <- as.factor(Orange$class)
levels(Orange$class) <- c(0,1)

# South African Heart Disease
SAHD <- read.csv(file.path(Path_Imb_Div,"South_African_Heart_Disease.csv"))
SAHD$famhist <- as.numeric(SAHD$famhist)
SAHD$famhist[SAHD$famhist==1] <- 0
SAHD$famhist[SAHD$famhist==2] <- 1
SAHD$class <- as.factor(SAHD$class)

# Forest cover
ForestCov <- read.csv(file.path(Path_Imb_Div,"covtype.csv"))
ForestCov$class <- as.factor(ForestCov$Cover_Type)
ForestCov <- ForestCov[,-55]
set.seed(1)
ForestCov1 <- ForestCov[sample(nrow(ForestCov), 3000),]

# Forest Cover 3
ForestCov3 <- ForestCov1[,-55]
ForestCov3$class[ForestCov1$class!=3] <- 0
ForestCov3$class[ForestCov1$class==3] <- 1
ForestCov3$class <- as.factor(ForestCov3$class)
levels(ForestCov3$class) 
table(ForestCov3$class)

# Forest Cover Multi-class
table(ForestCov1$class)
ForestCovMC <- ForestCov1[,-55]
ForestCovMC$class[ForestCov1$class==2] <- 0
ForestCovMC$class[ForestCov1$class==1] <- 1
ForestCovMC$class[ForestCov1$class==3] <- 2
ForestCovMC$class[ForestCov1$class==4] <- 3
ForestCovMC$class[ForestCov1$class==5] <- 3
ForestCovMC$class[ForestCov1$class==6] <- 4
ForestCovMC$class[ForestCov1$class==7] <- 5
table(ForestCovMC$class)/nrow(ForestCovMC)

# Vehicle
Vehicle <- read.csv(file.path(Path_Imb_Div,"Vehicle.csv"))

# Vehicle Bus
VehicleB <- Vehicle[,-19]
VehicleB$class[Vehicle$class!="bus"] <- 0
VehicleB$class[Vehicle$class=="bus"] <- 1
VehicleB$class <- as.factor(VehicleB$class)
table(VehicleB$class)/nrow(VehicleB)

# Vehicle Van
VehicleV <- Vehicle[,-19]
VehicleV$class[Vehicle$class!="van"] <- 0
VehicleV$class[Vehicle$class=="van"] <- 1
VehicleV$class <- as.factor(VehicleV$class)
table(VehicleV$class)/nrow(VehicleV)

# Vehicle Saab
VehicleS <- Vehicle[,-19]
VehicleS$class[Vehicle$class!="saab"] <- 0
VehicleS$class[Vehicle$class=="saab"] <- 1
VehicleS$class <- as.factor(VehicleS$class)
table(VehicleS$class)/nrow(VehicleS)

# Vehicle Opel
VehicleO <- Vehicle[,-19]
VehicleO$class[Vehicle$class!="opel"] <- 0
VehicleO$class[Vehicle$class=="opel"] <- 1
VehicleO$class <- as.factor(VehicleO$class)
table(VehicleO$class)/nrow(VehicleO)

# Vehicle Multi-class, Opel, Saab =0, Bus=1, Van=2
VehicleMC <- Vehicle[,-19]
VehicleMC$class[Vehicle$class=="opel" | Vehicle$class=="saab"] <- 0
VehicleMC$class[Vehicle$class=="bus"] <- 1
VehicleMC$class[Vehicle$class=="van"] <- 2
VehicleMC$class <- as.factor(VehicleMC$class)
table(VehicleMC$class)/nrow(VehicleMC)

# Pima
Pima <- read.csv(file.path(Path_Imb_Div,"pima-indians-diabetes.csv"))
names(Pima)[9] <- "class"
Pima$class <- as.factor(Pima$class)

# Wine 
Wine <- read.csv(file.path(Path_Imb_Div,"wine.csv"))
table(Wine$type)

# White Wine
WineW <- Wine[Wine$type=="W",-13]
table(WineW$quality)

# High Quality White Wine (7,8,9)
WineWH <- WineW[,-12]
WineWH$class[WineW$quality<=6] <- 0
WineWH$class[WineW$quality>=7] <- 1
WineWH$class <- as.factor(WineWH$class)
table(WineWH$class)

# Low Quality White Wine (3,4)
WineWL <- WineW[,-12]
WineWL$class[WineW$quality>=5] <- 0
WineWL$class[WineW$quality<=4] <- 1
WineWL$class <- as.factor(WineWL$class)
table(WineWL$class)

# White Wine Quality Low vs High
WineWLvH <- WineW[WineW$quality<5 | WineW$quality>6,]
WineWLvH$class[WineWLvH$quality>=7] <- 0
WineWLvH$class[WineWLvH$quality<=4] <- 1
WineWLvH$class <- as.factor(WineWLvH$class)
WineWLvH <- WineWLvH[,-12]
table(WineWLvH$class)

# White Wine Quality 3 vs 7
WineW3v7 <- WineW[WineW$quality==3 | WineW$quality==7,]
WineW3v7$class[WineW3v7$quality==7] <- 0
WineW3v7$class[WineW3v7$quality==3] <- 1
WineW3v7$class <- as.factor(WineW3v7$class)
WineW3v7 <- WineW3v7[,-12]
table(WineW3v7$class)

# Red Wine
WineR <- Wine[Wine$type=="R",-13]
table(WineR$quality)

# High Quality Red Wine (7,8)
WineRH <- WineR[,-12]
WineRH$class[WineR$quality<=6] <- 0
WineRH$class[WineR$quality>=7] <- 1
WineRH$class <- as.factor(WineRH$class)
table(WineRH$class)

# Low Quality Red Wine (3,4)
WineRL <- WineR[,-12]
WineRL$class[WineR$quality<=4] <- 1
WineRL$class[WineR$quality>=5] <- 0
WineRL$class <- as.factor(WineRL$class)
table(WineRL$class)

# White Wine Quality Low vs High
WineRLvH <- WineR[WineR$quality<5 | WineR$quality>6,]
WineRLvH$class[WineRLvH$quality>=7] <- 0
WineRLvH$class[WineRLvH$quality<=4] <- 1
WineRLvH$class <- as.factor(WineRLvH$class)
WineRLvH <- WineRLvH[,-12]
table(WineRLvH$class)

# Wine Multi-class, quality: 3,4 --> Low (0), 8,9 --> High (2), others --> Medium (1)
WineMC <- Wine[,-c(12,13)]
table(Wine$quality)
WineMC$class[Wine$quality<=4] <- 1
WineMC$class[Wine$quality>4] <- 0
WineMC$class[Wine$quality>=7] <- 2
table(WineMC$class)/nrow(WineMC)

# Wine Multi-class, quality: 3,4/5/6/7/8,9
WineMC1 <- Wine[,-c(12,13)]
WineMC1$class[Wine$quality<=4] <- 1
WineMC1$class[Wine$quality==5] <- 2
WineMC1$class[Wine$quality==6] <- 0
WineMC1$class[Wine$quality==7] <- 3
WineMC1$class[Wine$quality>=8] <- 4
WineMC1$class <- as.factor(WineMC1$class)
table(WineMC1$class)/nrow(WineMC)

# Wine Red Multi-class, quality: 3,4/5/6/7,8
WineRMC <- WineR[,-12]
WineRMC$class[WineR$quality<=4] <- 1
WineRMC$class[WineR$quality==5] <- 0
WineRMC$class[WineR$quality==6] <- 2
WineRMC$class[WineR$quality>=7] <- 3
WineRMC$class <- as.factor(WineRMC$class)
table(WineRMC$class)/nrow(WineRMC)

# Wine White Multi-class, quality: 3,4/5/6/7/8,9
WineWMC <- WineW[,-12]
WineWMC$class[WineW$quality<=4] <- 1
WineWMC$class[WineW$quality==5] <- 2
WineWMC$class[WineW$quality==6] <- 0
WineWMC$class[WineW$quality==7] <- 3
WineWMC$class[WineW$quality>=8] <- 4
WineWMC$class <- as.factor(WineWMC$class)
table(WineWMC$class)/nrow(WineWMC)


# Thoraric Surgery
Surgery <- read.csv(file.path(Path_Imb_Div,"ThoraricSurgery.csv"))
Surgery1 <- dummy_cols(Surgery, select_columns = names(Surgery)[-c(2,3,16)], remove_first_dummy = T, remove_selected_columns = T)
Surgery1$class <- as.factor(Surgery1$Risk1Yr_TRUE)
Surgery2 <- Surgery1[,-25]

# Ring Norm
Ring <- read.csv(file.path(Path_Imb_Div,"ring.csv"))
Ring$class <- as.factor(Ring$class)
tr_red <- sample(row.names(Ring[Ring$class==1,]), nrow(Ring[Ring$class==1,])/2)
Ring_red <- Ring[!(row.names(Ring) %in% tr_red),]

# Spam
Spam <- read.csv(file.path(Path_Imb_Div,"Spam.csv"))
Spam$class <- as.factor(Spam$class)

# Abalone
Abalone <- read.csv(file.path(Path_Imb_Div,"abalone.csv"))
Abalone1 <- dummy_cols(Abalone, select_columns = names(Abalone)[1], remove_first_dummy = T, remove_selected_columns = T)
Abalone2 <- Abalone1[Abalone1$X9>=9 & Abalone1$X9<=18,]
Abalone2$class[Abalone2$X9<=11] <- 0
Abalone2$class[Abalone2$X9>=12] <- 1
Abalone2 <- Abalone2[,-8]
Abalone2$class <- as.factor(Abalone2$class)
table(Abalone2$class)/nrow(Abalone2)

# Abalone Multiclass
AbaloneMC <- Abalone1[Abalone1$X9>=9 & Abalone1$X9<=18,]
AbaloneMC$class <- 2
AbaloneMC$class[AbaloneMC$X9<=15] <- 1
AbaloneMC$class[AbaloneMC$X9<=13] <- 0
table(AbaloneMC$class)/nrow(AbaloneMC)
AbaloneMC <- AbaloneMC[,-8]


##############################################################################
### Synthetic data generation
## Majority convex hull Normal
set.seed(0)
sigma <- matrix(c(11,4,4,2), ncol=2)
nsamples <- 100
x <- as.data.frame(rmvnorm(n=round(0.95*nsamples,0), mean=c(0,0), sigma=sigma))
x[,"class"] <- rep(0, round(0.95*nsamples,0))
names(x) <- c("x", "y", "class")
b_poly1 = Polygon(list(rbind(c(-5, 0), c(-2.5, 1), c(-5, 1), c(-5, 0))))
bp1 = SpatialPolygons(list(Polygons(list(Polygon(b_poly1)), "x")))
b_poly2 = Polygon(list(rbind(c(0, -2), c(0, -1.5), c(-2, -2), c(0, -2))))
bp2 = SpatialPolygons(list(Polygons(list(Polygon(b_poly2)), "x")))
set.seed(0)
df_b1 <- as.data.frame(spsample(bp1, n = 3, "random"))
df_b2 <- as.data.frame(spsample(bp2, n = 2, "random"))
df_b <- rbind(df_b1, df_b2)
df_b$class <- 1
df <- rbind(x, df_b)
df$class <- as.factor(df$class)
ggplot(data=df, aes(x=x, y=y, colour=class)) + geom_point() + labs(x="X", y="Y") + geom_density_2d(data = df[df$class==0,])

## Majority convex hull non-Normal 
a_poly = Polygon(list(rbind(c(0, 0.25), c(0.1, 0.23), c(0.2, 0.205), c(0.35, 0.19), c(0.5, 0.2), c(0.65, 0.25), c(0.6, 0.2), c(0.5, 0.15), c(0.4, 0.1), c(0.3, 0.1), c(0.25, 0.12), c(0.1, 0.15), c(0, 0.25))))
ap = SpatialPolygons(list(Polygons(list(Polygon(a_poly)), "x")))
plot(ap)
set.seed(0)
df <- as.data.frame(spsample(ap, n = 95, "random"))
df$class <- 0
b_poly1 = Polygon(list(rbind(c(0.15, 0.2), c(0.3, 0.23), c(0.4, 0.2), c(0.15, 0.2))))
bp1 = SpatialPolygons(list(Polygons(list(Polygon(b_poly1)), "x")))
b_poly2 = Polygon(list(rbind(c(0.1, 0.1), c(0.4, 0.1), c(0.1, 0.05), c(0.1, 0.1))))
bp2 = SpatialPolygons(list(Polygons(list(Polygon(b_poly2)), "x")))
set.seed(0)
df_b1 <- as.data.frame(spsample(bp1, n = 3, "random"))
df_b2 <- as.data.frame(spsample(bp2, n = 2, "random"))
df_b <- rbind(df_b1, df_b2)
df_b$class <- 1
df <- rbind(df, df_b)
df$class <- as.factor(df$class)
ggplot(data=df, aes(x=x, y=y, colour=class)) + geom_point(size=2) + labs(x="X", y="Y") + geom_density_2d(data = df[df$class==0,]) + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), panel.background = element_blank()) + scale_color_manual(breaks = c("0", "1"), values=c("red", "blue"))

## Minorinty instances inside majority space, Similar means and different covariances  
set.seed(0)
sigma <- matrix(c(8,0,0,8), ncol=2)
nsamples <- 100
x <- as.data.frame(rmvnorm(n=round(0.95*nsamples,0), mean=c(0,0), sigma=sigma))
x[,"class"] <- rep(0, round(0.95*nsamples,0))
sigma <- matrix(c(0.25,0,0,0.25), ncol=2)
y <- as.data.frame(rmvnorm(n=round(0.05*nsamples,0), mean=c(0,0), sigma=sigma))
y[,"class"] <- rep(1, round(0.05*nsamples,0))
df <- rbind(x, y)
names(df) <- c("x", "y", "class")
df$class <- as.factor(df$class)
ggplot(data=df, aes(x=x, y=y, colour=class)) + geom_point() + labs(x="X", y="Y") + geom_density_2d(data = df[df$class==0,])

## Majority non-convex hull 
set.seed(0)
sigma <- matrix(c(3.5,0,0,3.5), ncol=2)
nsamples <- 100
x <- as.data.frame(rmvnorm(n=round(0.475*nsamples,0), mean=c(4,0), sigma=sigma))
y <- as.data.frame(rmvnorm(n=round(0.475*nsamples,0), mean=c(-4,0), sigma=sigma))
df_a <- rbind(x, y)
df_a$class <- 0
names(df_a) <- c("x", "y", "class")
b_poly1 = Polygon(list(rbind(c(0, 0.5), c(0.5, 2.5), c(-1.5, 2.5), c(0, 0.5))))
bp1 = SpatialPolygons(list(Polygons(list(Polygon(b_poly1)), "x")))
b_poly2 = Polygon(list(rbind(c(-1, -1.25), c(1, -1.25), c(0, -1), c(-1, -1.25))))
bp2 = SpatialPolygons(list(Polygons(list(Polygon(b_poly2)), "x")))
set.seed(0)
df_b1 <- as.data.frame(spsample(bp1, n = 3, "random"))
df_b2 <- as.data.frame(spsample(bp2, n = 2, "random"))
df_b <- rbind(df_b1, df_b2)
df_b$class <- 1
df <- rbind(df_a, df_b)
df$class <- as.factor(df$class)
ggplot(data=df, aes(x=x, y=y, colour=class)) + geom_point(size=2) + labs(x="X", y="Y") + geom_density_2d(data = df[df$class==0,]) + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), panel.background = element_blank()) + scale_color_manual(breaks = c("0", "1"), values=c("red", "blue"))

##############################################################################################

## Auxiliary function for Proposed SOMM
SOMM <- function(df, n, K, a){
  # Step 1
  #mu_df <- apply(df[,names(df)!="class"], 2, mean)
  #sd_df <- apply(df[,names(df)!="class"], 2, sd)
  max_df <- apply(df[,names(df)!="class"], 2, max)
  min_df <- apply(df[,names(df)!="class"], 2, min)
  df_n <- df
  #df_n[,names(df)!="class"] <- apply(df[,names(df)!="class"], 2, FUN = function(x)((x-mean(x))/sd(x)))
  df_n[,names(df)!="class"] <- apply(df[,names(df)!="class"], 2, FUN = function(x)(if((max(x)-min(x))!=0){(x-min(x))/(max(x)-min(x))}else{x-min(x)}))
  
  A <- df_n[df_n$class!=a,names(df_n)!="class"]
  B <- df_n[df_n$class==a,names(df_n)!="class"]
  # Step 2
  U <- apply(B, 2, max)
  L <- apply(B, 2, min)
  # Step 3
  S <- data.frame(matrix(NA, 0, ncol = ncol(B)))
  names(S) <- names(B)
  Snew <- data.frame(matrix(NA, 0, ncol = ncol(B)))
  names(Snew) <- names(B)
  i <- 1
  while (i <= n) {
    for (j in 1:ncol(B)) {
      S[i, j] <- runif(1, L[j], U[j])
    }
    nn <- get.knnx(df_n[,names(df_n)!="class"], S[i,], k=K)
    nn_df <- df_n[nn$nn.index,]
    #nn_df$dis <- t(nn$nn.dist)
    #nn_dfA <- nn_df[nn_df$class==0,]
    #nn_dfB <- nn_df[nn_df$class==1,]
    if(nrow(nn_df[nn_df$class==a,])==0){i <- i-1}else if(nn_df$class[1]==a){Snew[i,] <- S[i,]*(max_df - min_df) + min_df}else{
      B_index <- which(nn_df$class==a)[1]
      nn_A <- nn_df[(1:B_index-1),names(df)!="class"]
      nn_B <- nn_df[B_index,names(df)!="class"]
      #mean_nn_A <- colMeans(nn_A)
      disA <- apply(nn_A, 1, FUN = function(x) (dist(rbind(S[i,], x))))
      disB <- dist(rbind(S[i,], nn_B))
      dirB <- nn_B - S[i,]
      disAonB <- disA
      for (k in 1:length(disA)) {
        dirA <- nn_A[k,] - S[i,]
        disAonB[k] <- disA[k]*sum(dirA*dirB)/(norm(dirA, type = "2")*norm(dirB, type = "2"))
      }
      MdisAonB <- max(max(disAonB),0)
      xB <- runif(1, MdisAonB, disB)
      Snew[i,] <- (S[i,]+dirB*xB)*(max_df - min_df) + min_df #  sd_df + mu_df
    }
    i <- i + 1
  }
  
  Snew$class <- a
  return(Snew)
}

############################################################################################################
############## Examples ####################################################################################
df <- df

# SMOTE
syn_smote <- SMOTE(df[,-3], df[,3], K=2, dup_size = 0)$syn_data
syn_smote$class <- 2
syn_smote$class <- as.factor(syn_smote$class)
df_SMOTE <- as.data.frame(rbind(df, syn_smote))
ggplot(data=df_SMOTE, aes(x=x, y=y)) + geom_point(aes(color=class,shape=class), size=3) + geom_density_2d(data = df[df$class==0,], colour="red") + theme(axis.title.x = element_blank(), axis.text.x = element_blank(), axis.ticks.x = element_blank(), axis.title.y = element_blank(), axis.text.y = element_blank(), axis.ticks.y = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), panel.background = element_blank(), legend.position = "top") + scale_color_manual(name = "", labels = c("Majority training", "Minority training", "Synthetic minority"), values=c("red", "blue", "blue")) + scale_shape_manual(name = "", labels = c("Majority training", "Minority training", "Synthetic minority"), values=c(16,16, 10))

# SOMM
Snew_SOMM <- SOMM(df, 90, 15, 1)
Snew_SOMM$class <- 2
Snew_SOMM$class <- as.factor(Snew_SOMM$class)
df_SOMM <- as.data.frame(rbind(df, Snew_SOMM))
ggplot(data=df_SOMM, aes(x=x, y=y)) + geom_point(aes(color=class,shape=class), size=3) + geom_density_2d(data = df[df$class==0,], colour="red") + theme(axis.title.x = element_blank(), axis.text.x = element_blank(), axis.ticks.x = element_blank(), axis.title.y = element_blank(), axis.text.y = element_blank(), axis.ticks.y = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), panel.background = element_blank(), legend.position = "top") + scale_color_manual(name = "", labels = c("Majority training", "Minority training", "Synthetic minority"), values=c("red", "blue", "blue")) + scale_shape_manual(name = "", labels = c("Majority training", "Minority training", "Synthetic minority"), values=c(16,16, 10))


##############################################################################################
### Visualisation
df <- Diabetes

tr <- sample(dim(df)[1], dim(df)[1]*0.75)
df_tr <- df[tr,]
df_te <- df[-tr,]
df_tr_red <- df_tr

df_tr1 <- df_tr_red
df_tr1$classU[df_tr1$class==1] <- "1Tr"
df_tr1$classU[df_tr1$class==0] <- "2Tr"

df_te1 <- df_te
df_te1$classU[df_te1$class==1] <- "1Ts"
df_te1$classU[df_te1$class==0] <- "2Ts"

df <- rbind.data.frame(df_tr1, df_te1)

#cl <- 11
cl <- c(9,10) 

df_n <- df
#df_n[,-cl] <- apply(df[,-cl], 1, FUN = function(x)((x-min(x))/(max(x)-min(x))))

p.comp <- prcomp(df_n[,-cl], scale. = T)
X.comp <- as.data.frame(p.comp$x[,1:2])
X_comp_Cl <- cbind.data.frame(X.comp, as.factor(df_n$classU))
names(X_comp_Cl) <- c("x", "y", "class")
#X_comp_Cl$class <- factor(X_comp_Cl$class, levels = c("MinTr", "MinTe", "MajTe", "MajTr"))
levels(X_comp_Cl$class)
ggplot(data=X_comp_Cl, aes(x=x, y=y)) + geom_point(data=X_comp_Cl[X_comp_Cl$class!="1Tr",], aes(color=class,shape=class), size=4) + theme(axis.title.x = element_blank(), axis.text.x = element_blank(), axis.ticks.x = element_blank(), axis.title.y = element_blank(), axis.text.y = element_blank(), axis.ticks.y = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), panel.background = element_blank(), legend.position = "top") + scale_color_manual(name = "", labels = c("Minority training", "Minority test", "Majority training", "Majority test"), values=c("blue", "cyan1", "red", "rosybrown1")) + scale_shape_manual(name = "", labels = c("Minority training", "Minority test", "Majority training", "Majority test"), values=c(16,16,16,16)) +
  geom_point(data=X_comp_Cl[X_comp_Cl$class=="1Tr",], aes(x,y,color=class, shape=class), size=5) 

OS <- -diff(table(df_tr_red$class))



# SOMM
Snew <- SOMM(df_tr_red, 100, 5, 1)
Snew$classU <- "G"
df_new <- rbind.data.frame(df_n, Snew)

p.comp <- prcomp(df_new[,-cl], scale. = T)
X.comp <- as.data.frame(p.comp$x[,1:2])
X_comp_Cl <- cbind.data.frame(X.comp, as.factor(df_new$classU))
names(X_comp_Cl) <- c("x", "y", "class")
#X_comp_Cl$class <- factor(X_comp_Cl$class, levels = c("G", "MinTr", "MinTe", "MajTe", "MajTr"))
levels(X_comp_Cl$class)
ggplot(data=X_comp_Cl, aes(x=x, y=-y)) + geom_point(data=X_comp_Cl[X_comp_Cl$class!="1Tr",], aes(color=class,shape=class), size=4) + theme(axis.title.x = element_blank(), axis.text.x = element_blank(), axis.ticks.x = element_blank(), axis.title.y = element_blank(), axis.text.y = element_blank(), axis.ticks.y = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), panel.background = element_blank(), legend.position = "top") + scale_color_manual(name = "", labels = c("Minority training", "Minority test", "Majority training", "Majority test", "Minority synthetic"), values=c("blue", "cyan1", "red", "rosybrown1", "blue")) + scale_shape_manual(name = "", labels = c("Minority training", "Minority test", "Majority training", "Majority test", "Minority synthetic"), values=c(16,16,16,16,10)) +
  geom_point(data=X_comp_Cl[X_comp_Cl$class=="1Tr",], aes(x,-y,color=class, shape=class), size=5) 

# SMOTE
Snew <- SMOTE(df_tr_red[,-9], df_tr_red[,9], K=5, dup_size = 12)$syn_data
Snew$classU <- "G"
df_new <- rbind.data.frame(df_n, Snew)

p.comp <- prcomp(df_new[,-cl], scale. = T)
X.comp <- as.data.frame(p.comp$x[,1:2])
X_comp_Cl <- cbind.data.frame(X.comp, df_new$classU)
names(X_comp_Cl) <- c("x", "y", "class")
ggplot(data=X_comp_Cl, aes(x=-x, y=y)) + geom_point(data=X_comp_Cl[X_comp_Cl$class!="1Tr",], aes(color=class,shape=class), size=4) + theme(axis.title.x = element_blank(), axis.text.x = element_blank(), axis.ticks.x = element_blank(), axis.title.y = element_blank(), axis.text.y = element_blank(), axis.ticks.y = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), panel.background = element_blank(), legend.position = "top") + scale_color_manual(name = "", labels = c("Minority training", "Minority test", "Majority training", "Majority test", "Minority synthetic"), values=c("blue", "cyan1", "red", "rosybrown1", "blue")) + scale_shape_manual(name = "", labels = c("Minority training", "Minority test", "Majority training", "Majority test", "Minority synthetic"), values=c(16,16,16,16,10)) +
  geom_point(data=X_comp_Cl[X_comp_Cl$class=="1Tr",], aes(-x,y,color=class, shape=class), size=5) 

