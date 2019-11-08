### Project: Product Profitability ###
## Date: 04 Nov 2019
## Author: Matias Barra

## Script purpose: analyze historical sales data and perform a sales prediction
## to understand how specific product types might impact sales across the enterprise

####--------------- Set Enviroment ---------------------------------------------####

load required libraries
if(require("pacman")=="FALSE"){
  install.packages("pacman")
}

pacman::p_load("readr", "ggplot2", "rpart", "rpart.plot", "caret","dplyr", "Hmisc", 
               "MASS", "mlr", "plotly", "corrplot", 
               "party", "ipred", "LearnBayes", "reshape")


####--------- import data sets and initial exploration -------------------------#### 
setwd("/Users/matiasbarra/Documents/Data_Analytics_Course/Data_Analytics_2/3_Multiple_Regression_R")
existing <- read.csv("./data/existingprod.csv", header = TRUE, check.names=FALSE,sep = ",", quote = "\'", as.is = TRUE)
summary(existing)
str(existing)

new <- read.csv("./data/newprod.csv", header = TRUE, check.names=FALSE,sep = ",", quote = "\'", as.is = TRUE)
summary(new)

existing1 <- filter(existing, ProductType %in% c("PC","Laptop","Smartphone","Netbook"))

str(existing) #80 obs. of  18 variables
str(existing1) #80 obs. of  18 variables
str(new)#24 obs. of  18 variables

summary(is.na(existing))  # confirm if any "NA" values in ds
summary(is.na(new))   # confirm if any "NA" values in ds


####----------------- Cleaning data -------------------------------------------------####

existing$BestSellersRank <- NULL 
existing$ProductNum<-NULL # It's only an identifier which doesn't add value
duplicated(existing)  # duplicates check -> no duplicates

#Dummify product types to use the attribute in Regresion 
existing.dummified <- dummyVars(" ~ .", data = existing)
existing.prod <- data.frame(predict(existing.dummified, newdata = existing))
# new df with dummified Product Type

####----------------- Outliers Detection --------------------------------------------####

# Plot data to evaluate the distribution by Product Type and try to find Outliers

plotly.volume.box <- plot_ly(existing, x=existing$Volume,
                             y=existing$ProductType, type = "box")

plotly.volume.box # Accessories, Extended Warranty and printer have volume outliers 

outliersVolume <- boxplot(existing.prod$Volume)$out # Boxplot to detect overall outliers 
boxplot(existing.prod$Volume)$out # Outliers values detected: 11204 and  7036

existing.prod <- existing.prod[-which(existing$Volume %in% 
                                      outliersVolume),] # Remove outliers in Volume

summary(existing.prod)

# mean of all prices for Extended Warranty (rows 34:41)
existing.prod[34:41,"Price"] <- mean(existing.prod[34:41,"Price"])
existing.prod <- existing.prod[-c(35:41),] 
# delete the rest of rows for Extended Warranty    

str(existing.prod)

####----------------- Correlation --------------------------------------------------####

corrData <- cor(existing.prod)
corrData


corre_mat <- corrplot(corrData, type = "upper", tl.pos = "td",
             method = "square", tl.cex = 0.5, order= "hclust", tl.col = 'black', diag = TRUE)

corre_mat_1 <- corrplot(corrData, 
                        method = "number", 
                        tl.cex= 0.6, number.cex = 0.6)

corre_mat_2 <- corrplot(corrData, type = "upper", order = "hclust", 
                        corrData = corrData, sig.level = 0.05, insig = "blank",
                        tl.srt=45, tl.cex = 0.7, tl.col="black")

# 4 and 3 stars reviews are the most correlated variables to volume (followed by Positive)
# however, as 4 and 3 stars are strongly correlated to each other, 
# 4 stars is kept as its correlation to Volume is higher.
# PositiveServiceReview has good correlated with Volume and not good correlation with 4 stars
# Moreover, 5 star Reviews is excluded as it might be flawed data (correlation = 1)
existing$x5StarReviews <- NULL
existing.prod$x5StarReviews <- NULL

control.tree <- ctree_control(maxdepth = 10)
Decision.tree <- ctree (Volume ~ ., data=existing.prod, 
                        controls = control.tree)
plot(Decision.tree) 
#Decision tree verifies that the most correlated variables to volume are 4 stars and Positive reviews


####-------------- Models -----------------------------------------------------####

# Train and test model 
set.seed(123)

inTraining <- createDataPartition(existing.prod$Volume, p =0.8, list = FALSE)
trainSet <- existing.prod[inTraining,]
testSet <- existing.prod[-inTraining,]

str(trainSet) # 59 obs of 26 var 
str(testSet) # 12 obs of 26 var

#Train Control | Using "Cross validation" method
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 5)


#Models and variables

mod <-  c("rf","svmLinear2","knn") #list of models to try
var <-  c("Volume ~ x4StarReviews", "Volume ~ x4StarReviews + PositiveServiceReview") #list of variables

compare_mod_var <- c()
models <- list() #define a list when i'm going to acumulate models metrics
k <- 1 

for(i in mod){
  print(i)
  for(j in var){
    print(j)
    model <- caret::train(formula(j), data = trainSet, method = i,
                   trControl=fitControl, preProcess=c("center", "scale"))
    
    models[[k]] <- model
    k <- k + 1 
    
    pred <- predict(model, newdata = testSet)
    pred_metric <- caret::postResample(testSet$Volume, pred)
    compare_mod_var <- cbind(compare_mod_var, pred_metric)
  }
}

names_var <- c()
for(i in mod){
  for (j in var) {
    names_var <- append(names_var, paste(i,j))
  }
}
models

colnames(compare_mod_var) <- names_var
compare_mod_var

compare_mod_var_melt <- melt(compare_mod_var, varnames = c("metric", "model"))
compare_mod_var_melt <- as.data.frame(compare_mod_var_melt)   #as_data_frame(compare_model_melt)
compare_mod_var_melt 

# Ploting Metrics of each Model

ggplot(compare_mod_var_melt[which(compare_mod_var_melt$metric=="MAE"),], aes(x=model, y=value,fill=model))+
  geom_col() + ggtitle("MAE Metrics in diferents Models") +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank())

ggplot(compare_mod_var_melt[which(compare_mod_var_melt$metric=="RMSE"),], aes(x=model, y=value,fill=model))+
  geom_col() + ggtitle("RMSE Metrics in diferents Models") +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank())

ggplot(compare_mod_var_melt[which(compare_mod_var_melt$metric=="Rsquared"),], aes(x=model, y=value,fill=model))+
  geom_col() + ggtitle("Rsqared Metrics in diferents Models") +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank())

# Ploting "Compare Models and Metrics"

ggplot(compare_mod_var_melt, aes(x=model, y=value,fill=model))+ 
  geom_col() + ggtitle("Compare Models and Metrics") +
  facet_grid(metric~., scales="free") +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank())

####-------------- Predictions of the models in New ds ----------------------------####

new$x5StarReviews <- NULL
new$x5StarReviews <- NULL

str(new)

PredNew <- predict(models[[6]], newdata = new)  
PredNew

new$PredictedVolume_2 <- PredNew #Sales volume prediction with models[[6]]

new$Profitability <- new$Price * new$ProfitMargin * new$PredictedVolume_2 #profitability in the New Products

#Plot prediction Sales Volume in 4 product categories 
ggplot(new[new$ProductType == "PC" | 
             new$ProductType == "Laptop" | 
             new$ProductType == "Netbook" | 
             new$ProductType == "Smartphone",], 
       aes(x = ProductType, y = PredictedVolume_2, fill = as.character(ProductNum))) + 
       geom_col() +
       ggtitle("Sales by Product Category") +
       ylab("Sales Volume") +
       xlab("Product Type") +
       guides(fill=guide_legend(title="Product Number"))

#Plot prediction Profitability in 4 product categories 
ggplot(new[new$ProductType == "PC" |
             new$ProductType == "Laptop" |
             new$ProductType == "Netbook" |
             new$ProductType == "Smartphone",],
       aes(x = ProductType, y = Profitability, fill = as.character(ProductNum))) +
       geom_col() +
       ggtitle("Profitability by Product Category") +
       ylab("Profitability") +
       xlab("Product Type") +
       guides(fill = guide_legend(title="Product Number"))

#Plot prediction Sales Volume in all products
ggplot(new, aes(x = ProductType, y = PredictedVolume_2, fill = as.character(ProductNum))) + 
      geom_col() +
      ggtitle("Sales by Product") +
      ylab("Sales Volume") +
      xlab("Product Type") +
      guides(fill = guide_legend(title="Product Number"))

      #Plot prediction Sales Volume in all products
ggplot(new, aes(x = ProductType, y = Profitability, fill = as.character(ProductNum))) + 
      geom_col() +
      ggtitle("Profitability by Product") +
      ylab("Sales Volume") +
      xlab("Product Type") +
      guides(fill = guide_legend(title="Product Number"))


