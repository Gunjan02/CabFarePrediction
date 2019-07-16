rm(list = ls())
setwd('C:/Users/admin/Desktop')

cab_df = read.csv('train_cab.csv',header = T)
 x = c("ggplot2","corrgram","DMwR","caret","randomForest","unbalanced",
       "C50","dummies","e1071","Information","MASS","rpart","gbm","ROSE","lubridate"
       ,"dplyr")
 lapply(x,require,character.only = TRUE)
 rm(x)
 
 str(cab_df)
cab_df$fare_amount = as.numeric(as.character(cab_df$fare_amount))
cab_df$pickup_datetime = parse_date_time(cab_df$pickup_datetime,orders = "ymd HMS")

missing_val = data.frame(apply(cab_df, 2,function(x)(sum(is.na(x))) ))
missing_val$columns = row.names(missing_val)
names(missing_val)[1] = "Missing_Percentage"
missing_val$Missing_Percentage = (missing_val$Missing_Percentage/nrow(cab_df))*100
missing_val = missing_val[order(-missing_val$Missing_Percentage),]
missing_val = missing_val[,c(2,1)]
row.names(missing_val) = NULL

 
colSums(is.na(cab_df))
cab_df = subset(cab_df,!is.na(pickup_datetime))

#Actual value : 6.9  
#mean : 15.01
#median : 8.5
#knn : 7.01
cab_df[71,7] = NA
#cab_df$fare_amount[is.na(cab_df$fare_amount)] = mean(cab_df$fare_amount,na.rm = T)
#cab_df$fare_amount[is.na(cab_df$fare_amount)] = median(cab_df$fare_amount,na.rm = T)

cab_df$pickup_datetime = as.numeric(cab_df$pickup_datetime)
cab_df = knnImputation(cab_df,k = 7)

cab_df$pickup_datetime = as.POSIXct( as.numeric( as.POSIXct( cab_df$pickup_datetime, origin = '1970-01-01', tz = "UTC" ) ), 
                                     origin = '1970-01-01', tz = "UTC" )

cab_df$passenger_count = round(cab_df$passenger_count)
summary(cab_df)


numeric_index = sapply(cab_df, is.numeric)
numeric_data = cab_df[,numeric_index]
cnames = colnames(numeric_data)

boxplot(cab_df$fare_amount,cab_df$pickup_longitude,cab_df$pickup_latitude)
boxplot(cab_df$dropoff_longitude,cab_df$dropoff_latitude,cab_df$passenger_count)


for (i in cnames){
        print(i)
        val = cab_df[,i][cab_df[,i]%in%boxplot.stats(cab_df[,i])$out]
        print(length(val))
        cab_df = cab_df[which(!cab_df[,i]%in%val),]
}

summary(cab_df)

cab_df = subset(cab_df,fare_amount >= 1)
cab_df = subset(cab_df,passenger_count>=1)

distance = function (long1, lat1, long2, lat2)
{
        rad <- pi/180
        a1 <- lat1 * rad
        a2 <- long1 * rad
        b1 <- lat2 * rad
        b2 <- long2 * rad
        dlon <- b2 - a2
        dlat <- b1 - a1
        a <- (sin(dlat/2))^2 + cos(a1) * cos(b1) * (sin(dlon/2))^2
        c <- 2 * atan2(sqrt(a), sqrt(1 - a))
        R <- 6378.145
        d <- R * c
        return(d)
}

cab_df$year = year(cab_df$pickup_datetime)
cab_df$month = month(cab_df$pickup_datetime)
cab_df$day = day(cab_df$pickup_datetime)
cab_df$weekday = wday(cab_df$pickup_datetime)
cab_df$hour = hour(cab_df$pickup_datetime)

cab_df$pickup_datetime = NULL

cab_df$distance = distance(cab_df$pickup_longitude,cab_df$pickup_latitude,
                           cab_df$dropoff_longitude,cab_df$dropoff_latitude)


corrgram(cab_df[,numeric_index],order = F,
         upper.panel = panel.pie,text.panel = panel.txt, main = "Correlation Plot")

library("scales")


library("psych")
library("gplots")




ggplot(cab_df,aes_string(x=cab_df$distance,y=cab_df$fare_amount))+
        geom_point(inherit.aes = TRUE,size=3)+
     theme_bw()+ylab("Fare")+xlab("Distance")+ggtitle("Scatter Plot b/w distance and fare")+
      theme(text=element_text(size = 15))+
       scale_x_continuous(breaks = pretty_breaks(10))+
        scale_y_continuous(breaks = pretty_breaks(10))


ggplot(cab_df,aes_string(x = cab_df$hour))+
        geom_bar(stat = "count",fill = "DarkSlateBlue")+theme_bw()+
        xlab("Hours")+ylab("Frequency")+scale_y_continuous(breaks = pretty_breaks(20))+
        ggtitle("Frequency of hours")+theme(text = element_text(size = 12))

ggplot(cab_df,aes_string(x=cab_df$hours,y=cab_df$fare_amount))+
        geom_point(aes_string(cab_df$hour,cab_df$fare_amount),size=3)+
        theme_bw()+ylab("Fare")+xlab("Hours")+ggtitle("Scatter Plot b/w hours and fare")+
        theme(text=element_text(size = 15))+
        scale_x_continuous(breaks = pretty_breaks(10))+
        scale_y_continuous(breaks = pretty_breaks(10))

ggplot(cab_df,aes_string(x = cab_df$passenger_count))+
        geom_histogram(fill = "red",colour="black",bins = 15)+geom_density()+
        scale_y_continuous(breaks = pretty_breaks(10))+
        scale_x_continuous(breaks = pretty_breaks(10))+theme_bw()+
        xlab("Passengers")+ylab("Frequency")+
        ggtitle("Frequency of Passengers")+theme(text = element_text(size = 10))

ggplot(cab_df,aes_string(x = cab_df$weekday))+
        geom_histogram(fill = "green",colour="black",bins = 70)+geom_density()+
        scale_y_continuous(breaks = pretty_breaks(10))+
        scale_x_continuous(breaks = pretty_breaks(10))+theme_bw()+
        xlab("Weekdays")+ylab("Frequency")+
        ggtitle("Frequency of weekdays")+theme(text = element_text(size = 10))


ggplot(cab_df,aes_string(x = cab_df$fare_amount))+
        geom_histogram(fill = "cornsilk",colour="black")+geom_density()+
        scale_y_continuous(breaks = pretty_breaks(10))+
        scale_x_continuous(breaks = pretty_breaks(10))+theme_bw()+
        xlab("Fare")+ylab("Frequency")+
        ggtitle("Fare")+theme(text = element_text(size = 12))



ggplot(cab_df,aes_string(x=cab_df$day,y=cab_df$fare_amount))+
        geom_point(aes_string(cab_df$day,cab_df$fare_amount),size=3)+
        theme_light()+ylab("Fare")+xlab("Day")+ggtitle("Scatter Plot b/w day and fare")+
        theme(text=element_text(size = 15))+
        scale_x_continuous(breaks = pretty_breaks(10))+
        scale_y_continuous(breaks = pretty_breaks(10))

ggplot(cab_df,aes_string(x=cab_df$passenger_count,y=cab_df$fare_amount))+
        geom_point(aes_string(cab_df$day,cab_df$fare_amount),size=3)+
        theme_light()+ylab("Fare")+xlab("Passengers")+ggtitle("Scatter Plot b/w Passenger and fare")+
        theme(text=element_text(size = 15))+
        scale_x_continuous(breaks = pretty_breaks(10))+
        scale_y_continuous(breaks = pretty_breaks(10))

library(rsq)
library(usdm)
vif(cab_df[,2:8])
vifcor(cab_df[,2:8],th = 0.9)


set.seed(1234)
train_index = sample(1:nrow(cab_df),0.8*nrow(cab_df))
train = cab_df[train_index,]
test = cab_df[-train_index,]
#Multiple Linear Regression
lm_model = lm(fare_amount~.,data = train)
summary(lm_model)
predictions_lr = predict(lm_model,test[,2:12])
RMSE(predictions_lr,test$fare_amount)
#2.17

#Decision Tree
fit = rpart(fare_amount~.,data = train,method = "anova")
predictions_dt = predict(fit,test[,-1])
RMSE(predictions_dt,test$fare_amount)
#2.30


#Random Forest
model_rf = randomForest(fare_amount~.,
                                    train,importance = TRUE, ntree = 300)

RF_predictions = predict(model_rf,test[,2:12])
RMSE(RF_predictions,test$fare_amount)
#1.99



test_df = read.csv('test.csv',header = T)
summary(test_df)

str(test_df)
test_df$pickup_datetime = parse_date_time(test_df$pickup_datetime,orders = "ymd HMS")

num_index = sapply(test_df, is.numeric)
num_data = test_df[,num_index]
c_names = colnames(num_data)

for (i in c_names){
        print(i)
        val = test_df[,i][test_df[,i]%in%boxplot.stats(test_df[,i])$out]
        print(length(val))
        test_df = test_df[which(!test_df[,i]%in%val),]
}

summary(test_df)

test_df$year = year(test_df$pickup_datetime)
test_df$month = month(test_df$pickup_datetime)
test_df$day = day(test_df$pickup_datetime)
test_df$weekday = wday(test_df$pickup_datetime)
test_df$hour = hour(test_df$pickup_datetime)

test_df$pickup_datetime = NULL

test_df$distance = distance(test_df$pickup_longitude,test_df$pickup_latitude,
                           test_df$dropoff_longitude,test_df$dropoff_latitude)
test_df = subset(test_df,distance>=1)
predicted_fare = predict(model_rf,test_df[,])
test_df$predicted_fare = predicted_fare
summary(test_df)

