library("survival")
library("survminer")

train <- read.csv(file="MSE246-2018-Project/hazard\ model/hazard_train.csv", head=TRUE, sep=",")
test <- read.csv(file="MSE246-2018-Project/hazard\ model/hazard_test.csv", head=TRUE, sep=",")

## BorrCity, BorrState, BorrZip
## CDC_Name, CDC_Street, CDC_City, CDC_State, CDC_Zip
## ThirdPartyLender_Name, ThirdPartyLender_City, ThirdPartyLender_State, ThirdPartyDollars
## GrossApproval, ApprovalFiscalYear, DeliveryMethod
## subpgmdesc, TermInMonths, NaicsCode, ProjectCounty, ProjectState
## BusinessType, GrossChargeOffAmount

res.cox <- coxph(Surv(DaysToDefault, Default.) ~ 
  DeliveryMethod + ThirdPartyDollars + BusinessType + TermInMonths + GrossApproval + BorrState, data = train)
summary(res.cox)

# Plot the baseline survival function
ggsurvplot(survfit(res.cox, data=train), palette = "#2E9FDF",
           ggtheme = theme_minimal())

# Plot the survival function for a single test datapoint
test[1,]
ggsurvplot(survfit(res.cox, data=test[1,]), palette = "#F84018",
           ggtheme = theme_minimal())

# There are more types for DeliveryMethod in test?
modified_test <- test[(test$DeliveryMethod == 'ALP' | test$DeliveryMethod == 'PCLP'),]
results <- modified_test[c('Default.', 'DaysToDefault')]
results <- cbind(results, lp=predict(res.cox, newdata=modified_test, type="lp"))
results <- cbind(results, risk=predict(res.cox, newdata=modified_test, type="risk"))
results <- cbind(results, expected=predict(res.cox, newdata=modified_test, type="expected"))
results <- cbind(results, terms=predict(res.cox, newdata=modified_test, type="terms"))
head(results)

# https://www.r-bloggers.com/cox-proportional-hazards-model/
# https://rdrr.io/cran/survivalROC/man/survivalROC.C.html