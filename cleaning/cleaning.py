import pandas as pd 
import numpy as np 

data = pd.read_excel("SBA_Loan_data_.xlsx", "Sheet1")

# drop irrelevant or bad data
data = data[data.LoanStatus != 'CANCLD']
data = data[data.LoanStatus != 'EXEMPT']
data = data.drop(columns=['Program'])
# drops 2 entries, one in Puerto Rico and one that's nonsense
data = data.dropna(subset=['BorrState'])

# specify that NAICS and ZIP codes are categorical, need to fill these before casting
data[['NaicsCode','BorrZip','CDC_Zip']] = data[['NaicsCode','BorrZip','CDC_Zip']].fillna("MISSING")
data[['NaicsCode','BorrZip','CDC_Zip']] = data[['NaicsCode','BorrZip','CDC_Zip']].astype(str)

# fill missing numerical data
numerics = data.select_dtypes(include=np.number)
means = dict(zip(numerics.columns, np.nanmean(numerics,axis = 0)))
numerics = numerics.fillna(means)
data.update(numerics)

# fill missing categorical data
cats = data.drop(columns=numerics.columns)
cats = cats.fillna("MISSING")
data.update(cats)

# remove nonsense ZIP codes
file = open("zipcodes.csv","r")
zip_ref = [line.strip().split(",") for line in file]
zip_ref = {line[2] : (int(line[3]),int(line[4])) for line in zip_ref}
zip_ref["MISSING"] = (1,-1)
zip_ref["GU"] = (96910,96932)
zip_ref["VI"] = (801, 851)
zips = data[['BorrState','BorrZip','CDC_State','CDC_Zip']]
for index, series in data.iterrows():
	borr_bot, borr_top = zip_ref[series['BorrState']]
	cdc_bot, cdc_top = zip_ref[series['CDC_State']]
	if(series['BorrZip'] == "MISSING"):
		borr_zip = -1
	else:
		borr_zip = float(series['BorrZip']) 
	if(series['CDC_Zip'] == "MISSING"):
		cdc_zip = -1
	else:
		cdc_zip = float(series['CDC_Zip']) 

	if borr_bot > borr_zip or borr_zip > borr_top:
		zips.at[index,'BorrZip'] = "MISSING"
	if borr_bot > cdc_zip or cdc_zip > borr_top:
		zips.at[index,'CDC_Zip'] = "MISSING"
data.update(zips)

data.to_csv("output.csv")