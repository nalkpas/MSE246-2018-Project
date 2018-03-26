import pandas as pd


# Load Cleaned SBA Data, GDP by Industry, Housing Price Index by state,
# Consumer Price Index of White Bread, S&P 500 yearly returns

cleaned_sba = pd.read_csv("output.csv")
gdp_by_industry = pd.read_csv("../cleaned_data/GDPByIndPct1990-2014.csv",sep="\t")
hpi_by_state = pd.read_csv("../cleaned_data/HPI_State.csv",sep="\t")
cpi_bread = pd.read_csv("../cleaned_data/CPI_Bread_White.csv",sep="\t")
sp500_yearly = pd.read_csv("../cleaned_data/SP500_yearly_return.csv")

# Clean remaining erratas
gdp_by_industry.fillna("NA")

# In this approach, we consider macroeconomic variables to be fixed at the start of the loan approval year
# Add columns based on loan approval year : cpi, sp500

sba_w_macroecon = pd.merge(cleaned_sba,sp500_yearly,left_on='ApprovalFiscalYear',right_on='Year',how='left').drop("Year", axis=1).rename(index=str, columns={"Average yearly return": "SP500_Yearly_Return"})
sba_w_macroecon = pd.merge(sba_w_macroecon,cpi,left_on='ApprovalFiscalYear',right_on='Year',how='left').drop("Year", axis=1).rename(index=str, columns={"Average": "CPI"})

# Add columns based on loan approval year and project state : hpi_by_state
sba_w_macroecon = pd.merge(sba_w_macroecon,hpi_by_state,left_on=['ProjectState','ApprovalFiscalYear'],right_on=['State Abbreviation','Year'],how='left').drop(["State Abbreviation","Year","FIPS"], axis=1)

# Add columns based on industry(NAICS Code) and loan approval year : gdp_by_industry

sba_w_macroecon.to_csv("SBA_Macroecon.csv")