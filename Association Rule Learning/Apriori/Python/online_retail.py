import numpy as np 
import pandas as pd 
from apyori import apriori




# Loading the Data 
data = pd.read_excel('Online_Retail.xlsx') 
data.head() 


# Exploring the columns of the data 
data.columns 

# Exploring the different regions of transactions 
data.Country.unique() 

# Stripping extra spaces in the description 
data['Description'] = data['Description'].str.strip() 

# Dropping the rows without any invoice number 
data.dropna(axis = 0, subset =['InvoiceNo'], inplace = True) 
data['InvoiceNo'] = data['InvoiceNo'].astype('str') 

# Dropping all transactions which were done on credit 
data = data[~data['InvoiceNo'].str.contains('C')] 

# Transactions done in France 
basket_France = (data[data['Country'] =="France"] 
		.groupby(['InvoiceNo', 'Description'])['Quantity'] 
		.sum().unstack().reset_index().fillna(0) 
		.set_index('InvoiceNo')) 

# Transactions done in the United Kingdom 
basket_UK = (data[data['Country'] =="United Kingdom"] 
		.groupby(['InvoiceNo', 'Description'])['Quantity'] 
		.sum().unstack().reset_index().fillna(0) 
		.set_index('InvoiceNo')) 

# Transactions done in Portugal 
basket_Por = (data[data['Country'] =="Portugal"] 
		.groupby(['InvoiceNo', 'Description'])['Quantity'] 
		.sum().unstack().reset_index().fillna(0) 
		.set_index('InvoiceNo')) 

basket_Sweden = (data[data['Country'] =="Sweden"] 
		.groupby(['InvoiceNo', 'Description'])['Quantity'] 
		.sum().unstack().reset_index().fillna(0) 
		.set_index('InvoiceNo')) 

# Defining the hot encoding function to make the data suitable 
# for the concerned libraries 
def hot_encode(x): 
	if(x<= 0): 
		return 0
	if(x>= 1): 
		return 1

# Encoding the datasets 
basket_encoded = basket_France.applymap(hot_encode) 
basket_France = basket_encoded 

basket_encoded = basket_UK.applymap(hot_encode) 
basket_UK = basket_encoded 

basket_encoded = basket_Por.applymap(hot_encode) 
basket_Por = basket_encoded 

basket_encoded = basket_Sweden.applymap(hot_encode) 
basket_Sweden = basket_encoded 

# Building the model 
frq_items = apriori(basket_France, min_support = 0.05, use_colnames = True) 

#results = list(frq_items)
#results


frq_items2 = apriori(basket_UK, min_support = 0.05, use_colnames = True) 

#results2 = list(frq_items2)
#results2

frq_items3 = apriori(basket_Por, min_support = 0.05, use_colnames = True) 
#results3 = list(frq_items3)
#results3


frq_items4 = apriori(basket_Sweden, min_support = 0.05, use_colnames = True) 
#results4 = list(frq_items4)
#results4



