# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 00:41:50 2022

@author: Titobiloba
"""

#As we begin our analysis, we need to import the necessary libraries (Pandas, Numpy and Matplotlib)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#For the purpose of this analysis, we would define a function

# A function to read in files in the Worldbank format returning original and transposedformat
def stats(filename,countries,columns,indicator):
    df = pd.read_csv(filename,encoding='latin1')
    df = df[df['Indicator Name'] == indicator]
    df = df[columns]
    df.set_index('Country Name', inplace = True)
    df = df.loc[countries]
    return df,df.transpose()

filename = 'Series_metadata.csv'
countries = ['Ireland','Angola','Jordan','Libya']
columns = ['Country Name', '2010','2011','2012','2013','2014','2015','2016','2017','2018','2019']
indicators = ['CO2 emissions (kt)', 'Arable land (% of land area)', 'Forest area (% of land area)','GDP per capita growth (annual %)']

cnty_co2,year_co2 = stats(filename,countries,columns,indicators[0])
cnty_land,year_land = stats(filename,countries,columns,indicators[1])
cnty_forest,year_forest = stats(filename,countries,columns,indicators[2])
cnty_gdp,year_gdp = stats(filename,countries,columns,indicators[3])

#The describe() method computes and displays summary statistics for the following
print(cnty_co2.describe())
print(year_co2.describe())

year_co2 = year_co2.astype(float)

#We plot a line graph for the Year on Year Trend of the CO2 Emission for these 4 countries
plt.figure(figsize=(11,7),dpi=500)
for i in range(len(countries)):
    plt.plot(year_co2.index,year_co2[countries[i]],label=countries[i])
plt.title('Year on Year Trend of the CO2 Emission for these 4 countries')
plt.xlabel('Year')
plt.ylabel('CO2 Emision')
plt.legend()
plt.show()

year_gdp = year_gdp.astype(float)

#We plot a line graph for the Year on Year Trend of the GDP for these 4 countries
plt.figure(figsize=(10,7),dpi=500)
for i in range(len(countries)):
    plt.plot(year_gdp.index,year_gdp[countries[i]],label=countries[i])
plt.title('Year on Year Trend of the GDP for these 4 countries')
plt.xlabel('Year')
plt.ylabel('GDP')
plt.legend()
plt.show()

#we want to plot the heat map for 2 countries, to do that we have to the correlation for them

#We want to do the correlation for Jordan to enable us do the heat map
Jordan = pd.DataFrame(
 {'C02 emission': year_co2['Jordan'].astype(float),
 'Arable land': year_land['Jordan'].astype(float),
 'Forest Area': year_forest['Jordan'].astype(float),
 'GDP': year_gdp['Jordan'].astype(float)},
 ['2010','2011','2012','2013','2014','2015','2016','2017','2018','2019'])

print(Jordan.corr())

#We plot the heat map for Jordan to figure out the correlation between the indicators
plt.figure(figsize=(8,5))
sns.heatmap(Jordan.corr(),annot=True,cmap='Greens')
plt.title('Correlation heatmap for Jordan')
plt.show()

#We want to do the correlation for Croatia to enable us do the heat map
Angola = pd.DataFrame(
 {'C02 emission': year_co2['Angola'].astype(float),
 'Arable land': year_land['Angola'].astype(float),
 'Forest Area': year_forest['Angola'].astype(float),
 'GDP': year_gdp['Angola'].astype(float)},
 ['2010','2011','2012','2013','2014','2015','2016','2017','2018','2019'])

print(Angola.corr())

#eE plot the heat map for Jordan to figure out the correlation between the indicators
plt.figure(figsize=(8,5))
sns.heatmap(Angola.corr(),annot=True,cmap='Greens')
plt.title('Correlation heatmap For Angola')
plt.show()

#We will now plot a bar chart for the forest area in the countries over the years
cnty_forest =  cnty_forest.astype(float)
cnty_forest.plot(kind='bar')
plt.title('Grouped bar for Forest Area in the countries over the years')
plt.xlabel('Countries')
plt.ylabel('Forest Area')
plt.rcParams["figure.dpi"] = 500
plt.show()

#We will now plot a bar chart for the Arable Land in the countries over the years
cnty_land =  cnty_land.astype(float)
cnty_land.plot(kind='bar')
plt.title('Grouped bar for Arable Land in the countries over the years')
plt.xlabel('Countries')
plt.ylabel('Arable Land')
plt.rcParams["figure.dpi"] = 500
plt.show()