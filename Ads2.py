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

def stats(filename,countries,columns,indicator):
    df = pd.read_csv(filename,encoding='latin1')
    df = df[df['Indicator Name'] == indicator]
    df = df[columns]
    df.set_index('Country Name', inplace = True)
    df = df.loc[countries]
    return df,df.transpose()

filename = 'Series_metadata.csv'
countries = ['Ireland','Croatia','Jordan','Libya']
columns = ['Country Name', '2010','2011','2012','2013','2014','2015','2016','2017','2018','2019']
indicators = ['CO2 emissions (kt)', 'Arable land (% of land area)', 'Forest area (% of land area)','GDP per capita growth (annual %)']

cnty_co2,year_co2 = stats(filename,countries,columns,indicators[0])
cnty_land,year_land = stats(filename,countries,columns,indicators[1])
cnty_forest,year_forest = stats(filename,countries,columns,indicators[2])
cnty_gdp,year_gdp = stats(filename,countries,columns,indicators[3])

print(cnty_co2.describe())






