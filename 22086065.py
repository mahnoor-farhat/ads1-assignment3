# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 04:51:42 2024

@author: Mahnoor Farhat
"""
#https://github.com/mahnoor-farhat/ads1-assignment3.git

# Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import scipy.optimize as opt

# Read file
df = pd.read_csv(r"methane_emissions.csv", skiprows=4)
df

df2 = pd.read_csv(r"co2_emission.csv")

# Transpose & set countries, years as columns
df = df.dropna(axis=1, how='all')
df.set_index("Country Name", inplace=True)
df = df.dropna()
df = df.apply(pd.to_numeric, errors="coerce")

# Dataframe transposed
df_years = df.T

df_years = df_years.dropna()

columns_to_drop = ['Country Code', 'Indicator Name', 'Indicator Code']
df_countries = df.drop(columns=columns_to_drop)

print("Dataframe with Years as Columns:\n\n", df_years.columns, "\n\n",
      "Dataframe with Countries as Columns:\n\n", df_countries.columns)

# Clean data
drop_df = df.columns[0:3]
clean_df = df.drop(columns=drop_df, axis=1)
new_df = clean_df.reset_index()
new_df

# Fill missing data with mean
mean_fill = SimpleImputer(missing_values=np.nan, strategy='mean')
df1 = mean_fill.fit_transform(new_df.iloc[:, 1:])
df1

# Normalize data
scaler = StandardScaler()
norm_df = scaler.fit_transform(df1)
norm_df


def silhoutte_score(xy, n):
    
    """
    This function produces a silhoutte score for the respective data.

    """
    
    kmeans = cluster.KMeans(n_clusters=n, n_init=20)
    kmeans.fit(xy)
    labels = kmeans.labels_
    score = (skmet.silhouette_score(xy, labels))
    return score


for ic in range(2, 11):
    score = silhoutte_score(norm_df, ic)
    print(f"Silhouette score for {ic: 3d} is {score: 7.4f}")


def clustering():
    
    """
    This function produces clusters for the data as a graph.

    """

    kmeans = cluster.KMeans(n_clusters=3, n_init=20)
    kmeans.fit(norm_df)
    new_df['Cluster'] = kmeans.labels_
    center = kmeans.cluster_centers_
    center = scaler.inverse_transform(center)
    xkmeans = center[:, 0]
    ykmeans = center[:, 1]

    plt.figure(figsize=(12, 6))
    s = plt.scatter(new_df['1990'], new_df['2020'],
                    c=new_df['Cluster'], cmap='rainbow')
    plt.scatter(xkmeans, ykmeans, 45, "k", marker="d")
    plt.title('Cluster of Countries for Methane Emissions (1990 vs 2020)')
    plt.xlabel('Methane Emissions 1990')
    plt.ylabel('Methane Emissions 2020')
    plt.colorbar(s, label='Cluster')
    plt.show()

region = df2[df2['Entity'] == 'World']
region

plt.plot(region['Year'], region['Emission'], color='#2DD427')
plt.show()


def logistic(t, n0, g, t0):
    
    """
    Calculates the logistic function with scale factor n0 and growth rate g.
    
    """
    
    f = n0 / (1 + np.exp(-g*(t - t0)))
    return f


region["trial"] = logistic(region["Year"], 3e12, 0.10, 1990)
param, covar = opt.curve_fit(
    logistic, region["Year"], region["Emission"], p0=(3e12, 0.1, 1990))

# create array for forecasting
year = np.linspace(1990, 2090, 100)
forecast = logistic(year, *param)
plt.figure()

plt.plot(region["Year"], region["Emission"], label="Emission", color='#2DD427')
plt.plot(year, forecast, label="Prediction", color='#F8240B')
plt.title("Prediction for CO2 Emissions of the World")
plt.xlabel("Year")
plt.ylabel("Emission")
plt.legend()
plt.show()

clustering()
