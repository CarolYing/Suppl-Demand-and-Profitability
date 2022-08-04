#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 10:19:40 2022

@author: mengfanying
"""

################################################################################
################################################################################
# Content
# MC Simulation for supply ------------------------------ Line 44
# MC Simulation for demand ------------------------------ Line 116
# Price ------------------------------------------------- Line 187
# MC Simulation for profit ------------------------------ Line 237
# MC Simulation for profit (Correlation model) ---------- Line 401
################################################################################
################################################################################

################################################################################
# packages
################################################################################
import numpy as np
import numpy.random as rand

import MonteCarlo_3 as mc

import pandas as pd
import numpy as np
#from scipy.stats import jexponnorm
from scipy.stats import norm, t, burr, betaprime, f, genexpon, laplace, geninvgauss, weibull_min, gumbel_r, expon, cauchy, gamma

from scipy.stats import probplot
import scipy.stats
import matplotlib.pyplot as plt
from os import chdir, getcwd
import csv

################################################################################
################################################################################
################################################################################

################################################################################
# MC Simulation for supply
################################################################################
class TeslaSupplySimulator(mc.MonteCarlo):
    '''
    Used to simulate the supply(production)
    '''
    
    def __init__(self, p_mu, p_std, p_curr, p_quarter):
        # production growth rate is a normal distribution. Mean: mu, Std: std
        self.mu = p_mu  
        self.std = p_std
        self.curr = p_curr # The latest production
        self.quarter = p_quarter   # prediction in how many quarters? 
        # end of 2022: 4 quarters
        # end of 2023: 8 quarters
        # end of 2024: 12 quarters


    def SimulateOnce(self):
        curr = self.curr
        production = []
        for i in range(self.quarter):

                rate = rand.normal(self.mu, self.std)
                curr = curr * (1 + rate)
                production.append(curr)
        
        yearly_accu_production = np.sum(production[-4:])
        
        return yearly_accu_production

################################################################################
# MC Simulation for supply
# This is a test: run simulation once
################################################################################
# historyial production data
production_cut = [145036, 179757, 180338, 206421, 237823, 305840]

# growth rate
def growthRate(curr, prev):
    rate = (curr - prev)/prev
    return rate

def growthRatepInfo(productionLst):
    rateLst = []
    for i in range(len(productionLst)-1):  
        curr = productionLst[i+1]
        prev = productionLst[i]
        rate = growthRate(curr, prev)
        rateLst.append(rate)
        
    mu = np.mean(rateLst)
    std = np.std(rateLst)
    
    return mu, std

# mean and std of growth rate
mu, std = growthRatepInfo(production_cut)

# run simulation once
projectPeriod = [4, 8, 12]
year = [2022, 2023, 2024]
for p, y in zip(projectPeriod, year):
    sim = TeslaSupplySimulator(mu, std, production_cut[-1], p)
    prod = round(sim.SimulateOnce(), 0)
    print(f'The estimated production by the end of {y} is {prod} unit vehicles')

################################################################################
################################################################################
################################################################################  
    
################################################################################
# MC Simulation for demand
################################################################################
def SimulateOnce(Sold, Residual,quarters, loc, scale):
    tsSold = Sold
    tsResid = Residual
    yearFirst=[]
    yearSecond=[]
    yearThird=[]
    projPeriod = quarters #12 quarters
    count = len(tsResid)

    #Loop through "quarters" number of projected auarters
    for i in range(projPeriod): #Goes from 0 to 11
        simErr = np.random.laplace(loc, scale, 1)[0] #Gives an array, so added [0] to get just the number
        predSold_Log = np.log(tsSold[count-1]) - 0.37 * tsResid[count-1] - 0.62 * tsResid[count-6]  + 0.135 + simErr
        #Used count-1 and count-6 because index for the ts's starts at zero (goes from 0 to 23)
        #print(count, np.log(tsSold[count-1]), tsResid[count-1], tsResid[count-6], simErr, predSold_Log)
        predSold = round(np.exp(predSold_Log),0).astype(int)
        #print(predSold)    
        tsSold.loc[len(tsSold)] = predSold #Add to tsSold dataframe
        #print(tsSold)
        #simErr
        tsResid.loc[len(tsResid)] = simErr #Add to tsSold dataframe
        count = count + 1 
    #print(tsResid)
    #print(tsSold)
    if quarters>=12:
        yearThird.append(tsSold[len(tsSold)-1] + tsSold[len(tsSold)-2] + tsSold[len(tsSold)-3] + tsSold[len(tsSold)-4])
        yearSecond.append(tsSold[len(tsSold)-5] + tsSold[len(tsSold)-6] + tsSold[len(tsSold)-7] + tsSold[len(tsSold)-8])
        yearFirst.append(tsSold[len(tsSold)-9] + tsSold[len(tsSold)-10] + tsSold[len(tsSold)-11] + tsSold[len(tsSold)-12])
    elif quarters>=8:
        yearSecond.append(tsSold[len(tsSold)-5] + tsSold[len(tsSold)-6] + tsSold[len(tsSold)-7] + tsSold[len(tsSold)-8])
        yearFirst.append(tsSold[len(tsSold)-9] + tsSold[len(tsSold)-10] + tsSold[len(tsSold)-11] + tsSold[len(tsSold)-12])
    elif quarters>=4:
        yearFirst.append(tsSold[len(tsSold)-9] + tsSold[len(tsSold)-10] + tsSold[len(tsSold)-11] + tsSold[len(tsSold)-12])
    #yearFirst
    #yearSecond
    #yearThird
    
    return yearFirst, yearSecond, yearThird #Demand_Total and Residuals with additional project quarters tacked on

################################################################################
# MC Simulation for demand
# This is a test: run simulation once
################################################################################
getcwd()
chdir('/Users/mengfanying/Documents/A_CSC 521/Final project/Final model')

df = pd.read_csv("TeslaSold_10K.csv")  #open dataset
print(df)
tsSold = df["Demand_Total"]
df_resid = pd.read_csv("TimeSeriesResiduals.csv")  #open dataset
tsResid = df_resid["Residuals"]  

loc, scale = laplace.fit(tsResid[1:]) 

# run simulation once
s = SimulateOnce(df["Demand_Total"], df_resid["Residuals"], 12, loc, scale)
demand_2022 = s[0][0]
demand_2023 = s[1][0]
demand_2024 = s[2][0]

print(f'The estimated demand by the end of 2022 is {demand_2022} unit vehicles')
print(f'The estimated demand by the end of 2023 is {demand_2023} unit vehicles')
print(f'The estimated demand by the end of 2024 is {demand_2024} unit vehicles')
    
################################################################################
################################################################################
################################################################################      
    
################################################################################
# Price
################################################################################
price = pd.read_csv("Tesla_PriceData.csv")  #open dataset
# print(price)

# starting price is the average price in 2021
startPrice = np.mean(price['AdjPrice_Total'].iloc[-4:])
# changing rate of price is a normal distribution with mean 0.008 and std 0.072
priceChangeRate = rand.normal(0.008, 0.072) 
#Cap simulation of price change at +/- 15.5% and +14.5%

def PriceEstimationOld(currPrice, quarter):
    count = 0
    priceChangeRate = 0.0 # limit the price
    while count< quarter:
        qtrChangeRate = rand.normal(0.008, 0.072)
        if qtrChangeRate > 0.145 or qtrChangeRate < -0.155:
            pass            
        else:
            priceChangeRate = qtrChangeRate + priceChangeRate #Sum the quarterly changes for an average calculation below
            count = count +1
    priceChangeRate = (1 + priceChangeRate / 4) ** 4 - 1 #Take the average and then take to the 4th to get an average annual change
    currPrice = currPrice * (1 + priceChangeRate)
    
    return currPrice
    

def PriceEstimation(currPrice):  ############CHANGED TO MATCH CODE in PROFIT - 2022, 2023, and 2024 SECTIONS
    count = 0
    priceChangeRate = 0.0 # limit the price
    while count<4:
        qtrChangeRate = rand.normal(0.008, 0.072)
        if qtrChangeRate > 0.145 or qtrChangeRate < -0.155:
            pass            
        else:
            priceChangeRate = qtrChangeRate + priceChangeRate #Sum the quarterly changes for an average calculation below
            count = count +1
    priceChangeRate = (1 + priceChangeRate / 4) ** 4 - 1 #Take the average and then take to the 4th to get an average annual change
    price = currPrice * (1 + priceChangeRate)
    
    return price

PriceEstimation(startPrice)


################################################################################
################################################################################
################################################################################  

################################################################################
# MC Simulation for profit
################################################################################
# profit = demand * price - supply * cost

###############################################################################
# profit - 2022
###############################################################################
profit_2022 = []
start_price_2023=[] #ADDED

for i in range (10000):
    
        # simulate supply
        sim = TeslaSupplySimulator(mu, std, production_cut[-1], 4) # 4 quarters
        prod_2022 = round(sim.SimulateOnce(), 0)

        # simulate demand
        s = SimulateOnce(df["Demand_Total"], df_resid["Residuals"], 12, loc, scale)
        demand_2022 = s[0][0] # 2022

        # simulate price - ###############################REPLACED CODE TO USE FUNCTIONN AND SIMULATE PRICE WITH THE UPDATED FUNCTION AND CAP EXTREME PRICES
        priceCount = 0 ###############################ADDED FOR WHILE STATEMENT
        while priceCount <1:
            price = PriceEstimation(startPrice)
            if price > startPrice * 1.25 or price < startPrice * 0.7: # Throw out price changes that are above 99.5 percentile and below 0.5 percentile of history
                pass            
            else:
                priceCount = priceCount + 1    

        # calculate profit
        sold_2022 = min(demand_2022, prod_2022) #ADDED
        res = (sold_2022 * price - prod_2022 * 36000)  #changed to sold_2022 from demand_2022

        profit_2022.append(res)
        
        start_price_2023.append(price) #########################################ADDED.  Starting price for 2023 should be ending price for 2022

plt.hist(profit_2022, bins=100)
plt.title("Tesla profit by the end of 2022") 
plt.ylabel("Distribution Density")
plt.xlabel("Profit in US dollar")
plt.show()

ave = round(np.mean(profit_2022)/1000000000, 2)
print(f'The average estimated profit by the end of 2022 is {ave} billion')

# 68% confidence interval
profit_2022.sort()

index_low = int(len(profit_2022)*0.16)
low = profit_2022[index_low]/1000000000
index_high = int(len(profit_2022)*0.84)
high = profit_2022[index_high]/1000000000

print(f'the low profit value for 2022 is {low} bil') 
print(f'the high profit value for 2022 is {high} bil') 

###############################################################################
# profit - 2023
###############################################################################
profit_2023 = []
start_price_2024=[] #ADDED

for i in range (10000):
    

        # simulate supply
        sim = TeslaSupplySimulator(mu, std, production_cut[-1], 8) # 8 quarters
        prod_2023 = round(sim.SimulateOnce(), 0)
        # simulate demand
        # s = SimulateOnce(df["Demand_Total"], df_resid["Residuals"], 12, loc, scale)
        demand_2023 = s[1][0] # 2023
    
    
        # simulate price - ###############################REPLACED CODE TO USE FUNCTIONN AND SIMULATE PRICE WITH THE UPDATED FUNCTION AND CAP EXTREME PRICES
        priceCount = 0 ###############################ADDED FOR WHILE STATEMENT
        while priceCount <1:
            price = PriceEstimation(start_price_2023[i])
            if price > start_price_2023[i] * 1.25 or price < start_price_2023[i] * 0.7: # Throw out price changes that are above 99.5 percentile and below 0.5 percentile of history
                pass            
            else:
                priceCount = priceCount + 1      

        # calculate profit
        sold_2023 = min(demand_2023, prod_2023) #ADDED
        
        res = sold_2023 * price - prod_2023 * 36000 #Changed demand_2023 to sold_2023
        profit_2023.append(res)
        
        start_price_2024.append(price) #ADDED.  Include as start price in simulation of 2022

plt.hist(profit_2023, bins=100)
plt.title("Tesla profit by the end of 2023") 
plt.ylabel("Distribution Density")
plt.xlabel("Profit in US dollar")
plt.show()

ave = round(np.mean(profit_2023)/1000000000, 2)
print(f'The average estimated profit by the end of 2023 is {ave} billion')

# 68% confidence interval
profit_2023.sort()

index_low = int(len(profit_2023)*0.16)
low = profit_2023[index_low]/1000000000
index_high = int(len(profit_2023)*0.84)
high = profit_2023[index_high]/1000000000

print(f'the low profit value for 2023 is {low} bil') 
print(f'the high profit value for 2023 is {high} bil') 

###############################################################################
# profit - 2024
###############################################################################
profit_2024 = []

for i in range (10000):
    
        # simulate supply
        sim = TeslaSupplySimulator(mu, std, production_cut[-1], 12) # 12 quarters
        prod_2024 = round(sim.SimulateOnce(), 0)

        # simulate demand
        # s = SimulateOnce(df["Demand_Total"], df_resid["Residuals"], 12, loc, scale)
        demand_2024 = s[2][0] #2024

        # simulate price - ###############################REPLACED CODE TO USE FUNCTIONN AND SIMULATE PRICE WITH THE UPDATED FUNCTION AND CAP EXTREME PRICES
        priceCount = 0 ###############################ADDED FOR WHILE STATEMENT
        while priceCount <1:
            price = PriceEstimation(start_price_2024[i])
            if price > start_price_2024[i] * 1.25 or price < start_price_2024[i] * 0.7: # Throw out price changes that are above 99.5 percentile and below 0.5 percentile of history
                pass            
            else:
                priceCount = priceCount + 1     
    
        # calculate profit
        sold_2024 = min(demand_2024, prod_2024) #ADDED
        res = sold_2024 * price - prod_2024 * 36000 #changed demand_2024 to sold_2024

        profit_2024.append(res)
        
    
plt.hist(profit_2024, bins=100)
plt.title("Tesla profit by the end of 2024") 
plt.ylabel("Distribution Density")
plt.xlabel("Profit in US dollar")
plt.show()

ave = round(np.mean(profit_2024)/1000000000, 2)
print(f'The average estimated profit by the end of 2024 is {ave} billion')

# 68% confidence interval
profit_2024.sort()

index_low = int(len(profit_2024)*0.16)
low = profit_2024[index_low]/1000000000
index_high = int(len(profit_2024)*0.84)
high = profit_2024[index_high]/1000000000

print(f'the low profit value for 2024 is {low} bil') 
print(f'the high profit value for 2024 is {high} bil') 

###############################################################################
###############################################################################
# MC Simulation for profit (Correlation model)
###############################################################################
###############################################################################
import scipy.linalg as la

# historical data
production_cut = [145036, 179757, 180338, 206421, 237823, 305840]
demand_cut = [139593, 180667, 184877, 201304, 241391, 320416]

# this is a function used to calculate growth rate and the output is a list of growth rate
def growthRateLst(Lst):
    '''
    this is a function used to calculate growth rate and the output is a list of growth rate
    '''
    rateLst = []
    for i in range(len(Lst)-1):  
        curr = Lst[i+1]
        prev = Lst[i]
        rate = growthRate(curr, prev)
        rateLst.append(rate)

    return rateLst

# calculate growth rate for production and demand and change to numpy array
production_growthRateLst = np.array(growthRateLst(production_cut))
demand_growthRateLst = np.array(growthRateLst(demand_cut))

# check correlation
plt.scatter(production_growthRateLst, demand_growthRateLst)
plt.xlabel('production growth rate')
plt.ylabel('demand growth rate')
plt.title('correlation between production growth rate and demand growth rate')

# calculate the mean of growth rate for production and demand
production_growthRate_mu = np.mean(production_growthRateLst)
demand_growthRate_mu = np.mean(demand_growthRateLst)

# Covariance matrix
# Use Cholesky Decomposition
C = np.cov(production_growthRateLst, demand_growthRateLst)
np.shape(C)
L = la.cholesky(C, lower = True)
print("Lower Triangle, L")
print(L)
print("L * L^T")  #Just a heck on calculations above
print(L@L.transpose())
print() 

def CorrelatedGrowthRateGenerator(Pmu, Dmu, C):
    '''
    This is a function using Cholesky Decomposition to generate correlated growth rate
    '''
    m = (Pmu, Dmu)
    c = C
    production_GrowthRate, demand_GrowthRate = rand.multivariate_normal(m, c) # with an assumption that growth rate is a normal distribution
    
    return production_GrowthRate, demand_GrowthRate

# this is a test
p_rate, d_rate = CorrelatedGrowthRateGenerator(production_growthRate_mu, demand_growthRate_mu, C)

class TeslaProfitSimulator(mc.MonteCarlo):
    '''
    This is a function used to simulate the profit using correlated growth rate
    '''
    
    def __init__(self, p_p_growthRate_mu, p_d_growthRate_mu, p_C, p_curr_p, p_curr_d, p_quarter, p_curr_price):
    
        self.p_growthRate_mu = p_p_growthRate_mu # mean of production growth rate
        self.d_growthRate_mu= p_d_growthRate_mu # mean of demand growth rate
        self.C = p_C # convariance matrix
        self.curr_p = p_curr_p   # current production 
        self.curr_d = p_curr_d  # current demand
        self.curr_price = p_curr_price # current price
        self.quarter = p_quarter  # project in how many quarters


    def SimulateOnce(self):
        
        curr_p = self.curr_p
        curr_d = self.curr_d
        
        production_accu = []
        demand_accu = []
        
        for q in range(self.quarter):
            p_rate, d_rate = CorrelatedGrowthRateGenerator(self.p_growthRate_mu, self.d_growthRate_mu, self.C)
            
            curr_p = curr_p * (1 + p_rate)
            curr_d = curr_d * (1 + d_rate)
            
            production_accu.append(curr_p)
            demand_accu.append(curr_d)
            
        price = PriceEstimationOld(self.curr_price, self.quarter)
        
        yearly_prod = np.sum(production_accu[-4:])
        yearly_demand = np.sum(demand_accu[-4:])
        
        yearly_sold = min(yearly_prod, yearly_demand) #cap
        
        profit = yearly_sold * price - yearly_prod * 36000
        
        return profit

###############################################################################
# profit - 2022
###############################################################################
sim = TeslaProfitSimulator(production_growthRate_mu, demand_growthRate_mu, C, production_cut[-1], demand_cut[-1], 4, startPrice) # 4 quarters
res = sim.RunSimulation(simCount=10000)

plt.hist(sim.results, bins=100, facecolor='blue', alpha=0.5, density=True)
plt.title("Tesla profit by the end of 2022") 
plt.ylabel("Distribution Density")
plt.xlabel("Profit in US dollar")
plt.show()  

var = round(sim.var(.05)/1000000000, 2)

ave = round(np.mean(sim.results)/1000000000, 2)
minimum = round(min(sim.results)/1000000000, 2)
maximum = round(max(sim.results)/1000000000, 2)
print(f"Average profit: {ave} billion")
print(f"Average range for 10,000 trials is: [{minimum} billion, {maximum} billion]")
print(f'5% VaR is {var} billion')

# 68% confidencen interval
low = round(sim.var(.16)/1000000000, 2)
high = round(sim.var(.84)/1000000000, 2)
print(f'low profit value for 2022 is {low} bil')
print(f'high profit value for 2022 is {high} bil')

###############################################################################
# profit - 2023
###############################################################################
sim = TeslaProfitSimulator(production_growthRate_mu, demand_growthRate_mu, C, production_cut[-1], demand_cut[-1], 8, startPrice) # 8 quarters
res = sim.RunSimulation(simCount=10000)

plt.hist(sim.results, bins=100, facecolor='blue', alpha=0.5, density=True)
plt.title("Tesla profit by the end of 2023") 
plt.ylabel("Distribution Density")
plt.xlabel("Profit in US dollar")
plt.show()  

var = round(sim.var(.05)/1000000000, 2)

ave = round(np.mean(sim.results)/1000000000, 2)
minimum = round(min(sim.results)/1000000000, 2)
maximum = round(max(sim.results)/1000000000, 2)
print(f"Average profit: {ave} billion")
print(f"Average range for 10,000 trials is: [{minimum} billion, {maximum} billion]")
print(f'5% VaR is {var} billion')

# 68% confidencen interval
low = round(sim.var(.16)/1000000000, 2)
high = round(sim.var(.84)/1000000000, 2)
print(f'low profit value for 2023 is {low} bil')
print(f'high profit value for 2023 is {high} bil')

###############################################################################
# profit - 2024
###############################################################################
sim = TeslaProfitSimulator(production_growthRate_mu, demand_growthRate_mu, C, production_cut[-1], demand_cut[-1], 12, startPrice) # 12 quarters
res = sim.RunSimulation(simCount=10000)

plt.hist(sim.results, bins=100, facecolor='blue', alpha=0.5, density=True)
plt.title("Tesla profit by the end of 2024") 
plt.ylabel("Distribution Density")
plt.xlabel("Profit in US dollar")
plt.show()  

var = round(sim.var(.05)/1000000000, 2)

ave = round(np.mean(sim.results)/1000000000, 2)
minimum = round(min(sim.results)/1000000000, 2)
maximum = round(max(sim.results)/1000000000, 2)
print(f"Average profit: {ave} billion")
print(f"Average range for 10,000 trials is: [{minimum} billion, {maximum} billion]")
print(f'5% VaR is {var} billion')

# 68% confidencen interval
low = round(sim.var(.16)/1000000000, 2)
high = round(sim.var(.84)/1000000000, 2)
print(f'low profit value for 2024 is {low} bil')
print(f'high profit value for 2024 is {high} bil')