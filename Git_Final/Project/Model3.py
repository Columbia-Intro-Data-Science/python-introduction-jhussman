 # -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 13:47:30 2017

@author: Brian and Jamie
"""

import numpy as np;
#import seaborn as sns;
import pandas as pd
#from scipy import stats
import matplotlib.pyplot as plt
import random
#from random import *
import glob

#read multiple csvs into 1 dataframe
pathb =r'/Users/user/Documents/RESIDENTIAL_LOAD_DATA_E_PLUS_OUTPUT/BASE' 

def make_frame(path):
    allFiles = glob.glob(path + "/*.csv")
    frame = pd.DataFrame()
    list_ = []
    for file_ in allFiles:
        df = pd.read_csv(file_,index_col=0, header=0,usecols =['Date/Time','Electricity:Facility [kW](Hourly)'])
        list_.append(df)
    frame = pd.concat(list_,axis=1)
    frame = frame.iloc[:8760]
    return frame.dropna(axis=1)
    
#clean data:
#raw = pd.read_csv('/Users/user/python-introduction-jhussman/FinalProject/raw.csv',index_col='localminute')
#raw2 = pd.read_csv('/Users/user/python-introduction-jhussman/FinalProject/raw2.csv',index_col='localminute')
#raw3 = pd.read_csv('/Users/user/python-introduction-jhussman/FinalProject/raw3.csv',index_col='localminute')
#raw4 = pd.read_csv('/Users/user/python-introduction-jhussman/FinalProject/raw4.csv',index_col='localminute')

def Clean(df):
    grouped = df.groupby(['dataid'])
    l_grouped = list(grouped)
    num_dataid = df['dataid'].nunique()
    cleaned = l_grouped[0][1]
    num = cleaned['dataid'].iloc[0]
    cleaned = cleaned.rename(columns={'use': str(num)})
    cleaned = cleaned.drop('dataid',axis=1)
    for x in np.arange(1,num_dataid):
        bla = l_grouped[x][1]
        num = bla['dataid'].iloc[0]
        cleaned = pd.concat([cleaned, bla.drop('dataid',axis=1)], axis=1)
        cleaned = cleaned.rename(columns={'use': str(num)})
    return cleaned
    
#ultimate = pd.concat([Clean(raw),Clean(raw2),Clean(raw3),Clean(raw4),],axis=1) 
#ultimate = ultimate.drop(ultimate.index[len(ultimate)-1])
    
#Distance metric within data frame
def distance(ts,x,y):
    each = []
    each = np.absolute(ts.ix[:,x] - ts.ix[:,y])
    return np.sum(each)

#distance between different data frames
def sum_square(ts,ts2,x,y):
    each = []
    each = np.absolute(ts.ix[:,x] - ts2.ix[:,y])
    return (np.sum(each)) ** 2

#Initialize (forgy)
def initialize(df,k):
    maximum = len(df.columns)
    rand =  random.sample(range(0,maximum),k)
    return make_clusters(df,k,rand)

#Initialize (k-means++)
def k_init(df,k):
    maximum = len(df.columns)
    rand = []
    one = np.random.randint(maximum,size=1)
    rand.append(one[0])
    dist = []
    for y in range(maximum):
        dist.append(distance(df,y,rand[0])**2)
    rand.append(np.argmax(dist))   
    #rand.append(5)
    for x in range(k-2):
        dist = []
        for y in range(maximum): 
            dist.append(distance(df,y,rand[x])**2)
        total = np.sum(dist)
        dist = dist/total
        num = np.random.choice(np.arange(maximum),p=dist)
        bla = 1
        while bla == 1:
            if num in rand:
                num = np.random.choice(np.arange(maximum),p=dist);
            else: 
                rand.append(num); bla = 0
    return make_clusters(df,k,rand)

#k-means++ initialization
def k_means_pp(df,k):
    maximum = len(df.columns)
    rand = []
    one = np.random.randint(maximum,size=1)
    rand.append(one[0])
    dist = []
    for x in range(k-1):
        dist = []
        for y in range(maximum): 
            dist.append(distance(df,y,rand[x])**2)
        total = np.sum(dist)
        dist = dist/total
        num = np.random.choice(np.arange(maximum),p=dist)
        rand.append(num)
    return make_clusters(df,k,rand)
    
#Initialize (farthest means)
def k_far(df,k):
    maximum = len(df.columns)
    rand = []
    one = np.random.randint(maximum,size=1)
    rand.append(one[0])
    #rand.append(5)
    for x in range(k-1):
        dist = []
        for y in range(maximum):
            tots = []
            for z in range(len(rand)):
                tots.append([distance(df,y,rand[z])**2])
            dist = [ sum(l) for l in zip(*tots) ]
        rand.append(np.argmax(dist))
    return make_clusters(df,k,rand)
    
#Clustering
def make_clusters(df,k,mean):
    clusters = []
    for y in range(k): clusters.append([])
    for x in range(len(df.columns)):
        dist = []
        for y in range(k):
            dist.append(distance(df,x,mean[y]))
        clust_num = np.argmin(dist)
        clusters[clust_num].append(x)
            
    return clusters
     
    
#Return new mean
def mean(df,cluster):
    error = []
    for x in range(len(cluster)):
        err = 0
        for y in range(len(cluster)):
            err = err + distance(df,x,y) ** 2 
        error.append(err ** 0.5)
    return cluster[np.argmin(error)]    

#k-means function
def k_means(ts,k):
    clusters = k_init(ts,k)
    past_clusters, new_mean = [], []
    for y in range(k): past_clusters.append([])
    while past_clusters != clusters:
        past_clusters = clusters
        for x in range(k): new_mean.append(mean(ts,clusters[x][:]))
        clusters = make_clusters(ts,k,new_mean)
    return clusters
    
#Return curve_id list
def return_curve(ts,k):
    curve = k_means(ts,k)
    curve_id = []
    for x in range(len(ts.columns)):
        for y in range(len(curve)): 
            if x in curve[y]: curve_id.append(y)
    return curve_id
                
    
#Generate cluster curves
def cluster_curves(ts,k):
    clusters = k_means(ts,k)
    dfs = list()
    for x in range(k): dfs.append(ts[clusters[x]].mean(axis=1))
    mean_curves = pd.concat(dfs, axis=1)
    return mean_curves

#Generate cluster curves
def cluster_curves_k(ts,k):
    clusters = k_means(ts,k)
    dfs = list()
    for x in range(k): dfs.append(ts[clusters[x]].mean(axis=1))
    mean_curves = pd.concat(dfs, axis=1)
    return mean_curves
        
#Sum of squares
def RSS(ts,k,clusters,means):
    resid = []
    for x in range(k): 
        for y in clusters[x]: resid.append(sum_square(means,ts,x,y))
    return (np.sum(resid))** 0.5

def choose_min(ts,k):
    error_p = 200
    total_e = []
    for x in range(10):
        clusters = k_means(ts,k)
        dfs = list()
        for x in range(k): dfs.append(ts[clusters[x]].mean(axis=1))
        mean_curve = pd.concat(dfs, axis=1)
        error = RSS(ts,k,clusters,mean_curve)
        total_e.append(error)
        if error < error_p:
            meancurve_optimal = mean_curve
            clusters_optimal = clusters
        error_p = error
        print('Hi')
    return {'clusters':clusters_optimal, 'curves':meancurve_optimal ,'error':min(total_e)}
        
def elbow(ts):
    error = []
    K_s = [2, 3, 4, 5, 6, 7, 8,9,10,11,12]
    error = [324, 310.8, 294.5, 278.2, 269, 267.7, 262.3, 255, 251.4, 248, 241]
    #for k in K_s:
        #result = []
        #result = choose_min(ts,k)
        #error.append(result['error'])
    plt.plot(K_s,error)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('RSS')
    plt.savefig('/Users/user/python-introduction-jhussman/FinalProject/FlaskApp/templates/elbow2.png', format='png', dpi=1000)
    plt.show()

def optimize(ts):
    error = []
    k_s = np.arange(1,10)
    error_p = 1000000
    rateofchange_past = 0
    for k in k_s:
        result = []
        result = choose_min(ts,k)
        error = result['error']
        rateofchange = error_p - error
        if rateofchange < rateofchange_past:
            break
        rateofchange_past = rateofchange
        error_p = error
    return k
    
    
def plot_clusters(ts):
    plt.figure();
    colormap = ('green','r','m','c','b','y','k','orange') 
    ax = ts.plot(color=colormap)
    #ax.legend_.remove()
    ax.legend(('1','2','3','4','5','6','7','8'),loc='upper center', bbox_to_anchor=(0.5, 1.1),
    ncol=3, fancybox=True, shadow=True)
    ax.set_xlim(0,23)
    plt.title('Cluster Groups')
    m = [4,8,12,16,20]
    labels = ['04:00','08:00','12:00','16:00','20:00']
    plt.xticks(m,labels)
    plt.xlabel('Time')
    plt.ylabel('% of Daily Consumption')
    plt.savefig('/Users/user/python-introduction-jhussman/FinalProject/FlaskApp/templates/optimal.png', format='png', dpi=1000)
    plt.show()
    
def rand_select(df, n):
    return df.ix[:,np.random.choice(len(df.columns), n)]
    
def plot_rand_scaled(df, n):
    ts =  rand_select(df,n)
    for y in range(len(ts.columns)):
        ts.ix[:,y] = ts.ix[:,y] / sum(ts.ix[:,y]) * 100
    plt.figure(); 
    ax = ts.plot() #plt.legend(loc='best')
    ax.legend_.remove()
    ax.set_xlim(0,23)
    plt.title('Average Day')
    m = [4,8,12,16,20]
    labels = ['04:00','08:00','12:00','16:00','20:00']
    plt.xticks(m,labels)
    plt.xlabel('Time')
    plt.ylabel('% of Daily Consumption')
    #plt.savefig('/Users/user/python-introduction-jhussman/FinalProject/FlaskApp/templates/all.png', format='png', dpi=1000)
    plt.show()
    
def plot_rand(ts):
    #ts =  rand_select(df,n)
    plt.figure(); 
    ax = ts.plot() #plt.legend(loc='best')
    #ax.legend_.remove()
    ax.set_xlim(0,23)
    plt.title('Average Day')
    m = [4,8,12,16,20]
    labels = ['04:00','08:00','12:00','16:00','20:00']
    plt.xticks(m,labels)
    plt.xlabel('Time')
    plt.ylabel('% of Daily Consumption')
    #plt.savefig('/Users/user/python-introduction-jhussman/FinalProject/FlaskApp/templates/total.png', format='png', dpi=1000)
    plt.show()
            
def average_day(df):
    df = df.set_index(pd.date_range('1/1/2016', periods=8784, freq='H'))
    year_hour_means = df.groupby([df.index.year, df.index.hour]).mean()
    return year_hour_means
    
def scale_data(ts):
    for y in range(len(ts.columns)):
        ts.ix[:,y] = ts.ix[:,y] / sum(ts.ix[:,y]) * 100
    return ts
    
#function for initializing data base    
def curve_id_plot(ts,k,curve):
    #curve = k_means(ts,k)
    #return curve ids
    curve_id = []
    for x in range(len(ts.columns)):
        for y in range(len(curve)): 
            if x in curve[y]: curve_id.append(y+1)
    #make average curves
    dfs = list()
    for x in range(k): dfs.append(ts[curve[x]].mean(axis=1))
    #plot_clusters(pd.concat(dfs, axis=1))
    return curve_id
    
def curve_id(ts,k):
    curve = k_means(ts,k)
    #return curve ids
    curve_id = []
    for x in range(len(ts.columns)):
        for y in range(len(curve)): 
            if x in curve[y]: curve_id.append(y)
    return curve_id
   
def sum_data(ts):
    ts = ts.sum(axis=1)
    ts = ts / sum(ts) * 100
    return ts
    
    
#map_data = pd.read_csv('/Users/user/python-introduction-jhussman/FinalProject/Data/austin.csv', converters={'LON': str,'LAT': str}, usecols=['LON','LAT','NUMBER','STREET'])
#map_data = map_data.dropna(axis=0)
#map_data = map_data[map_data.NUMBER != 0]
#map_data['NUMBER'] = map_data['NUMBER'].astype(int)
#map_data['NUMBER'] = map_data['NUMBER'].astype(str)
#map_data['ADDRESS'] = map_data['NUMBER'] + ' ' + map_data['STREET']
#map_data = map_data.drop('NUMBER', 1)
#map_data = map_data.drop('STREET', 1)
#map_data = map_data.ix[600:1183,:]
#map_data['resident_id'] = list(ultimate.columns.values)
#map_data['curve_id'] = [random.randrange(0,3,1) for _ in range (363)]
#map_data.to_csv('/Users/user/python-introduction-jhussman/FinalProject/Data/cleaned_locations.csv')
  
