# Original source of this code is from CPMP's Optimal Expected Submission Value
# https://www.kaggle.com/cpmpml/optimal-expected-submission-value/code
# I add my additional comment and result of each code execution onto his code

# We can focus on bag types because all bags have same capacity (50 pounds).
# There is finite number of bag types which are possible. 
# We define one random variables for each bag type.

# All we need is to estimate expected value and variance of each possible bag type. 
# Then we use two properties to find combination of bags,
# which maximizes combination of expected value and standard deviation:

# 1. Expected value of sum of random variables is sum of expected values of the random variables
# 1. Variance of sum of independent random variables is the sum of the variances of the random variable

# Kernels or scripts with similar approaches have been proposed by Dominic Breuker and Ben Gorman.
# The difference is that here we find optimal solution in probabilistic sense.

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

from subprocess import check_output

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

print(check_output(["ls", "../input"]).decode("utf8"))
# Any results you write to the current directory are saved as output.
# gifts.csv
# sample_submission.csv

# @
# First, you do some definitions.
# c gift_types: 9 gift types in list
gift_types = ['horse', 'ball', 'bike', 'train', 'coal', 'book', 'doll', 'blocks', 'gloves']

# c ngift_types: number of gift types
ngift_types = len(gift_types)
# print("ngift_types",ngift_types) 9

# print("range(ngift_types)",range(ngift_types))
# c range(ngift_types) range(0, 9)

horse, ball, bike, train, coal, book, doll, blocks, gloves = range(ngift_types)
# print("horse",horse) 0
# ...
# print("gloves",gloves) 9

# We will use Monte Carlo simulation.
# Let's agree on number of samples (10000) to use.
# Set it to higher value to get more accurate results.
# nsample=10000
nsample=100

# Let's look at bags composed of single gift type. 
# We use vectorized version of original numpy distributions.

def gift_weights(gift, ngift, n=nsample):
    """
    This method outputs weight of gift\n
    Args:
        gift():
        ngift():
        n=nsample():
    Returns:
        1. np.array([0.0]):
        if ngift == 0    
        1. 1D array containing 100 numbers for each gift type
    """

    # print("ngift",ngift)
    # 1
    # 2
    # ...
    # random end number like 19, 27
    
    # print("np.array([0.0])",np.array([0.0]))
    # np.array([0.0]) [0.]
    # np.array([0.0]) [0.]
    # ...
    # np.array([0.0]) [0.]
    # np.array([0.0]) [0.]

    
    if ngift == 0:
        return np.array([0.0])
    
    # print("gift",gift)
    # 0
    # 0
    # 0
    # 0
    # 0
    # 0
    # 0
    # 0
    # 0
    # 1
    # 1
    # 1

    np.random.seed(2016)
    if gift == horse:
        dist = np.maximum(0, np.random.normal(5,2,(n, ngift))).sum(axis=1)
        # print("dist",dist)
        # c dist::
        #  [37.80275326 31.59152792 29.44532569 18.70779218 34.13556584 28.27652369
        #   28.89533487 27.03911451 37.06759632 31.67836359 27.32651038 25.4913139
        #   30.66614669 24.6225446  28.7085631  25.71322699 29.03438805 30.84599603
        #   29.22088174 30.56358118 29.92727097 23.16210569 26.98499376 28.42839645
        #   25.66229893 29.22522215 29.11737032 29.86862687 34.79701929 25.79656646
        #   32.63287049 26.04114122 36.95664054 27.44112491 31.57264857 26.34379299
        #   28.26708985 31.43787495 24.34236301 31.02358905 24.95436919 27.37308398
        #   25.69026301 29.37001781 28.49361443 31.59540799 35.09101703 36.22718028
        #   27.31459002 30.36484927 27.36692203 22.69424749 35.67545769 31.64443058
        #   26.44683085 27.32586053 22.74489675 29.05866567 31.67320251 18.86649888
        #   16.92622093 32.15851237 37.60154665 31.78768873 25.8484756  19.98344576
        #   39.89012975 29.90868196 40.59958989 20.19279923 26.60867715 28.27647295
        #   30.59495175 26.57681642 32.08616465 33.16809191 26.48826667 26.28419461
        #   36.72348805 22.7427796  36.48299203 31.64535903 26.4860315  25.94226282
        #   28.22701676 23.57698765 33.43583942 28.42457466 28.69901528 24.87358803
        #   30.69617206 26.79088228 35.70516079 30.58322635 25.90022445 27.17525734
        #   35.2007798  29.80690676 33.54497265 35.51434625]
        
        # print("dist",np.array(dist).shape)
        # c dist::
        # (100,)
        # (100,)

    if gift == ball:
        dist = np.maximum(0, 1 + np.random.normal(1,0.3,(n, ngift))).sum(axis=1)
        # print("dist",np.array(dist).shape)
    if gift == bike:
        dist = np.maximum(0, np.random.normal(20,10,(n, ngift))).sum(axis=1)
        # print("dist",np.array(dist).shape)
    if gift == train:
        dist = np.maximum(0, np.random.normal(10,5,(n, ngift))).sum(axis=1)
        # print("dist",np.array(dist).shape)
    if gift == coal:
        dist = 47 * np.random.beta(0.5,0.5,(n, ngift)).sum(axis=1)
        # print("dist",np.array(dist).shape)
    if gift == book:
        dist = np.random.chisquare(2,(n, ngift)).sum(axis=1)
        # print("dist",np.array(dist).shape)
    if gift == doll:
        dist = np.random.gamma(5,1,(n, ngift)).sum(axis=1)
        # print("dist",np.array(dist).shape)
    if gift == blocks:
        dist = np.random.triangular(5,10,20,(n, ngift)).sum(axis=1)
        # print("dist",np.array(dist).shape)
    if gift == gloves:
        gloves1 = 3.0 + np.random.rand(n, ngift)
        gloves2 = np.random.rand(n, ngift)
        gloves3 = np.random.rand(n, ngift)
        dist = np.where(gloves2 < 0.3, gloves1, gloves3).sum(axis=1)
        # print("dist",np.array(dist).shape)
    return dist

# Let's find reasonable upper bound on number of gifts in bag. 
# For this, we compute expected score for bags with increasing number of toys,
# until score decreases. 
# Bag with largest score is determining maximum value. 
# This is fine when optimizing expected value, 
# as adding additional toys uses more toys without improving objective function.

epsilon = 1
max_type = np.zeros(ngift_types).astype('int')
# print("max_type",max_type)
# c max_type::
# [0 0 0 0 0 0 0 0 0]

for gift, gift_type in enumerate(gift_types):
    best_value = 0.0
    for j in range(1, 200):
        # c weights: 1D array holding 100 numbers (maybe weight?)
        weights = gift_weights(gift, j, nsample)
        raw_value = np.where(weights <= 50.0, weights, 0.0)
        value = raw_value.mean()
        if value > best_value:
            best_value = value
        else:
            break
    max_type[gift] = j
print("max_type",max_type)
# c max_type:: 
# [ 9 24  3  5  2 19  9  5 27]

# We can now look at more general bag types. 
# First we precompute weights of bags with single type. 
# Code is similar to above one.

# For each gift type, 
# we create 2D array with nsample rows, and ntype columns. 
# Column j contains weights of bag made of j+1 toys of given gift type.

def gift_distributions(gift, ngift, n=nsample):
    """
    This method finds distribution of gift\n
    Args:
        gift():
        ngift():
        n=nsample():
    Returns:
        2D array:
            row: number of sample
            column: number of gift type
    """
    if ngift == 0:
        return np.array([0.0])
    np.random.seed(2016)
    if gift == horse:
        dist = np.maximum(0, np.random.normal(5,2,(n, ngift)))
    if gift == ball:
        dist = np.maximum(0, 1 + np.random.normal(1,0.3,(n, ngift)))
    if gift == bike:
        dist = np.maximum(0, np.random.normal(20,10,(n, ngift)))
    if gift == train:
        dist = np.maximum(0, np.random.normal(10,5,(n, ngift)))
    if gift == coal:
        dist = 47 * np.random.beta(0.5,0.5,(n, ngift))
    if gift == book:
        dist = np.random.chisquare(2,(n, ngift))
    if gift == doll:
        dist = np.random.gamma(5,1,(n, ngift))
    if gift == blocks:
        dist = np.random.triangular(5,10,20,(n, ngift))
    if gift == gloves:
        gloves1 = 3.0 + np.random.rand(n, ngift)
        gloves2 = np.random.rand(n, ngift)
        gloves3 = np.random.rand(n, ngift)
        dist = np.where(gloves2 < 0.3, gloves1, gloves3)
    for j in range(1, ngift):
        dist[:,j] += dist[:,j-1]
    return dist

distributions = dict()

for gift in range(ngift_types):
    distributions[gift] = gift_distributions(gift, max_type[gift])

# We can now compute expected value of complex bags,
# with lookups of precomputed weight distributions. 
# With slight change of code,
# it's easy to compute additional statistics like variance of weight.

def gift_distributions(gift, ngift):
    """
    This method finds distribution of gift\n
    Args:
        1. gift
        1. ngift
    Returns:
        1. 0 if ngift <= 0:
        1. 51 if ngift >= max_type[gift]:
        1. distributions[gift][:,ngift-1]
    """

    # print("gift",gift)
    # print("ngift",ngift)
    # gift 2
    # ngift 0
    # gift 3
    # ngift 1
    # gift 4
    # ngift 1
    # gift 5
    # ngift 1

    if ngift <= 0:
        return 0
    if ngift >= max_type[gift]:
        return 51
    # print("distributions[gift][:,ngift-1]",distributions[gift][:,ngift-1])
    return distributions[gift][:,ngift-1]

def gift_value(ntypes):
    """ 
    Args:
        1: ntypes
    Returns:
        1. weights.mean()
        1. weights.std()
    """
    weights = np.zeros(nsample)
    # print("np.array(weights).shape",np.array(weights).shape)
    # c weights:: 
    # (100,)
    # (100,)

    for gift in range(ngift_types):
        dist = gift_distributions(gift, ntypes[gift])
        weights += dist
    weights = np.where(weights <= 50.0, weights, 0.0)
    return weights.mean(), weights.std()

# We can now generate bag types. 
# Idea is to start with empty bag, and to add one item at time. 
# We do it until expected value of bag decreases. 
# When this happens then we can discard newly created bag, 
# as it uses more items and yields lower expected value. 
# We use queue and some dictionaries to keep track of what bag types are relevant.

# Once relevant bags are found we put all of them in dataframe. 
# We remove those with less than three elements.

# This takes time proportional to nsample. 
# With 10000 is takes less than minute. 
# Go grab coffee if you set nsample to larger value, say 100,000.

from collections import deque

def get_update_value(bag, bag_stats):
    """
    Args:
        1. bag
        1. bag_stats
    Returns:
        1. bag_mean
        1. bag_std
    """
    # print("bag",bag)
    # c bag::
    # (1, 1, 0, 1, 1, 1, 1, 0, 2)
    # (1, 1, 0, 1, 1, 0, 2, 0, 2)
    # (1, 1, 0, 1, 1, 0, 1, 1, 2)

    # print("bag_stats",bag_stats)
    # c bag_stats::
    # {(1, 1, 0, 0, 0, 0, 0, 1, 1): (19.96891479793928, 3.9623633822029367),
    #   ...,
    #  (1, 1, 0, 0, 0, 0, 0, 0, 2): (10.063929317573994, 2.8134336735371153)}

    if bag in bag_stats:
        bag_mean, bag_std = bag_stats[bag]
    else:
        bag_mean, bag_std = gift_value(bag)
        bag_stats[bag] = (bag_mean, bag_std)
    return bag_mean, bag_std

def gen_bags():
    """ 
    This method generates bags\n
    Returns:
        1. bags
    Member variables:
        bag_stats:
            dictionary for statistics of bag
        queued:
            dictionary object
        queue:
            deque object
        bags:
            list
        bag0
            tuple containing 9 zeros
        queue.append(bag0)
        queued[bag0] = True
        bag_stats[bag0] = (0,0)
        counter = 0    
    """
    bag_stats = dict()
    queued = dict()
    queue = deque()
    bags = []
    bag0 = (0,0,0,0,0,0,0,0,0)
    queue.append(bag0)
    queued[bag0] = True
    bag_stats[bag0] = (0,0)
    counter = 0
    try:
        while True:
            if counter % 1000 == 0:
                print(counter, end=' ')
            counter += 1
            bag = queue.popleft()
            bag_mean, bag_std = get_update_value(bag, bag_stats)
            bags.append(bag+(bag_mean, bag_std ))
            for gift in range(ngift_types):
                new_bag = list(bag)
                new_bag[gift] = 1 + bag[gift]
                new_bag = tuple(new_bag)
                if new_bag in queued:
                    continue
                new_bag_mean, new_bag_std = get_update_value(new_bag, bag_stats)
                if new_bag_mean > bag_mean:
                    queue.append(new_bag)
                    queued[new_bag] = True
    except:
        return bags

    
bags = gen_bags()
# print("bags",np.array(bags).shape)
# c bags::
# (40753, 11)

# print("np.array(bags)[:5,:]",np.array(bags)[:5,:])
# c bags::
# [[ 0.          0.          0.          0.          0.          0.     0.          0.          0.          0.          0.        ]
#  [ 1.          0.          0.          0.          0.          0.     0.          0.          0.          5.12348449  1.82744184]
#  [ 0.          1.          0.          0.          0.          0.     0.          0.          0.          2.0264457   0.28046397]
#  [ 0.          0.          1.          0.          0.          0.     0.          0.          0.         18.9285857   9.34741888]
#  [ 0.          0.          0.          1.          0.          0.     0.          0.          0.          9.26074177  5.00431703]]

nbags = len(bags)
# print("nbags",nbags)
# c nbags:: 40753

# print("gift_types+['mean','std']",gift_types+['mean','std'])
# c gift_types+['mean','std']::
# ['horse', 'ball', 'bike', 'train', 'coal', 'book', 'doll', 'blocks', 'gloves', 'mean', 'std']

bags = pd.DataFrame(columns=gift_types+['mean','std'],data=bags)
# print("bags.head()",bags.head())
# c bags.head()::
#    horse  ball  bike  train  coal  book  doll  blocks  gloves       mean        std  
# 0      0     0     0      0     0     0     0       0       0   0.000000   0.000000  
# 1      1     0     0      0     0     0     0       0       0   5.123484   1.827442  
# 2      0     1     0      0     0     0     0       0       0   2.026446   0.280464  
# 3      0     0     1      0     0     0     0       0       0  18.928586   9.347419  
# 4      0     0     0      1     0     0     0       0       0   9.260742   5.004317  

# print("bags['std']**2",bags['std']**2)
# c bags['std']**2::
# 0          0.000000
# 1          3.339544
# 2          0.078660
# ...    
# 40751    134.337231
# 40752    136.571148


bags['var']=bags['std']**2
# print("bags.head()",bags.head())
# c bags.head()
#    horse  ball  bike  train  coal  book  doll  blocks  gloves       mean       std        var
# 0      0     0     0      0     0     0     0       0       0   0.000000  0.000000   0.000000
# 1      1     0     0      0     0     0     0       0       0   5.123484  1.827442   3.339544
# 2      0     1     0      0     0     0     0       0       0   2.026446  0.280464   0.078660
# 3      0     0     1      0     0     0     0       0       0  18.928586  9.347419  87.374240
# 4      0     0     0      1     0     0     0       0       0   9.260742  5.004317  25.043189


# print("bags[gift_types]",bags[gift_types])
# c bags[gift_types]
#        horse  ball  bike  train  coal  book  doll  blocks  gloves
# 0          0     0     0      0     0     0     0       0       0
# 1          1     0     0      0     0     0     0       0       0
# ...      ...   ...   ...    ...   ...   ...   ...     ...     ...
# 40751      0     3     0      0     0     0     0       0      23
# 40752      0     2     0      0     0     1     0       0      23
# [40753 rows x 9 columns]


# print("bags[gift_types].sum(axis=1)",bags[gift_types].sum(axis=1))
# bags[gift_types].sum(axis=1)
# 0         0
# 1         1
# 2         1
# ..
# 40751    26
# 40752    26


# print("bags[gift_types].sum(axis=1) >= 3",bags[gift_types].sum(axis=1) >= 3)
# c bags[gift_types].sum(axis=1) >= 3::
# 0        False
# 1        False
#          ...  
# 40751     True
# 40752     True

# print("bags[bags[gift_types].sum(axis=1) >= 3]",bags[bags[gift_types].sum(axis=1) >= 3].head())
# c bags[bags[gift_types].sum(axis=1) >= 3]
# horse  ball  bike  train  coal  book  doll  blocks  gloves       mean             std         var
# 54      3     0     0      0     0     0     0       0       0  15.110189    3.292962   10.843601
# 55      2     1     0      0     0     0     0       0       0  12.206113    2.630779    6.920997
# 56      2     0     1      0     0     0     0       0       0  28.581106    9.549405   91.191127
# 57      2     0     0      1     0     0     0       0       0  19.440409    5.548013   30.780448
# 58      2     0     0      0     1     0     0       0       0  20.599628   15.806595  249.848440

bags=bags[bags[gift_types].sum(axis=1) >= 3].reset_index(drop=True)

# print("bags.head()",bags.head())
# c bags.head()::    
# horse  ball  bike  train  coal  book  doll  blocks  gloves       mean           std         var
# 0      3     0     0      0     0     0     0       0       0  15.110189   3.292962   10.843601
# 1      2     1     0      0     0     0     0       0       0  12.206113   2.630779    6.920997
# 2      2     0     1      0     0     0     0       0       0  28.581106   9.549405   91.191127
# 3      2     0     0      1     0     0     0       0       0  19.440409   5.548013   30.780448
# 4      2     0     0      0     1     0     0       0       0  20.599628  15.806595  249.848440

# print("bags.shape[0]",bags.shape[0])
# c bags.shape[0]:: 40699


# Let's look at available gifts. 
# We will one hot encode gift type.
# c gifts: 7165 gifts, gift types are 9, dataframe
gifts = pd.read_csv('../input/gifts.csv')
# print("gifts",gifts)
# c gifts::
#          GiftId
# 0        horse_0
# 1        horse_1
# 2        horse_2
# ...
# 7163  gloves_197
# 7164  gloves_198
# 7165  gloves_199

for gift in gift_types:
    # print("1.0 * gifts['GiftId']",1.0 * gifts['GiftId'])
    # print("gifts['GiftId'].str.startswith(gift)",gifts['GiftId'].str.startswith(gift))
    # c gifts['GiftId'].str.startswith(gift)::
    # 0       False
    # 1       False
    # 2       False
    # ...  
    # 7164    False
    # 7165    False

    gifts[gift] = 1.0 * gifts['GiftId'].str.startswith(gift)
    # print("gifts[gift]",gifts[gift])
    # gifts[gift] 
    # 0       0.0
    # 1       0.0
    # 2       0.0
    # ... 
    # 7164    1.0
    # 7165    1.0
    # Name: gloves, Length: 7166, dtype: float64

# print("gifts.head()",gifts.head())
# c gifts.head()::
#     GiftId  horse  ball  bike  train  coal  book  doll  blocks  gloves
# 0  horse_0    1.0   0.0   0.0    0.0   0.0   0.0   0.0     0.0     0.0
# 1  horse_1    1.0   0.0   0.0    0.0   0.0   0.0   0.0     0.0     0.0
# 2  horse_2    1.0   0.0   0.0    0.0   0.0   0.0   0.0     0.0     0.0
# 3  horse_3    1.0   0.0   0.0    0.0   0.0   0.0   0.0     0.0     0.0
# 4  horse_4    1.0   0.0   0.0    0.0   0.0   0.0   0.0     0.0     0.0


# Number of gift of each type is easy to get.
allgifts = gifts[gift_types].sum()
# print("allgifts",allgifts)
# c allgifts::
# horse     1000.0
# ball      1100.0
# bike       500.0
# train     1000.0
# coal       166.0
# book      1200.0
# doll      1000.0
# blocks    1000.0
# gloves     200.0
