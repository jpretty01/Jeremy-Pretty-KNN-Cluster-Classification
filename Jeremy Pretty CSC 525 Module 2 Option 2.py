# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 22:31:26 2021

@author: jpret
"""

import pandas as pd
import numpy as np

age = 58
height = 70
weight = 125

def get_distance(p1,p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    distance = np.sqrt(np.sum((p1-p2)**2))
    return distance

def sort(test,matrix,k):
    #for each row of matrix
    #get distance between test data and each row of matrix
    #sort the distance
    #return the top k
    results = []
    for i,row in enumerate(matrix):
        distance = get_distance(test,row)
        results.append([distance,i])
        # print(distance)
    results.sort(key=lambda x:x[0])
    # print(results)
    indices = [x[1] for x in results]
    # print('Indices:',indices)
    return indices[:k]

def knn(test,matrix,k):
    labels = matrix[:,4]
    matrix = matrix[:,:4]
    indices = sort(test,matrix,k)
    labels = labels[indices]
    return labels

if __name__ == "__main__":
    #initialise the value of k
    k = 3
    df = pd.read_csv('https://gist.githubusercontent.com/dhar174/14177e1d874a33bfec565a07875b875a/raw/7aa9afaaacc71aa0e8bc60b38111c24e584c74d8/data.csv',names=['Age','Height','Weight','Gender','label'])
    
    #our test input
    test = [age,height,weight,1]

    matrix = df.values
    answers = knn(test,matrix,k)
    print('Predicted classes are:')
    print(answers)
    
