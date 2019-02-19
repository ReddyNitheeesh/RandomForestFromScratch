# -*- coding: utf-8 -*-
"""
Decision tree from Scrach for Classification model
@author: Nitheesh Reddy 

"""

# =============================================================================
# #importing required packages
# =============================================================================
import pandas as pd
from random import randrange
from math import sqrt


# =============================================================================
# Converting string to interger
# =============================================================================
def convert_str_int(data,index):
    unique=set(data.iloc[:,index])
    lookUp={}
    for i,value in enumerate(unique):
        lookUp[value]=i
    for i in range(data.shape[0]):
        data.iloc[i,index]=lookUp[data.iloc[i,index]]
    return data
        
# =============================================================================
#Returns gini value used to split features and form tree             
# =============================================================================
def gini_index(groups,classValues):    
    total=0
    gini=0
    for group in groups:
        total+=len(group)
    
    for group in groups:
        individual=len(group)
        p1=0
        for clas in classValues:
            if len(group)==0:
                continue
            propotion=[g[-1] for g in group].count(clas)/len(group)
            p1+=propotion**2
        propotion=1-p1
        gini+=propotion*(individual/total)
    return gini
                
# =============================================================================
# method accepts three arguments @traindata,@no of features to consider for split
#@targetfeatureindex
# =============================================================================
def gini_split(data,n_features,targetFeatureIndex):
    if isinstance(data,pd.DataFrame):        
        unique=list(set(data.iloc[:,targetFeatureIndex]))
        b_index,b_value,b_score,b_groups=999,999,999,None
        features12=[]
        while(len(features12)<n_features):
            f=randrange(data.shape[1])
            if f not in features12:
                features12.append(f)
        for feature in features12:
            for i in range(data.shape[0]):
                value=data.iloc[i,feature]
                left=[]
                right=[]
                for j in range(data.shape[0]):
                    if i==j:
                        continue
                    if value<data.iloc[j,feature]:
                        left.append(data.iloc[j,:])
                    else:
                        right.append(data.iloc[j,:])
                b_groups=left,right
                gini=gini_index(b_groups,unique)
                if gini<b_score:
                    b_index,b_value,b_score,b_groups=feature,value,gini,b_groups
        return  {'index':b_index,'value':b_value,'groups':b_groups}
    else:
        unique=list(set(d[targetFeatureIndex]for d in data))
        #unique=list(set(data.iloc[:,targetFeatureIndex]))
        b_index,b_value,b_score,b_groups=999,999,999,None
        features12=[]
        while(len(features12)<n_features):
            f=randrange(len(data[0])-1)
            if f not in features12:
                features12.append(f)
        for feature in features12:
            for i,data0 in enumerate(data):
                value=float(data0[feature])
                left=[]
                right=[]
                for j,data1 in enumerate(data):
                    if i==j:
                        continue
                    if value<float(data1[feature]):
                        left.append(data1)
                    else:
                        right.append(data1)
                b_groups=left,right
                gini=gini_index(b_groups,unique)
                if gini<b_score:
                    b_index,b_value,b_score,b_groups=feature,value,gini,b_groups
        return  {'index':b_index,'value':b_value,'groups':b_groups}
             
# =============================================================================
# Returns max outcome for group
# =============================================================================
def to_terminal(group,targetFeatureIndex):
    #select a class value for a group of rows. 
	outcomes = [row[targetFeatureIndex] for row in group]
    #returns the most common output value in a list of rows.
	return max(set(outcomes),key=outcomes.count)

# =============================================================================
# This method perform continous recersive operation to find gini and split
# based on it
# =============================================================================
def split(node, max_depth, min_size, n_features, depth,targetFeatureIndex):
    #print("depth ",depth)
    left,right = node['groups']
    del(node['groups'])
	
    if not left or not right:
        node['left']=node['right']=to_terminal(left+right,targetFeatureIndex)
        return
    
    if depth>=max_depth:
        node['left'],node['right']=to_terminal(left,targetFeatureIndex),to_terminal(right,targetFeatureIndex)
        return
    
    if len(left)<=min_size:
        node['left']=to_terminal(left,targetFeatureIndex)
    else:
        node['left']=gini_split(left,n_features,targetFeatureIndex)
        split(node['left'], max_depth, min_size, n_features, depth+1,targetFeatureIndex)
    
    if len(right)<=min_size:
        node['right']=to_terminal(right,targetFeatureIndex)
    else:
        node['right']=gini_split(right,n_features,targetFeatureIndex)
        split(node['right'], max_depth, min_size, n_features, depth+1,targetFeatureIndex)
        

# =============================================================================
# Build a tree 
# =============================================================================
def tree_build(train, max_depth, min_size, n_features,targetFeatureIndex):
    root = gini_split(train, n_features,targetFeatureIndex)
    split(root,max_depth,min_size,n_features,1,targetFeatureIndex)
    return root
    
# =============================================================================
# Prediction on test data for individual sample
# =============================================================================
def predict(node,row):
    if row[node['index']]<node['value']:
        if isinstance(node['left'],dict):
            return predict(node['left'],row)
        else:
            #print(node['left'])
            return node['left']
    else:
        if isinstance(node['right'],dict):
            return predict(node['right'],row)
        else:
            #print(node['right'])
            return node['right']


# =============================================================================
# split data into train and test
# =============================================================================
def train_testsplit(data):    
    rows=data.shape[0]
    trainrows=int(rows*0.7)
    testrows=rows-trainrows
    
    trainIndexList=[]
    while(len(trainIndexList)<trainrows):
        index=randrange(rows)
        if index not in trainIndexList:
            trainIndexList.append(index)
    
    testIndexList=[]
    while(len(testIndexList)<testrows):
        index=randrange(rows)
        if index not in trainIndexList:
            if index not in testIndexList:
                testIndexList.append(index)
    
    return data.iloc[trainIndexList,:],data.iloc[testIndexList,:]

# =============================================================================
# Prediction on list of data
# =============================================================================
def predictionsOnTest(trainedModel,testdata):
    return [predict(trainedModel,testdata.iloc[i,:])for i in range(testdata.shape[0])]

# =============================================================================
# prediction on list of data by different trees and taking more occureance value
# =============================================================================
def predictionsOnTest_rf(trainedModels,testdata):
    predictions=[]
    for i in range(testdata.shape[0]):
        eachPrediction=[]
        for j in range(len(trainedModels)):
            p=predict(trainedModels[j],testdata.iloc[i,:])
            eachPrediction.append(p)
        predictions.append(max(set(eachPrediction),key=eachPrediction.count))
    return predictions
# =============================================================================
# To check accuracy of our trained model
# =============================================================================
def accuracy_fun(testdata,targetFeature,predictions):
    rightPrediction=0
    for i in range(testdata.shape[0]):
        if testdata.iloc[i,targetFeature]==predictions[i]:
            rightPrediction+=1
    return (rightPrediction/testdata.shape[0])*100

# =============================================================================
# split the data into samples
# =============================================================================
def subsample(data,totalSampleSize):
    requiredSampleSize=[]
    while len(requiredSampleSize)<totalSampleSize:
        requiredSampleSize.append(randrange(data.shape[0]))
    return data.iloc[requiredSampleSize,:]


def randomForest(noOfTrees,traindata, max_depth, min_size, n_features,targetFeatureIndex):
    sampleCount=(traindata.shape[0]/noOfTrees)
    trees=[tree_build(subsample(traindata,int(sampleCount)), max_depth, min_size, n_features,targetFeatureIndex) for tree in range(noOfTrees)]
    return trees

# =============================================================================
# main method for Random Tree
# =============================================================================
def main_rf():
    #reading data
    data=pd.read_csv('sonar.all-data.csv')    
    ##converting string to interger type
    convert_str_int(data,60)   
    n_features=int(sqrt(data.shape[1]))
    targetFeatureIndex=60
    max_depth=10
    min_size=1
    noOfTrees=5
    traindata,testdata=train_testsplit(data)    
    listmodel=randomForest(noOfTrees,traindata, max_depth, min_size, n_features,targetFeatureIndex)
    predictions=predictionsOnTest_rf(listmodel,testdata)
    print("Accuracy of model for test data" ,accuracy_fun(testdata,targetFeatureIndex,predictions))

 


# =============================================================================
# Execution starts from here
# =============================================================================
if __name__=='__main__':
    main_rf()
    







