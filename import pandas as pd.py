import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import seaborn as sns


 # max number of iterations

#create datasets

def normalize(points):
    maxcost = points['Cost'].max()
    mincost = points['Cost'].min()
    maxweight = points['Weight'].max()
    minweight = points ['Weight'].min()
    points['Cost'] = round((points['Cost'] - mincost)/(maxcost - mincost),5)
    points['Weight'] = round((points['Weight'] - minweight)/(maxweight - minweight),5)
    return points


def graph(data, weights, graphTitle):
    Small = data[data['Type'] == 0]
    Big = data[data['Type'] == 1]
    plt.scatter(Big['Weight'], Big['Cost'], color='r', marker='o', s=5)
    plt.scatter(Small['Weight'], Small['Cost'], color='b', marker='x', s=5)
    
    x = np.linspace(0, 1.1, 1000)
    b  = -weights[2]/weights[1]
    slope = -weights[0]/weights[1]
    
    y = slope*x + b
    
    #print("slope is:", slope)
    #print("b is", b)


    plt.plot(x, y, '-g', label="Regression Line")
    
    plt.axis([0, 1.1, 0, 1.1])
    plt.xlabel('Costs')
    plt.ylabel('Weights')
    plt.title(graphTitle, loc='center')
    plt.legend()
    
    plt.show()

def initialize_array():
    return np.random.uniform(-0.5, 0.5, 3)

def net(arr1, arr2):
    return np.dot(arr1[:2], arr2[:2]) + arr1[2]

def HAF(training_set, p, epsilon, alpha, ni):
    weights_arr = initialize_array()
    r = int(len(training_set) * p)
    te = 0
    iterations = 0  
    
    while iterations <= ni:
        if(te >= epsilon):
            print(te)
            return weights_arr 
        for x in range(r):
            pattern = training_set.iloc[x]
            p_array = [pattern['Cost'], pattern['Weight'], 1]
            net1 = net(weights_arr, p_array)
            fire = 1 if net1 >= 0 else 0
            delta = pattern['Type'] - fire
            weights_arr[:2] += delta * alpha * np.array(p_array[:2])
            weights_arr[2] += delta * alpha
            te += abs(delta)
        iterations += 1  

    print(te)
    return weights_arr

def SAF(training_set, p, epsilon, alpha, gain, ni):
    weights_arr = initialize_array()
    r = int(len(training_set) * p)
    te = 0
    iterations = 0  
    
    while iterations <= ni: 
        if(te >= epsilon):
            print(te)
            return weights_arr
        for x in range(r):
            pattern = training_set.iloc[x]
            p_array = [pattern['Cost'], pattern['Weight'], 1]
            net1 = net(weights_arr, p_array)
            w = 1 / (1 + np.exp(-gain * net1))
            delta = pattern['Type'] - w
            weights_arr[:2] += delta * alpha * np.array(p_array[:2])
            weights_arr[2] += delta * alpha
            te += abs(delta)
        iterations += 1 

    print(te)
    return weights_arr


def perDataset(dataset, percent):
    train_amount = int(len(dataset) * percent)
    select_rows = np.random.choice(dataset.index, size=train_amount, replace=False)

    train = dataset.loc[select_rows]
    test = dataset.drop(index=select_rows)

    return train, test

Dfa = pd.read_csv('groupA.txt', header = None, names = ['Cost', 'Weight', 'Type'])
Dfb = pd.read_csv('groupB.txt', header = None, names = ['Cost', 'Weight', 'Type'])
Dfc = pd.read_csv('groupC.txt', header = None, names = ['Cost', 'Weight', 'Type'])

#normalize the data
Dfan = normalize(Dfa)
Dfbn= normalize(Dfb)
Dfcn = normalize(Dfc)

#create Traning and Testing Sets
train_a_75, test_a_25 = perDataset(Dfan, .75)
train_b_75, test_b_25= perDataset(Dfbn, .75)
train_c_75, test_c_25= perDataset(Dfcn, .75)

train_a_25, test_a_75= perDataset(Dfan, .25)
train_b_25, test_b_75= perDataset(Dfbn, .25)
train_c_25, test_c_75= perDataset(Dfcn, .25)


train_a_75_df = pd.DataFrame(train_a_75, columns=Dfan.columns)
train_b_75_df = pd.DataFrame(train_b_75, columns=Dfbn.columns)
train_c_75_df = pd.DataFrame(train_c_75, columns=Dfcn.columns)

train_a_25_df = pd.DataFrame(train_a_25, columns=Dfan.columns)
train_b_25_df = pd.DataFrame(train_b_25, columns=Dfbn.columns)
train_c_25_df = pd.DataFrame(train_c_25, columns=Dfcn.columns)


epsilon_A = 0.00001
epsilon_B = 40
epsilon_C = 700
ni = 5000


weightsHAFA_75 = HAF(train_a_75_df, 0.75, epsilon_A, 0.5, ni)
print(weightsHAFA_75)
graph(train_a_75_df, weightsHAFA_75, "75% training HAF Group A")

weightsHAFB_75 = HAF(train_b_75_df, 0.75, epsilon_B, 0.5, ni)
print(weightsHAFB_75)
graph(train_b_75_df, weightsHAFB_75, "75% training HAF Group B")

weightsHAFC_75 = HAF(train_c_75_df, 0.75, epsilon_C, 0.5, ni)
print(weightsHAFC_75)
graph(train_c_75_df, weightsHAFC_75, "75% training HAF Group C")


weightsHAFA_25 = HAF(train_a_25_df, 0.25, epsilon_A, 0.5, ni)
print(weightsHAFA_25)
graph(train_a_25_df, weightsHAFA_25, "25% training HAF Group A")

weightsHAFB_25 = HAF(train_b_25_df, 0.25, epsilon_B, 0.5, ni)
print(weightsHAFB_25)
graph(train_b_25_df, weightsHAFB_25, "25% training HAF Group B")

weightsHAFC_25 = HAF(train_c_25_df, 0.25, epsilon_C, 0.5, ni)
print(weightsHAFC_25)
graph(train_c_25_df, weightsHAFC_25, "25% training HAF Group C")



weightSAFA_75 = SAF(train_a_75_df, .75, epsilon_A, 0.5, 1, ni)
print(weightSAFA_75)
graph(train_a_75_df, weightSAFA_75, "75% training SAF Group A")

weightSAFB_75 = SAF(train_b_75_df, .75, epsilon_B, 0.5, 1, ni)
print(weightSAFB_75)
graph(train_b_75_df, weightSAFB_75, "75% training SAF Group B")

weightSAFC_75 = SAF(train_c_75_df, .75, epsilon_C, 0.5, 1, ni)
print(weightSAFC_75)
graph(train_c_75_df, weightSAFC_75, "75% training SAF Group C")


weightSAFA_25 = SAF(train_a_25_df, .25, epsilon_A, 0.5, 1, ni)
print(weightSAFA_25)
graph(train_a_25_df, weightSAFA_25, "25% training SAF Group A")

weightSAFB_25 = SAF(train_b_25_df, .25, epsilon_B, 0.5, 1, ni)
print(weightSAFB_25)
graph(train_b_25_df, weightSAFB_25, "25% training SAF Group B")

weightSAFC_25 = SAF(train_c_25_df, .25, epsilon_C, 0.5, 1, ni)
print(weightSAFC_25)
graph(train_c_25_df, weightSAFC_25, "25% training SAF Group C")


def calculate_inequal(num,kk):
   weighted_sum = num['Cost'] * kk[0] + num['Weight'] * kk[1] + kk[2]
   if weighted_sum > 0:
       return 1
   else:
       return 0
   
def testingFunction(df_test,weights, stringTitle):
    
    df_testPerceptrn=df_test.apply(calculate_inequal,axis=1,kk=(weights))
    actual_matrix=df_test['Type']
    
    predicted_matrix=df_testPerceptrn
    
    confusion_matrix=pd.crosstab(actual_matrix,predicted_matrix)
    
    graph(df_test, weights,stringTitle)
    
    print(confusion_matrix)
  
    return


testingFunction(test_a_25, weightsHAFA_75, "HAF Test (25%) on 75% training data of A")
testingFunction(test_b_25, weightsHAFB_75, "HAF Test (25%) on 75% training data of B")
testingFunction(test_c_25, weightsHAFC_75, "HAF Test (25%) on 75% training data of C")


testingFunction(test_a_75, weightsHAFA_25, "HAF Test (75%) on 25% of training data of A")
testingFunction(test_b_75, weightsHAFB_25, "HAF Test (75%) on 25% of training data of B")
testingFunction(test_c_75, weightsHAFC_25, "HAF Test (75%) on 25% of training data of C")

testingFunction(test_a_25, weightSAFA_75, "SAF Test (25%) on 75% training data of A")
testingFunction(test_b_25, weightSAFB_75, "SAF Test (25%) on 75% training data of B")
testingFunction(test_c_25, weightSAFC_75, "SAF Test (25%) on 75% training data of C")


testingFunction(test_a_75, weightSAFA_25, "SAF Test (75%) on 25% of training data of A")
testingFunction(test_b_75, weightSAFB_25, "SAF Test (75%) on 25% of training data of B")
testingFunction(test_c_75, weightSAFC_25, "SAF Test (75%) on 25% of training data of C")