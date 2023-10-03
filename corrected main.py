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


def graph(theDFo, theEQ, graphTitle):
    Small = theDFo[theDFo.Type == 0]
    Big = theDFo[theDFo.Type == 1]
    plt.scatter(Big['Weight'], Big['Cost'], color='r', marker= 'o', s = 5)
    plt.scatter(Small['Weight'], Small['Cost'], color='b', marker= 'x', s =5 )
    
    x = np.linspace(0, 1.1, 1000)
    b  = -theEQ[2]/theEQ[1]
    slope = -theEQ[0]/theEQ[1]
    
    y = slope*x+b
    
    #print("slope is:", slope)
    #print("b is", b)


    plt.plot(x, y, '-g', label="Regression Line")
    
    plt.axis([0, 1.1, 0, 1.1])
    plt.xlabel('Costs')
    plt.ylabel('Weights')
    plt.title(graphTitle, loc='center')
    plt.legend()
    
    plt.show()

def initalizeArray():
    Arry = []
    for x in range(0,3):
        Arry.append(random.uniform(-0.5, 0.5))
    return Arry

def net(arr1, arr2):
    net = 0
    for i in range (0,2):
        net = net + arr1[i]*arr2[i]
    net += arr1[2]
    return net

def HAF(trainingSet, p, epsilon, alpha):
    originalArray = initalizeArray()
    r = len(trainingSet)
    r = int(r * p)
    TE = 2000  
    iterations = 0  
    
    while TE > epsilon and iterations < ni:  
        TE = 0
        for x in range(0, r):
            pattern = trainingSet[x]
            pArray = [pattern['Cost'], pattern['Weight'], 1]
            net1 = net(originalArray, pArray)
            if net1 >= 0:
                fire = 1
            else:
                fire = 0
            delta = pattern['Type'] - fire
            for s in range(0, 2):
                originalArray[s] += delta * alpha * pArray[s]
            originalArray[2] += delta * alpha
            TE += abs(delta)
        iterations += 1  
    print(TE)
    return originalArray

def SAF(trainingSet, p, epsilon, alpha, gain):
    originalArray = initalizeArray()
    r = len(trainingSet)
    r = int(r * p)
    TE = 2000
    iterations = 0  
    
    while TE > epsilon and iterations < ni:  #
        TE = 0
        for x in range(0, r):
            pattern = trainingSet[x]
            pArray = [pattern['Cost'], pattern['Weight'], 1]
            net1 = net(originalArray, pArray)
            w = 1 / (1 + np.exp(-gain * net1))
            delta = pattern['Type'] - w
            for s in range(0, 2):
                originalArray[s] += delta * alpha * pArray[s]
            originalArray[2] += delta * alpha
            TE += abs(delta)
        iterations += 1 
    print(TE)
    return originalArray
            

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

Dfaaa = Dfan.to_dict('index')
Dfbbb = Dfbn.to_dict('index')
Dfccc = Dfcn.to_dict('index')

Dfa_75 = train_a_75.reset_index().to_dict('index')
Dfb_75 = train_b_75.reset_index().to_dict('index') 
Dfc_75 = train_c_75.reset_index().to_dict('index')

Dfa_25 = train_a_25.reset_index().to_dict('index')
Dfb_25 = train_b_25.reset_index().to_dict('index')
Dfc_25 = train_c_25.reset_index().to_dict('index')



ε_A = 0.00001
ε_B = 40
ε_C = 700
ni = 300 


weightsHAFA = HAF(Dfa_75, 0.75, ε_A, 0.5)
print(weightsHAFA)
graph(train_a_75, weightsHAFA, "75% training HAF Group A")

weightsHAFB = HAF(Dfb_75, 0.75, ε_B, .5)
print(weightsHAFB)
graph(train_b_75, weightsHAFB, "75% training HAF Group B")

weightsHAFC = HAF(Dfc_75, 0.75, ε_C, .1)
print(weightsHAFC)
graph(train_c_75, weightsHAFC, "75% training HAF Group C")

weightsHAFA = HAF(Dfa_25, 0.25, ε_A, 0.5)
print(weightsHAFA)
graph(train_a_25, weightsHAFA, "25% training HAF Group A")

weightsHAFB = HAF(Dfb_25, 0.25, ε_B, 0.5)
print(weightsHAFB)
graph(train_b_25, weightsHAFB, "25% training HAF Group B")

weightsHAFC = HAF(Dfc_25, 0.25, ε_C, 0.5)
print(weightsHAFC)
graph(train_c_25, weightsHAFC, "25% training HAF Group C")

weightSAFA = SAF(Dfa_75, .75, ε_A, .5, 5)
print(weightSAFA)
graph(train_a_75, weightSAFA, "75% training SAF Group A")

weightSAFB = SAF(Dfb_75, .75, ε_B, .5, 5)
print(weightSAFB)
graph(train_b_75, weightSAFB, "75% training SAF Group B")

weightSAFC = SAF(Dfc_75, .75, ε_C, .5, 5)
print(weightSAFC)
graph(train_c_75, weightSAFA, "75% training SAF Group C")

weightSAFA = SAF(Dfa_25, .25, ε_A, .5, 5)
print(weightSAFA)
graph(train_a_25, weightSAFA, "25% training SAF Group A")

weightSAFB = SAF(Dfb_25, .25, ε_B, .5, 5)
print(weightSAFB)
graph(train_b_25, weightSAFB, "25% training SAF Group B")

weightSAFC = SAF(Dfc_25, .25, ε_C, .5, 5)
print(weightSAFC)
graph(train_c_25, weightSAFC, "25% training SAF Group C")

def calculate_inequal(num,kk):
   weighted_sum = num['Cost'] * kk[0] + num['Weight'] * kk[1] + kk[2]
   if weighted_sum > 0:
       return 1
   else:
       return 0
   
def testingFunction(df_train,weights, stringTitle):
    
    df_testPerceptrn=df_train.apply(calculate_inequal,axis=1,kk=(weights))
    actual_matrix=df_train['Type']
    
    predicted_matrix=df_testPerceptrn
    
    confusion_matrix=pd.crosstab(actual_matrix,predicted_matrix)
    
    df = normalize(df_train)
    graph(df,weights,stringTitle)
    
    print(confusion_matrix)
  
    return


testingFunction(test_a_25,weightsHAFA, "HAF Test on 25% of A")
testingFunction(test_b_25,weightsHAFB, "HAF Test on 25% of B")
testingFunction(test_c_25,weightsHAFC, "HAF Test on 25% of C")


testingFunction(test_a_75,weightsHAFA, "HAF Test on 75% of A")
testingFunction(test_b_75,weightsHAFB, "HAF Test on 75% of B")
testingFunction(test_c_75,weightsHAFC, "HAF Test on 75% of C")

testingFunction(test_a_25,weightSAFA, "SAF Test on 25% of A")
testingFunction(test_b_25,weightSAFB, "SAF Test on 25% of B")
testingFunction(test_c_25,weightSAFC, "SAF Test on 25% of C")


testingFunction(test_a_75,weightSAFA, "SAF Test on 75% of A")
testingFunction(test_b_75,weightSAFB, "SAF Test on 75% of B")
testingFunction(test_c_75,weightSAFC, "SAF Test on 75% of C")
