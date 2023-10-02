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
    maxweight = points['Weight'].max()
    points['Cost'] = points['Cost'].divide(maxcost)
    points['Weight'] = points['Weight'].divide(maxweight)
    return points


def graph(theDF, theDFo, theEQ, graphTitle):
    Small = theDFo[theDFo.Type == 0]
    Big = theDFo[theDFo.Type == 1]
    plt.scatter(Big['Weight'], Big['Cost'], color='r', s = 5, alpha=0.5)
    plt.scatter(Small['Weight'], Small['Cost'], color='b', s = 5, alpha=0.5)
    
    x = np.linspace(0, 20, 1000)
    slope = -theEQ[0]/theEQ[1]
    b = -theEQ[2]/theEQ[1]
    
    y = slope*x+b
   

    plt.plot(x, y, '-g', label="Regression Line")
    
    plt.axis([0.6, 1.1, .6, 1.1])
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
    r = r // 2
    r = round(r * p)
    TE = 2000  # Initialize total error
    while TE > epsilon:
        TE = 0
        for bigNum in range(0, ni):
            for x in range(0, r):
                pattern = trainingSet[x]
                pArray = [pattern['Cost'], pattern['Weight'], 1]
                net1 = net(originalArray, pArray)
                if net1 >= 0:
                    fire = 1
                else:
                    fire = 0
                delta = pArray[2] - fire
                for s in range(0, 2):
                    originalArray[s] += delta * alpha * pArray[s]
                originalArray[2] += delta * alpha
                TE += abs(delta)
    return originalArray

def SAF(trainingSet, p, epsilon, alpha, gain):
    originalArray = initalizeArray()
    r = len(trainingSet)
    r = r // 2
    r = round(r * p)
    TE = 2000
    while TE > epsilon:
        TE = 0
        for bigNum in range(0, ni):
            for x in range(0, r):
                pattern = trainingSet[x]
                pArray = [pattern['Cost'], pattern['Weight'], 1]
                net1 = net(originalArray, pArray)
                w = 1 / (1 + np.exp(-gain*net1))
                delta = pArray[2] - w
                for s in range(0, 2):
                    originalArray[s] += delta * alpha * pArray[s]
                originalArray[2] += delta * alpha
                TE += abs(delta)**2
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
train_a_75, test_a_75 = perDataset(Dfan, .75)
train_b_75, test_b_75= perDataset(Dfbn, .75)
train_c_75, test_c_75= perDataset(Dfcn, .75)

train_a_25, test_a_25= perDataset(Dfan, .25)
train_b_25, test_b_25= perDataset(Dfbn, .25)
train_c_25, test_c_25= perDataset(Dfcn, .25)

Dfaaa = Dfan.to_dict('index')
Dfbbb = Dfbn.to_dict('index')
Dfccc = Dfcn.to_dict('index')

Dfa_75 = train_a_75.reset_index().to_dict('index')
Dfb_75 = train_b_75.reset_index().to_dict('index') 
Dfc_75 = train_c_75.reset_index().to_dict('index')

Dfa_25 = train_a_25.reset_index().to_dict('index')
Dfb_25 = train_b_25.reset_index().to_dict('index')
Dfc_25 = train_c_25.reset_index().to_dict('index')


#print(Dfa75)
ε_A = 0.00001
ε_B = 70
ε_C = 700
ni = 5000 


weightsHAFA = HAF(Dfa_75, 0.75, ε_A, 0.5)
#print(weightsHAFA)
#graph(Dfa_75, train_a_75, weightsHAFA, "75% training haf")

weightsHAFB = HAF(Dfb_75, 0.75, ε_B, 0.5)
#print(weightsHAFB)

weightsHAFC = HAF(Dfc_75, 0.75, ε_C, 0.5)
#print(weightsHAFC)

weightsHAFA = HAF(Dfa_25, 0.25, ε_A, 0.5)
#print(weightsHAFA)

weightsHAFB = HAF(Dfb_25, 0.25, ε_B, 0.5)
#print(weightsHAFB)

weightsHAFC = HAF(Dfc_25, 0.25, ε_C, 0.5)
#print(weightsHAFC)


weightSAFA = SAF(Dfa_75, .75, ε_A, .5, 5)
#print(weightSAFA)

weightSAFB = SAF(Dfb_75, .75, ε_B, .5, 5)
#print(weightSAFB)

weightSAFC = SAF(Dfc_75, .75, ε_C, .5, 5)
#print(weightSAFC)

weightSAFA = SAF(Dfa_25, .25, ε_A, .5, 5)
#print(weightSAFA)

weightSAFB = SAF(Dfb_25, .25, ε_B, .5, 5)
#print(weightSAFB)

weightSAFC = SAF(Dfc_25, .25, ε_C, .5, 5)
#print(weightSAFC)

def calculate_inequal(num,ww):
   if (num['Cost']*ww[0]+ww[1]*num['Weight']+ww[2])>0:
       return 1
   else:
       return 0
   
def testingFunction(df_train,weights, stringTitle):
    
    df_test=df_train
#    print(df_test)
    df_testPerceptrn=df_test.apply(calculate_inequal,axis=1,ww=(weights))
#    print(df_testPerceptrn)
    actual_matrix=df_test['Type']
    
    predicted_matrix=df_testPerceptrn
    
    confusion_matrix=pd.crosstab(actual_matrix,predicted_matrix)
    
    df = normalize(df_test)
    graph(df_test,df,weights,stringTitle)
    
    print(confusion_matrix)
  
    return


testingFunction(test_a_75,weightsHAFA, "HAF Test on 25% of A")