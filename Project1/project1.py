import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#normalize the data from 0 to 1
def normalizeData (points):
    maxcost = points['cost'].max()
    maxweight = points['weight'].max()
    points['cost'] = points['cost'].divide(maxcost)
    points['weight'] = points['weight'].divide(maxweight)
    return points;

# read the txt files as a csv file and add the coloum names
columns = ['cost', 'weight', 'type']

# Read data from CSV-formatted text files for sets A, B, and C
data_A_df = pd.read_csv(r'C:\Users\Stephen\OneDrive\Documents\Classes\Fall 2023\CMSC 409\Project1_Data_F23\groupA.txt', delimiter=',', names=columns)
data_B_df = pd.read_csv(r'C:\Users\Stephen\OneDrive\Documents\Classes\Fall 2023\CMSC 409\Project1_Data_F23\groupB.txt', delimiter=',', names=columns)
data_C_df = pd.read_csv(r'C:\Users\Stephen\OneDrive\Documents\Classes\Fall 2023\CMSC 409\Project1_Data_F23\groupC.txt', delimiter=',', names=columns)

data_A_df = normalizeData(data_A_df)
data_B_df = normalizeData(data_B_df)
data_C_df = normalizeData(data_C_df)

#create dataframe for predicted vs actual values for group a
confusion_df = pd.DataFrame(columns=['actual', 'predicted'])
#use line to predict on data and populate datafram
for index, row in data_A_df.iterrows():
    prediction = 0
    if(0.98*row['cost'] + row['weight'] - 1.7 > 0):
        prediction = 1
    confusion_df.loc[len(confusion_df.index)] = [row['type'], prediction]
#create confusion matrix from dataframe
confusion_matrix = pd.crosstab(confusion_df['actual'], confusion_df['predicted'], rownames=['Actual'], colnames=['Predicted'])
print("Group A confusion matrix")
print (confusion_matrix)
print()

#repeat for group b
confusion_df = pd.DataFrame(columns=['actual', 'predicted'])
for index, row in data_B_df.iterrows():
    prediction = 0
    if(1.01*row['cost'] + row['weight'] - 1.835 > 0):
        prediction = 1
    confusion_df.loc[len(confusion_df.index)] = [row['type'], prediction]
confusion_matrix = pd.crosstab(confusion_df['actual'], confusion_df['predicted'], rownames=['Actual'], colnames=['Predicted'])
print("Group B confusion matrix")
print (confusion_matrix)
print()

#repeat for group c
confusion_df = pd.DataFrame(columns=['actual', 'predicted'])
for index, row in data_C_df.iterrows():
    prediction = 0
    if(1.21*row['cost'] + row['weight'] - 2.01 > 0):
        prediction = 1
    confusion_df.loc[len(confusion_df.index)] = [row['type'], prediction]
confusion_matrix = pd.crosstab(confusion_df['actual'], confusion_df['predicted'], rownames=['Actual'], colnames=['Predicted'])
print("Group C confusion matrix")
print (confusion_matrix)
print()

# Create scatter plots for sets A, B, and C using Seaborn
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.scatterplot(data=data_A_df, x='cost', y='weight', hue='type', palette={0: 'red', 1: 'blue'})
x1, y1 = [0.7, 1], [1.104, 0.72]
plt.plot(x1, y1, marker = 'o')
plt.title("Group A")
plt.xlabel("Cost (USD)")
plt.ylabel("Weight (lbs)")
plt.legend(title="Car Type", labels=["Big", "Small"])

plt.subplot(1, 3, 2)
sns.scatterplot(data=data_B_df, x='cost', y='weight', hue='type', palette={0: 'red', 1: 'blue'})
x2, y2 = [0.85, 1], [0.9765, 0.825]
plt.plot(x2, y2, marker = 'o')
plt.title("Group B")
plt.xlabel("Cost (USD)")
plt.ylabel("Weight (lbs)")
plt.legend(title="Car Type", labels=["Big", "Small"])
 
plt.subplot(1, 3, 3)
sns.scatterplot(data=data_C_df, x='cost', y='weight', hue='type', palette={0: 'red', 1: 'blue'})
x2, y2 = [0.75, 1], [1.1025, 0.8]
plt.plot(x2, y2, marker = 'o')
plt.title("Group C")
plt.xlabel("Cost (USD)")
plt.ylabel("Weight (lbs)")
plt.legend(title="Car Type", labels=["Big", "Small"])

plt.tight_layout()
plt.show()
