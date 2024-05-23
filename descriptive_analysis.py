import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import scipy.stats as stats 

#############################################
# CHANGE PATH !!!!
#############################################

path = 'C:/Users/chloe/OneDrive/Documents/DATS2M_1/MEMOIRE/Code/code officiel/data'

#############################################
# Importing datasets and data preprocessing
#############################################

data = pd.read_csv(path + "/train.csv", sep=',')
data2 = pd.read_csv(path + "/test.csv", sep=',')

#concatenate the two datasets
df = pd.concat([data, data2])

#drop the unnecessary columns
df.drop(['Unnamed: 0','id'],axis=1,inplace=True)

df.isnull().sum() #missing values

#fill in missing values with median value
df['Arrival Delay in Minutes'].fillna(value=df['Arrival Delay in Minutes'].median(axis=0),inplace=True)

#dummy for satisfaction
df['satisfaction'] = df['satisfaction'].map({'neutral or dissatisfied': 0, 'satisfied': 1})

#convert continuous variables into floats
for col in ['Arrival Delay in Minutes','Departure Delay in Minutes','Age','Flight Distance', 'satisfaction']:
    df[col]=df[col].astype('float64')
    
#convert categorical variables into categories
for col in ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking',
            'Gate location','Food and drink','Online boarding', 'Seat comfort', 'Inflight entertainment', 'On-board service', 'Leg room service',
            'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness']:
    df[col]=df[col].astype('category')

data = df

desc = data.describe() #descriptive statistics

#############################################
# Correlation matrix of continuous variables
#############################################

sns.set_style('darkgrid')
plt.figure(figsize=(14,5))
sns.heatmap(data.corr(),annot=True,cmap='viridis',annot_kws={"size":15})

#############################################
# Data transformation : delays
#############################################

#transform satisfaction to category
data['satisfaction'] = data['satisfaction'].astype("category")

#categorize Departure delay in Minutes and Arrival delay in Minutes
labels = ["0 - 5", "6 - 60", "61 - 120", "121 - 240", "240+"]
tmp = pd.cut(data['Departure Delay in Minutes'], [
             0, 6, 61, 121, 241, 1600], right=False, labels=labels)
data = data.assign(DepartDelayCat=tmp)


labels = ["0 - 5", "6 - 60", "61 - 120", "121 - 240", "241+"]
tmp = pd.cut(data['Arrival Delay in Minutes'], [
             0, 6, 61, 121, 241, 16000], right=False, labels=labels)
data = data.assign(ArrivalDelayCat=tmp)

#drop the unnecessary variables
data = data.drop(['Departure Delay in Minutes','Arrival Delay in Minutes'], axis=1)

#############################################
# Barplots in function of satisfaction part 1
#############################################

num_sub_plot=len(data.columns)
fig,ax=plt.subplots(4,3,figsize=(18,24))
col = data.columns
sns.countplot(data=data,x=col[0],hue='satisfaction',ax=ax[0,0], palette=['#2DA3CD',"#F8C302"])
sns.countplot(data=data,x=col[1],hue='satisfaction',ax=ax[0,1], palette=['#2DA3CD',"#F8C302"])
sns.histplot(data=data, x=col[2],hue='satisfaction', palette = ['#2DA3CD',"#F8C302"], ax=ax[0,2])
sns.countplot(data=data,x=col[3],hue='satisfaction',ax=ax[1,0], palette=['#2DA3CD',"#F8C302"])
sns.countplot(data=data,x=col[4],hue='satisfaction',ax=ax[1,1], palette=['#2DA3CD',"#F8C302"])
sns.histplot(data=data, x=col[5],hue='satisfaction', palette = ['#2DA3CD',"#F8C302"], ax=ax[1,2])
sns.countplot(data=data,x=col[6],hue='satisfaction',ax=ax[2,0], palette=['#2DA3CD',"#F8C302"])
sns.countplot(data=data,x=col[7],hue='satisfaction',ax=ax[2,1], palette=['#2DA3CD',"#F8C302"])
sns.countplot(data=data,x=col[8],hue='satisfaction',ax=ax[2,2], palette=['#2DA3CD',"#F8C302"])
sns.countplot(data=data,x=col[9],hue='satisfaction',ax=ax[3,0], palette=['#2DA3CD',"#F8C302"])
sns.countplot(data=data,x=col[10],hue='satisfaction',ax=ax[3,1], palette=['#2DA3CD',"#F8C302"])
sns.countplot(data=data,x=col[11],hue='satisfaction',ax=ax[3,2], palette=['#2DA3CD',"#F8C302"])

#############################################
# Barplots in function of satisfaction part 2
#############################################

num_sub_plot=len(data.columns)
fig,ax=plt.subplots(4,3,figsize=(18,24))
col = data.columns
sns.countplot(data=data,x=col[12],hue='satisfaction',ax=ax[0,0], palette=['#2DA3CD',"#F8C302"])
sns.countplot(data=data,x=col[13],hue='satisfaction',ax=ax[0,1], palette=['#2DA3CD',"#F8C302"])
sns.countplot(data=data,x=col[14],hue='satisfaction',ax=ax[0,2], palette=['#2DA3CD',"#F8C302"])
sns.countplot(data=data,x=col[15],hue='satisfaction',ax=ax[1,0], palette=['#2DA3CD',"#F8C302"])
sns.countplot(data=data,x=col[16],hue='satisfaction',ax=ax[1,1], palette=['#2DA3CD',"#F8C302"])
sns.countplot(data=data,x=col[17],hue='satisfaction',ax=ax[1,2], palette=['#2DA3CD',"#F8C302"])
sns.countplot(data=data,x=col[18],hue='satisfaction',ax=ax[2,0], palette=['#2DA3CD',"#F8C302"])
sns.countplot(data=data,x=col[19],hue='satisfaction',ax=ax[2,1], palette=['#2DA3CD',"#F8C302"])
sns.countplot(data=data,x=col[21],hue='satisfaction',ax=ax[2,2], palette=['#2DA3CD',"#F8C302"])
sns.countplot(data=data,x=col[22],hue='satisfaction',ax=ax[3,0], palette=['#2DA3CD',"#F8C302"])
sns.countplot(data=data,x=col[20],ax=ax[3,1], palette=['#2DA3CD',"#F8C302"])
ax[3,2].set_axis_off()

#############################################
# Barplots part 1
#############################################

num_sub_plot=len(data.columns)
fig,ax=plt.subplots(4,3,figsize=(18,24))
col = data.columns
sns.countplot(data=data,x=col[0],ax=ax[0,0], color= '#2DA3CD')
sns.countplot(data=data,x=col[1],ax=ax[0,1], color= '#2DA3CD')
sns.histplot(data=data, x=col[2], color = '#2DA3CD', ax=ax[0,2])
sns.countplot(data=data,x=col[3],ax=ax[1,0], color= '#2DA3CD')
sns.countplot(data=data,x=col[4],ax=ax[1,1], color= '#2DA3CD')
sns.histplot(data=data, x=col[5], color = '#2DA3CD', ax=ax[1,2])
sns.countplot(data=data,x=col[6],ax=ax[2,0], color= '#2DA3CD')
sns.countplot(data=data,x=col[7],ax=ax[2,1], color= '#2DA3CD')
sns.countplot(data=data,x=col[8],ax=ax[2,2], color= '#2DA3CD')
sns.countplot(data=data,x=col[9],ax=ax[3,0], color= '#2DA3CD')
sns.countplot(data=data,x=col[10],ax=ax[3,1], color= '#2DA3CD')
sns.countplot(data=data,x=col[11],ax=ax[3,2], color= '#2DA3CD')

#############################################
# Barplots part 2
#############################################

num_sub_plot=len(data.columns)
fig,ax=plt.subplots(4,3,figsize=(18,24))
col = data.columns
sns.countplot(data=data,x=col[12],ax=ax[0,0], color= '#2DA3CD')
sns.countplot(data=data,x=col[13],ax=ax[0,1], color= '#2DA3CD')
sns.countplot(data=data,x=col[14],ax=ax[0,2], color= '#2DA3CD')
sns.countplot(data=data,x=col[15],ax=ax[1,0], color= '#2DA3CD')
sns.countplot(data=data,x=col[16],ax=ax[1,1], color= '#2DA3CD')
sns.countplot(data=data,x=col[17],ax=ax[1,2], color= '#2DA3CD')
sns.countplot(data=data,x=col[18],ax=ax[2,0], color= '#2DA3CD')
sns.countplot(data=data,x=col[19],ax=ax[2,1], color= '#2DA3CD')
sns.countplot(data=data,x=col[21],ax=ax[2,2], color= '#2DA3CD')
sns.countplot(data=data,x=col[22],ax=ax[3,0], color= '#2DA3CD')
sns.countplot(data=data,x=col[20],ax=ax[3,1], color= '#2DA3CD')
ax[3,2].set_axis_off()

#############################################
# Data transformation : flight distance, age
#############################################

# Create categories for Age and Flight distance
labels = ["7 - 18", "19 - 30", "31 - 40", "41 - 50", "51 - 64", "65-85"]
tmp = pd.cut(data['Age'], [7, 19, 31, 41, 51, 65, 85], right=False, labels=labels)
data = data.assign(AgeCat=tmp)

labels = ["0 - 1000", "1001 - 2000", "2001 - 3000", "3001 - 4000", "4001 - 5000"]
tmp = pd.cut(data['Flight Distance'], [0, 1001, 2001, 3001, 4001, 5000], right=False, labels=labels)
data = data.assign(FlightDistanceCat=tmp)

#drop unnecessary variables
data = data.drop(['Age', 'Flight Distance'], axis=1)


##########################################################################################
# Cramer's V matrix ()
#code inspired from : https://www.kaggle.com/code/chrisbss1/cramer-s-v-correlation-matrix
##########################################################################################

#function for computing the cramer's V
def cramers_V(var1,var2) :
  crosstab =np.array(pd.crosstab(var1,var2, rownames=None, colnames=None)) # Cross table building
  stat = chi2_contingency(crosstab)[0] # Keeping of the test statistic of the Chi2 test
  obs = np.sum(crosstab) # Number of observations
  mini = min(crosstab.shape)-1 # Take the minimum value between the columns and the rows of the cross table
  return np.sqrt(stat/(obs*mini))

#drop continuous variables
data_bis = data.drop(['AgeCat', 'ArrivalDelayCat', 'DepartDelayCat', 'FlightDistanceCat'], axis = 1)

rows= []
for var1 in data_bis :
  col = []
  for var2 in data_bis :
    cramers =cramers_V(data_bis[var1], data_bis[var2]) # Cramer's V test
    col.append(round(cramers,2)) # Keeping of the rounded value of the Cramer's V  
  rows.append(col)
  
cramers_results = np.array(rows)
df = pd.DataFrame(cramers_results, columns = data_bis.columns, index = data_bis.columns)

mask = np.zeros_like(df, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

plt.figure(figsize=(14, 10))
with sns.axes_style("white"):
  ax = sns.heatmap(df, mask=mask, vmax=1, square=True, cmap='viridis',annot_kws={"size":10}, annot = True)
plt.show()

data_bis = np.array(data_bis)
X2 = stats.chi2_contingency(data_bis, correction=False)[0] 
N = np.sum(data_bis) 
minimum_dimension = min(data_bis.shape)-1
  
# Calculate Cramer's V 
result = np.sqrt((X2/N) / minimum_dimension) 
  
# Print the result 
print(result) 
