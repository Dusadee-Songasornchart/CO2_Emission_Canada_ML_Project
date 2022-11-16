import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn import utils
from sklearn import model_selection
from sklearn import tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.svm import SVR
#from matplotlib.pylab import rcParams   
#import matplotlib as mpl
import warnings
warnings.filterwarnings('ignore')

class predict_CO2_emission:
  def __init__(self, Engine_size, Cylinders,Fuel_comb_L,Fuel_Comb_mpg,Fuel_Type):
    self.Engine_size = Engine_size
    self.Cylinders = Cylinders
    self.Fuel_comb_L = Fuel_comb_L
    self.Fuel_Comb_mpg = Fuel_Comb_mpg
    self.Fuel_Type = Fuel_Type
    self.finish = False

  def __str__(self):
    return f"Engine size : {self.Engine_size} , Cylinders : {self.Cylinders} , Fuel_comb_L : {self.Fuel_comb_L} , Fuel_comb_mpg : {self.Fuel_Comb_mpg} , Fuel_Type : {self.Fuel_Type}"

  def predict(self):
    df_CO2_Emission = pd.read_csv('CO2 Emissions_Canada.csv')
    df_CO2_Emission.drop_duplicates(inplace=True)
    df_CO2_Emission_wait = pd.DataFrame()
    df_CO2_Emission_check = pd.DataFrame()
    columns_cont = []
    for i in df_CO2_Emission.columns:
        temp = i
        if df_CO2_Emission[i].dtypes == 'object' or i == 'CO2 Emissions(g/km)':
            df_CO2_Emission_wait[temp] = df_CO2_Emission[i]
        else:
            columns_cont.append(i)
            df_CO2_Emission_check[temp] = df_CO2_Emission[i]
    df_CO2_Emission_wait.reset_index(inplace=True)
    df_CO2_Emission_wait.drop(columns='index',inplace=True)
    x_data_test = {'Engine Size(L)': [self.Engine_size], 'Cylinders': [self.Cylinders],'Fuel Consumption Comb (L/100 km)': [self.Fuel_comb_L]
    ,'Fuel Consumption Comb (mpg)': [self.Fuel_Comb_mpg]}
    df_x_test = pd.DataFrame(data=x_data_test)
    df_CO2_Emission_check = df_CO2_Emission_check.append(df_x_test)
    scale = preprocessing.StandardScaler()
    df_CO2_Emission_check = pd.DataFrame(scale.fit_transform(df_CO2_Emission_check),columns = columns_cont)
    df_use_predict = df_CO2_Emission_check.tail(1)
    df_CO2_Emission_check.drop(df_CO2_Emission_check.tail(1).index,inplace=True)
    df_use_predict.pop("Fuel Consumption City (L/100 km)")
    df_use_predict.pop("Fuel Consumption Hwy (L/100 km)")
    df_use_predict["Fuel Type_D"] = 0
    df_use_predict["Fuel Type_E"] = 0
    df_use_predict["Fuel Type_N"] = 0
    df_use_predict["Fuel Type_X"] = 0
    df_use_predict["Fuel Type_Z"] = 0
    Fuel = self.Fuel_Type
    for i in df_use_predict:
      if i == Fuel:
          df_use_predict[i] = 1
    
    def remove_outlier(df_in,col_name_x):
      q1_x = df_in[col_name_x].quantile(0.25)
      q3_x = df_in[col_name_x].quantile(0.75)
      iqr_x = q3_x-q1_x #Interquartile range
      fence_low_x  = q1_x-1.5*iqr_x
      fence_high_x = q3_x+1.5*iqr_x
      df_out = df_in.loc[((df_in[col_name_x] > fence_low_x) & (df_in[col_name_x] < fence_high_x))]
      return df_out

    for i in df_CO2_Emission_check.columns:
      df_CO2_Emission_check = remove_outlier(df_CO2_Emission_check,i)
    df_CO2_Emission_ver_2 = pd.concat([df_CO2_Emission_wait,df_CO2_Emission_check], axis=1)
    df_CO2_Emission_ver_2.dropna(inplace=True)
    df_CO2_Emission_ver_2.reset_index(inplace=True)
    df_CO2_Emission_ver_2.drop(columns='index',inplace=True)
    Y = df_CO2_Emission_ver_2['CO2 Emissions(g/km)']
    df_CO2_Emission_ver_2.drop(columns='CO2 Emissions(g/km)',inplace=True)
    df_CO2_Emission_cont = pd.DataFrame()
    df_CO2_Emission_cate = pd.DataFrame()
    columns_cont = []
    for i in df_CO2_Emission_ver_2.columns:
      temp = i
      if df_CO2_Emission_ver_2[i].dtypes == 'object' :
          df_CO2_Emission_cate[temp] = df_CO2_Emission_ver_2[i]
      else:
          columns_cont.append(i)
          df_CO2_Emission_cont[temp] = df_CO2_Emission_ver_2[i]
    data = df_CO2_Emission_cont.corr()
    lower = pd.DataFrame(np.tril(data, -1),columns = data.columns)
    to_drop = [column for column in lower if any(lower[column] > 0.95)]
    df_CO2_Emission_cont.drop(to_drop, inplace=True, axis=1)
    df_CO2_Emission_cate.drop(columns='Make',inplace= True)
    df_CO2_Emission_cate.drop(columns='Model',inplace= True)
    df_CO2_Emission_cate.drop(columns= 'Vehicle Class',inplace = True)
    df_CO2_Emission_cate.drop(columns= 'Transmission',inplace = True)
    df_CO2_Emission_cate_dum = pd.get_dummies(df_CO2_Emission_cate,columns=['Fuel Type'])
    X = pd.DataFrame()
    X = df_CO2_Emission_cont.join(df_CO2_Emission_cate_dum)
    rating_pctile = np.percentile( Y, [20,40,60,80])
    Y_list = []
    range_1 = 0
    range_2 = 0
    range_3 = 0
    range_4 = 0
    range_5 = 0
    for i in Y:
      if (i < rating_pctile[0]): 
          CO2_grade = 1
          Y_list.append(CO2_grade)
          range_1=range_1+1
      elif (rating_pctile[0] <= i < rating_pctile[1]): 
          CO2_grade = 2
          Y_list.append(CO2_grade)
          range_2=range_2+1 
      elif (rating_pctile[1] <= i < rating_pctile[2]): 
          CO2_grade = 3
          Y_list.append(CO2_grade)
          range_3=range_3+1 
      elif (rating_pctile[2] <= i < rating_pctile[3]): 
          CO2_grade = 4
          Y_list.append(CO2_grade)
          range_4=range_4+1 
      else:
          CO2_grade = 5
          Y_list.append(CO2_grade)
          range_5=range_5+1
    Y_0 = pd.DataFrame()
    Y_0['CO2_LEVEL'] = Y_list
    Y = Y_0['CO2_LEVEL']
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.3, random_state=42)
    # Create Model List
    classification = { 'KNN': KNeighborsClassifier(), 'DT': DecisionTreeClassifier(), 'RF': RandomForestClassifier(),'SVC' : SVC() }
    ASM_function = ['entropy', 'gini']
    maxD = [ 4, 5, 6, None]
    maxF = ['auto', 'log2', None]
    minSample = [1,2, 4]
    nEst = [10, 30, 50, 100]
    RF_param = dict(n_estimators= nEst, criterion=ASM_function, max_depth = maxD, min_samples_leaf = minSample,max_features = maxF)
    grid_RF = GridSearchCV( estimator = classification['RF'],n_jobs = 1,verbose = 10,scoring = 'accuracy', cv = 2,param_grid = RF_param )
    grid_result_RF = grid_RF.fit(X_train,Y_train)
    RandomF = RandomForestClassifier(criterion=grid_result_RF.best_params_['criterion'],
    max_depth = grid_result_RF.best_params_['max_depth'], 
    max_features = grid_result_RF.best_params_['max_features'],
    min_samples_leaf = grid_result_RF.best_params_['min_samples_leaf'],
    n_estimators = grid_result_RF.best_params_['n_estimators'])
    RandomF.fit(X_train,Y_train)
    y_pred= RandomF.predict(df_use_predict)
    self.finish = True
    return (y_pred)

#data_x = predict_CO2_emission(1.4,4,7.1,40,"Fuel Type_X")
#j = data_x.predict()
#print(j[0],j[1])


