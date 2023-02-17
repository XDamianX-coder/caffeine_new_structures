## Load libraries
#Libraries import
import pandas as pd
from mordred import Calculator, descriptors
import mordred
import numpy as np
from rdkit import Chem

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn import linear_model


from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def prepare_data(file):
    
    df = pd.read_excel(file)
    
    try:
        mol_objs = [Chem.MolFromSmiles(smi) for smi in df['SMILES']]
    except:
        mol_objs = [Chem.MolFromSmiles(smi) for smi in df['new_SMILES']]
    
    calculate_descriptors = True
    
    if calculate_descriptors:
        calc = Calculator(descriptors, ignore_3D=True)
        molecular_descriptors = calc.pandas(mol_objs)
        molecular_descriptors = molecular_descriptors.applymap(is_morder_missing)
        molecular_descriptors = molecular_descriptors[sorted(molecular_descriptors.columns)]
    else:
        pass
    print("Data size (rows, columns): "+ str(molecular_descriptors.shape))
    
    simple_preprocessing = True
    if simple_preprocessing:
        molecular_descriptors_cleaned = molecular_descriptors.dropna(axis=1, how='any')
        molecular_descriptors_cleaned
    print("Data size after first reduction (rows, columns): "+ str(molecular_descriptors_cleaned.shape))
    molecular_descriptors_cleaned = molecular_descriptors_cleaned.loc[:, (molecular_descriptors_cleaned != 0).any(axis=0)]
    print("Data size after second reduction (rows, columns): "+ str(molecular_descriptors_cleaned.shape))
    
    try:
        molecular_descriptors_cleaned['Aktywność cytoprotekcyjna [%]'] = df['Aktywność cytoprotekcyjna [%]']
    except:
        print('There is issue with the target values...')
    
    
    return molecular_descriptors_cleaned

def is_morder_missing(x):
    return np.nan if type(x) == mordred.error.Missing or type(x) == mordred.error.Error else x 


def correlation_dataframe(molecular_descriptors_cleaned, correlation_threshold, verbose = False):
    
    if verbose:
        correlation_table = pd.DataFrame(data=molecular_descriptors_cleaned.columns.to_list(), 
                                         columns=["molecular descriptor name"])
        print(correlation_table.head())
        correlation_to_ak_cyt = []
        for mol_desc in correlation_table['molecular descriptor name']:
            x = np.corrcoef(np.array(molecular_descriptors_cleaned[mol_desc]), 
                            np.array(molecular_descriptors_cleaned['Aktywność cytoprotekcyjna [%]']))
            x = x.tolist()[0][1]
            correlation_to_ak_cyt.append(x)
        correlation_table['corr_value'] = correlation_to_ak_cyt
        print(correlation_table.head())
        correlation_table['absolute correlation value'] = [abs(x) for x in correlation_table['corr_value']]
        print(correlation_table[:-1].head())
    
        mol_desc_best_corr = correlation_table[correlation_table['absolute correlation value'] > correlation_threshold]
    
        print(mol_desc_best_corr.head())
        table_with_descriptors_to_be_used = mol_desc_best_corr[:-1]
        print(table_with_descriptors_to_be_used.head())
    else:
        correlation_table = pd.DataFrame(data=molecular_descriptors_cleaned.columns.to_list(), 
                                         columns=["molecular descriptor name"])
        
        correlation_to_ak_cyt = []
        for mol_desc in correlation_table['molecular descriptor name']:
            x = np.corrcoef(np.array(molecular_descriptors_cleaned[mol_desc]), 
                            np.array(molecular_descriptors_cleaned['Aktywność cytoprotekcyjna [%]']))
            x = x.tolist()[0][1]
            correlation_to_ak_cyt.append(x)
        correlation_table['corr_value'] = correlation_to_ak_cyt
        
        correlation_table['absolute correlation value'] = [abs(x) for x in correlation_table['corr_value']]
        
    
        mol_desc_best_corr = correlation_table[correlation_table['absolute correlation value'] > correlation_threshold]
    
        
        table_with_descriptors_to_be_used = mol_desc_best_corr[:-1]
        
    return table_with_descriptors_to_be_used
    
    

def test_data(molecular_descriptors_cleaned):
    
    test123 = molecular_descriptors_cleaned.loc[(molecular_descriptors_cleaned['Aktywność cytoprotekcyjna [%]'] == 90)
                                                | (molecular_descriptors_cleaned['Aktywność cytoprotekcyjna [%]'] == 70)
                                        | (molecular_descriptors_cleaned['Aktywność cytoprotekcyjna [%]'] == 45)
                                       | (molecular_descriptors_cleaned['Aktywność cytoprotekcyjna [%]'] == 10)
                                               | (molecular_descriptors_cleaned['Aktywność cytoprotekcyjna [%]'] == 0)]
    
    test_data = test123.iloc[[0, 1, 2]] #0, 1, 2, 5, 15 #It allows to get 5 different points of known activity
    
    return test_data
    
    


def prepare_model(data, features, model_type, test_data, n_estimators_ = 2, max_depth = 2, kernel_ = 'linear', gamma_ = 'auto', train_test_split_ = False, verbose = False):
    

    if verbose:
        if model_type == 'RandomForestRegressor':
            model = RandomForestRegressor(random_state=15, n_estimators=n_estimators_)
            print("The model used is: RandomForest...")
        elif model_type == 'DecisionTreeRegressor':
            model = DecisionTreeRegressor(random_state=15, max_depth=max_depth)
            print("The model used is: DecisionTree...")
        elif model_type == 'KNeighborsRegressor':
            model = KNeighborsRegressor()
            print("The model used is: KNeighbors...")
        elif model_type == 'SVR':
            model = SVR(gamma = gamma_, kernel=kernel_)
            print("The model used is: SVR...")
        elif model_type == 'linear_model':
            model = linear_model.LinearRegression()
            print("The model used is: LinearReg...")
        else:
            model = linear_model.LinearRegression()
            print("The model used is: Linear...")
    else:
        if model_type == 'RandomForestRegressor':
            model = RandomForestRegressor(random_state=15, n_estimators=n_estimators_)
            
        elif model_type == 'DecisionTreeRegressor':
            model = DecisionTreeRegressor(random_state=15, max_depth=max_depth)
            
        elif model_type == 'KNeighborsRegressor':
            model = KNeighborsRegressor()
           
        elif model_type == 'SVR':
            model = SVR(gamma=gamma_, kernel=kernel_)
            
        elif model_type == 'linear_model':
            model = linear_model.LinearRegression()
            
        else:
            model = linear_model.LinearRegression()
           
            

    if verbose:
        if train_test_split_:
            X_train, X_test, y_train, y_test = train_test_split(data[features['molecular descriptor name']], 
                                                        data['Aktywność cytoprotekcyjna [%]'], 
                                                        test_size=0.05, random_state=42) #test_size = 0.1
            model.fit(X_train, y_train)
            try:
                print("Return the coefficient of determination of the prediction: ")
                print(model.score(X_test, y_test))
            except:
                pass

            pred = model.predict(X_train)
            print("R^2 score: "+ str(r2_score(y_train, pred)))
            sqrt_r2 = np.sqrt(r2_score(y_train, pred))
            training_data_r2 = r2_score(y_train, pred)
            print('Correlation coefficient: '+ str(sqrt_r2))
            print("Test data - unseen during training:")
            pred = model.predict(X_test)
            print("R^2 score: "+ str(r2_score(pred, y_test)))
            sqrt_r2 = np.sqrt(r2_score(pred, y_test))
            print('Correlation coefficient: '+ str(sqrt_r2))
            print(pred)
            print(y_test) 
            test_data_r2 = r2_score(pred, y_test)

        else:
            X = data[features['molecular descriptor name']]
    
            y = data['Aktywność cytoprotekcyjna [%]']
    
    
            model.fit(X, y)
            print("Return the coefficient of determination of the prediction: ")
            print(model.score(test_data[features['molecular descriptor name']], test_data['Aktywność cytoprotekcyjna [%]']))
        
            pred = model.predict(X)
            print("R^2 score: "+ str(r2_score(y, pred)))
            sqrt_r2 = np.sqrt(r2_score(y, pred))
            training_data_r2 = r2_score(y, pred)
            print('Correlation coefficient: '+ str(sqrt_r2))
            print("Test data - unseen during training:")
            pred = model.predict(test_data[features['molecular descriptor name']])
            print("R^2 score: "+ str(r2_score(pred, test_data['Aktywność cytoprotekcyjna [%]'])))
            sqrt_r2 = np.sqrt(r2_score(pred, test_data['Aktywność cytoprotekcyjna [%]']))
            print('Correlation coefficient: '+ str(sqrt_r2))
            print(pred)
            print(test_data['Aktywność cytoprotekcyjna [%]']) 
            test_data_r2 = r2_score(pred, test_data['Aktywność cytoprotekcyjna [%]'])

    else:
        if train_test_split_:
            X_train, X_test, y_train, y_test = train_test_split(data[features['molecular descriptor name']], 
                                                        data['Aktywność cytoprotekcyjna [%]'], 
                                                        test_size=0.07, random_state=42)
            
            model.fit(X_train, y_train)
            
            pred = model.predict(X_train)
            sqrt_r2 = np.sqrt(r2_score(y_train, pred))
            training_data_r2 = r2_score(y_train, pred)
            pred = model.predict(X_test)
            sqrt_r2 = np.sqrt(r2_score(pred, y_test))
            
            test_data_r2 = r2_score(pred, y_test)
        else:
            X = data[features['molecular descriptor name']]

            y = data['Aktywność cytoprotekcyjna [%]']
    
    
            model.fit(X, y)
            
            pred = model.predict(X)
            
            sqrt_r2 = np.sqrt(r2_score(y, pred))
            training_data_r2 = r2_score(y, pred)
            pred = model.predict(test_data[features['molecular descriptor name']])
            sqrt_r2 = np.sqrt(r2_score(pred, test_data['Aktywność cytoprotekcyjna [%]']))
            test_data_r2 = r2_score(pred, test_data['Aktywność cytoprotekcyjna [%]'])
    

    return model, training_data_r2, test_data_r2
    

def data_standarization(dataframe):
    
    dataframe_ = dataframe.drop(['Aktywność cytoprotekcyjna [%]'], axis=1)
    
    to_be_returned = (dataframe_ - dataframe_.mean()) / dataframe_.std()
    to_be_returned['Aktywność cytoprotekcyjna [%]'] = dataframe['Aktywność cytoprotekcyjna [%]']
    
    return to_be_returned


def prepare_data_and_create_model(molecular_descriptors_df, correlation_threshold, standarization, model_type, n_estimators_ = 12, max_depth = 2, kernel_ = 'linear', gamma_ = 'auto', train_test_split_ = False, verbose = False):
    
    if standarization == True:
        
        if verbose:
            print("I am doing standarization...")
        else:
            pass
        
        data_to_be_prepared = molecular_descriptors_df
        
        stand = data_standarization(data_to_be_prepared)
        
        corr = correlation_dataframe(stand, correlation_threshold, verbose)
        
        if train_test_split_:
            test_data_ = 'None'
            model, train_r2, test_r2 = prepare_model(data_to_be_prepared, corr, model_type, test_data_, n_estimators_, max_depth, kernel_, gamma_, train_test_split_, verbose)
        else:
            test_ = test_data(stand)

            data_to_be_prepared = stand.drop(test_.index.to_list(), axis=0)
        
            model, train_r2, test_r2 = prepare_model(data_to_be_prepared, corr, model_type, test_, n_estimators_, max_depth, kernel_, gamma_, train_test_split_, verbose)
        
    elif standarization == False:
        
        if verbose:
            print("I am not doing standarization...")
        else:
            pass
        
        data_to_be_prepared = molecular_descriptors_df
        
        corr = correlation_dataframe(data_to_be_prepared, correlation_threshold, verbose)
        
        if train_test_split_:
            test_data_ = 'None'
            model, train_r2, test_r2 = prepare_model(data_to_be_prepared, corr, model_type, test_data_, n_estimators_, max_depth, kernel_, gamma_, train_test_split_, verbose)
        else:
            test_ = test_data(data_to_be_prepared)

            data_to_be_prepared = data_to_be_prepared.drop(test_.index.to_list(), axis=0)
        
            model, train_r2, test_r2 = prepare_model(data_to_be_prepared, corr, model_type, test_, n_estimators_, max_depth, kernel_, gamma_, train_test_split_, verbose)
    else:
        print("Error...")
    
    return model, train_r2, test_r2, data_to_be_prepared, corr