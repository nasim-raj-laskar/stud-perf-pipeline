import os,sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(X_train,y_train,X_test,y_test,models,param):
    try:
        report={}
        model_names = list(models.keys())
        total_models = len(model_names)
        
        print(f"\n Starting hyperparameter tuning for {total_models} models...")
        print("=" * 60)
        
        for i in range(total_models):
            model_name = model_names[i]
            model = list(models.values())[i]
            para = param[model_name]
            
            print(f"\n[{i+1}/{total_models}] Tuning {model_name}...")
            
            if para:  # Only show param count if there are parameters
                param_combinations = 1
                for param_values in para.values():
                    param_combinations *= len(param_values)
                print(f" Testing {param_combinations} parameter combinations with 5-fold CV")
            else:
                print(f" No hyperparameters to tune")
            
            gs = GridSearchCV(model, para, cv=5, verbose=0, scoring='r2', n_jobs=-1)
            gs.fit(X_train, y_train)
            
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            
            y_test_pred = model.predict(X_test)
            y_train_pred = model.predict(X_train)
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            
            print(f"  Best params: {gs.best_params_}")
            print(f"  Test R² Score: {test_model_score:.4f}")
            
            report[model_name] = test_model_score
            
        print("\n" + "=" * 60)
        print(" Hyperparameter tuning completed!")
        return report
    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)