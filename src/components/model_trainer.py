import os,sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestRegressor(random_state=42),
                "Extra Trees": ExtraTreesRegressor(random_state=42),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(random_state=42),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False, random_seed=42),
                "AdaBoost Regressor": AdaBoostRegressor(random_state=42),
            }
            param={
                "Decision Tree":{
                    'criterion':['squared_error','friedman_mse'],
                    'max_depth':[3,5,7,10,None],
                    'min_samples_split':[2,5,10]
                },
                "Random Forest":{
                    'n_estimators':[100,200,300],
                    'max_depth':[5,10,None],
                    'min_samples_split':[2,5]
                },
                "Gradient Boosting":{
                    'n_estimators':[100,200],
                    'learning_rate':[0.05,0.1,0.15],
                    'max_depth':[3,5,7]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[0.05,0.1,0.15],
                    'n_estimators':[100,200,300],
                    'max_depth':[3,5,7]
                },
                "CatBoosting Regressor":{
                    'depth':[4,6,8],
                    'learning_rate':[0.05,0.1,0.15],
                    'iterations':[100,200,300]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[0.05,0.1,0.15],
                    'n_estimators':[50,100,200]
                },
                "Extra Trees":{
                    'n_estimators':[100,200],
                    'max_depth':[5,10,None],
                    'min_samples_split':[2,5]
                }
            }
            model_report:dict=evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=param
                )
            
            best_model_score=max(sorted(model_report.values()))

            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
                ]
            

            best_model=models[best_model_name]  
            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted=best_model.predict(X_test)
            r2_scores=r2_score(y_test,predicted)
            return r2_scores

        except Exception as e:
            raise CustomException(e,sys)