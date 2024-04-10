import pandas as pd
import numpy as np
import random
from sklearn.model_selection import RandomizedSearchCV, cross_validate
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

class MetricsCV():
    ''' Create CV metrics with hyperparameter tuning'''

    def __init__(self,pipe,pipe_name,X_train,y_train,random_state=None):

        self.model = pipe
        self.name = pipe_name
        self.X_train = X_train
        self.y_train = y_train
        self.random_state = random_state

    
    def objective(self,space,scoring, n_calls=50, n_initial_points=10, verbose=None, pipe=None,
                  X_train=None,y_train=None,kfolds=5,return_train_score=True, val_df=None):

        # Ensuring that the pipe and data used are either passed to the method or the instance defaults
        obj_pipe = pipe if pipe else self.model
        obj_X = X_train if X_train else self.X_train
        obj_y = y_train if y_train else self.y_train

        @use_named_args(space)
        def objective_function(**params):
            # Set the hyperparameters on the model
            self.model['model'].set_params(**params)
            
            cv_results = cross_validate(obj_pipe, obj_X, obj_y, 
                                             scoring=scoring, cv=kfolds, return_train_score=return_train_score)
            mean_test_score = -np.mean(cv_results['test_score'])
            if return_train_score:
                mean_train_score = -np.mean(cv_results['train_score'])
            else:
                mean_train_score = None
            return mean_test_score  # We minimize this value

        # Running Bayesian optimization
        res_gp = gp_minimize(objective_function, space, n_calls=n_calls, random_state=self.random_state, 
                             verbose=verbose, n_initial_points=n_initial_points)
        
        param_names = [dim.name for dim in space]  # Names like 'alpha'
        param_values = res_gp.x  # Best values found for these parameters
        param_dict = dict(zip(param_names, param_values))

        # Set parameters directly on the 'model' which is Lasso here
        obj_pipe['model'].set_params(**param_dict)
        
        # Refit the entire pipeline to update transformations with the new model settings
        obj_pipe.fit(obj_X, obj_y)  
        
        self.best_estimator_ = obj_pipe  # Pipeline as the best estimator
        
        best_train_score = -res_gp.fun if return_train_score else 'Not evaluated'
        best_test_score = -res_gp.fun

        # Prepare the validation score dictionary and DataFrame
        val_score_dict = {'Model Name': self.name,
                          'Val Train Score': -best_train_score,
                          'Val Test Score': -best_test_score}

        new_val_df = pd.DataFrame([val_score_dict])
        
        if val_df is not None:
            new_val_df = pd.concat([val_df, new_val_df])
            new_val_df.index = range(len(new_val_df))

        return new_val_df, self.best_estimator_


    def run_rcv(self,params,scoring,verbose=None, pipe=None, X_train=None, y_train=None, n_iter=10, kfolds=5, return_train_score=True):

        rcv_pipe = pipe if pipe else self.model 
        rcv_X = X_train if X_train else self.X_train
        rcv_y = y_train if y_train else self.y_train

        self.rcv_instance = RandomizedSearchCV(rcv_pipe,param_distributions=params,scoring=scoring,
                        return_train_score=return_train_score,n_iter=n_iter,verbose=verbose,cv=kfolds)
        
        self.rcv_instance.fit(rcv_X,rcv_y)
        
        return self.rcv_instance.best_params_
    
    def rcv_metrics(self,rcv_instance=None,X_train=None,y_train=None,val_df=None):
        
        rcv_X = X_train if X_train else self.X_train
        rcv_y = y_train if y_train else self.y_train
        rcv_instance = rcv_instance if rcv_instance is not None else self.rcv_instance


        self.cv_train_mean = -np.mean(rcv_instance.cv_results_['mean_train_score'])
        self.cv_test_mean = -np.mean(rcv_instance.cv_results_['mean_test_score'])

        val_score_dict = {'Model Name': self.name,
                          'Val Train Score': self.cv_train_mean,
                          'Val Test Score': self.cv_test_mean}
        new_val_df = pd.DataFrame([val_score_dict]) 
        
        if val_df is not None:
            new_val_df = pd.concat([val_df, new_val_df])
            new_val_df.index = range(len(new_val_df))
        
        return new_val_df, rcv_instance.best_estimator_
    
    
    def test_metrics(self,X_test,y_test,test_pipe=None,X_train=None, y_train=None,test_df=None):
        
        test_model = test_pipe if test_pipe else self.best_estimator_
        X_train_full = X_train if X_train else self.X_train
        y_train_full = y_train if y_train else self.y_train
        self.X_test = X_test
        self.y_test = y_test
        
        rmse_train = mean_squared_error(y_train_full, test_model.predict(X_train_full), squared=False)
        rmse_test = mean_squared_error(self.y_test, test_model.predict(self.X_test), squared=False)
        r2_train = r2_score(y_train_full,test_model.predict(X_train_full))
        r2_test = r2_score(self.y_test, test_model.predict(self.X_test))
        mae_train = mean_absolute_error(y_train_full,test_model.predict(X_train_full))
        mae_test = mean_absolute_error(self.y_test, test_model.predict(self.X_test))
    
        test_score_dict = {'Model Name':self.name,
                           'Train Score': rmse_train,
                           'Test Score': rmse_test,
                           'Train R2': r2_train,
                           'Test R2': r2_test,
                           'Train MAE': mae_train,
                           'Test MAE': mae_test}
        
        new_test_df = pd.DataFrame([test_score_dict]) 
    
        if test_df is not None:
            new_test_df = pd.concat([test_df, new_test_df], ignore_index=True)
            new_test_df.sort_values(by='Test Score', inplace=True)
    
        return new_test_df