import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, cross_validate
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

class MetricsCV():
    ''' Create CV metrics with hyperparameter tuning'''


    def __init__(self,pipe,pipe_name,X_train,y_train,rcv_now=False,obj=False):

        '''Args:
            rcv_now: bool 
                When 'True' rvc_metrics() method can proceed to provide results from previously defined RandomizedSearchCV are'''
        self.model = pipe
        self.name = pipe_name
        self.X_train = X_train
        self.y_train = y_train

        # For CV results
        self.cv_results= None
        self.cv_mean = None
        self.cv_train_mean = None
        self.cv_test_mean = None

        # For objective function
        self.space = space
        
        if rcv_now:
            self.rcv_metrics()
            self.test_metrics()
        if obj:
            self.objective()
    
    def objective(self,pipe,space,X_train=None,y_train=None,kfolds=5,return_train_score=True):
        
        space = self.space
        obj_pipe = pipe if pipe else self.model
        obj_X = X_train if X_train else self.X_train
        obj_y = y_train if y_train else self.y_train

        @use_named_args(self.space)
        def objective_function(**params):
            # Set the hyperparameters on the model
            self.model['model'].set_params(**params)
            
            self.cv_results = cross_validate(pipe, obj_X, obj_y, 
                                             scoring='neg_root_mean_squared_error', cv=kfolds, return_train_score=return_train_score)
            return objective_function, -np.mean(self.cv_results['test_score']), -np.mean(self.cv_results['train_score'])
        return objective_function
    
    def rcv_metrics(self,rcv_instance,X_train=None,y_train=None,val_df=None):

        rcv_X = X_train if X_train else self.X_train
        rcv_y = y_train if y_train else self.y_train

        self.rcv_instance = rcv_instance
        self.cv_results = self.rcv_instance.cv_results_
        self.cv_train_mean = -np.mean(self.cv_results['mean_train_score'])
        self.cv_test_mean = -np.mean(self.cv_results['mean_test_score'])
        
        best_estimator = self.rcv_instance.best_estimator_

        val_score_dict = {'Val Train Score': self.cv_train_mean,
                     'Val Test Score': self.cv_test_mean,
                     'Model Name': self.name}
        val_score_df = pd.DataFrame(score_dict,columns=['Model Name','Val Train Score',
                                                    'Val Test Score'], index=range(1))
        if val_df is None:
            pass
        else:
           val_score_df = pd.concat([val_df,val_score_df])
           val_score_df.index = range(len(val_score_df))
        return val_score_df, best_estimator
    
    
    
    def test_metrics(self,X_test,y_test,test_pipe=None,X_train=None, y_train=None,test_df=None):
        
        test_model = test_pipe if test_pipe else self.model
        X_train_full = X_train if X_train else self.X_train
        y_train_full = y_train if y_train else self.y_train
        self.X_test = X_test
        self.y_test = y_test
        
        train_score = mean_squared_error(y_train_full, test_model.predict(X_train_full), squared=False)
        test_score = mean_squared_error(self.y_test, test_model.predict(self.X_test), squared=False)
    
        test_score_dict = {'Model Name': self.name,  
                           'Train Score': train_score,
                           'Test Score': test_score}
        test_score_df = pd.DataFrame([test_score_dict]) 
    
        if test_df is not None:
            test_score_df = pd.concat([test_df, test_score_df], ignore_index=True)
            test_score_df.sort_values(by='Test Score', inplace=True)
    
        return test_score_df