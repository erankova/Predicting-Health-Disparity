import pandas as pd
import numpy as np
import random
from sklearn.model_selection import RandomizedSearchCV, cross_validate
from skopt import gp_minimize
from skopt.callbacks import DeltaXStopper
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

class MetricsCV():
    """
    Manages cross-validation metrics with hyperparameter tuning options including Bayesian Optimization or RandomizedSearchCV.
    
    Attributes:
        model (BaseEstimator): The pipeline including preprocessing steps. The model within the pipeline must be named 'model' for it to be
        called on within the class.
        name (str): Human-readable name for the model, used in reporting.
        X_train (pd.DataFrame): Training feature dataset.
        y_train (pd.Series): Training target dataset.
        random_state (Optional[int]): Random state to ensure reproducibility.
    
    Methods:
        objective(space, scoring, n_calls, n_initial_points, verbose, pipe, X_train, y_train, kfolds, return_train_score, val_df):
            Conducts Bayesian optimization and returns a DataFrame of validation scores along with the best estimator.
        
        run_rcv(params, scoring, verbose, pipe, X_train, y_train, n_iter, kfolds, return_train_score):
            Executes a RandomizedSearchCV to find the best parameters and returns them.
        
        rcv_metrics(rcv_instance, X_train, y_train, val_df):
            Computes and returns cross-validation scores from a RandomizedSearchCV instance as a DataFrame.
        
        test_metrics(X_test, y_test, test_pipe, X_train, y_train, test_df):
            Evaluates the model on the test set and returns test metrics including RMSE, MAE, and R2 scores as a DataFrame.
    """

    def __init__(self,pipe,pipe_name,X_train,y_train,random_state=None):

        
        """
        Initializes the MetricsCV instance with the pipeline, its name, training data, and an optional random state.
        """

        self.model = pipe
        self.name = pipe_name
        self.X_train = X_train
        self.y_train = y_train
        self.random_state = random_state

    def cross_val(self,scoring,kfolds=5, return_train_score=True, pipe=None,pipe_name=None,
                  X_train=None,y_train=None,val_df=None,verbose=None):

        """Cross validate model or pipeline without hyperparameter tuning.
        
        If validation dataframe was provided as an argument, new
        validation dataframe is merged.
        """
        
        cv_pipe = pipe if pipe else self.model
        cv_pipe_name = pipe_name if pipe_name else self.name
        cv_X = X_train if X_train else self.X_train
        cv_y = y_train if y_train else self.y_train

        cv_results = cross_validate(cv_pipe, cv_X, cv_y, scoring=scoring, cv=kfolds, return_train_score=return_train_score,verbose=verbose)

        self.cv_train_mean = -np.mean(cv_results['train_score'])
        self.cv_test_mean = -np.mean(cv_results['test_score'])

        val_score_dict = {'Model Name': self.name,
                          'Val Train Score': self.cv_train_mean,
                          'Val Test Score': self.cv_test_mean}
        new_val_df = pd.DataFrame([val_score_dict]) 
        
        if val_df is not None:
            new_val_df = pd.concat([val_df, new_val_df])
            new_val_df.index = range(len(new_val_df))
        
        return new_val_df
        
    
    def objective(self,space,scoring, n_calls=50, n_initial_points=10, verbose=None, pipe=None,
                  X_train=None,y_train=None,kfolds=5,return_train_score=True, val_df=None, 
                  n_points=None,n_jobs=None, dx_stopper=None):

        """
        Defines and executes Bayesian optimization to minimize the cross-validation loss and returns updated validation scores and the best
        estimator.
        
        If validation dataframe was provided as an argument, new
        validation dataframe is merged.
        """

        # Ensuring that the pipe and data used are either passed to the method or the instance defaults
        obj_pipe = pipe if pipe else self.model
        obj_X = X_train if X_train else self.X_train
        obj_y = y_train if y_train else self.y_train

        @use_named_args(space)
        def objective_function(**params):
            # Set the hyperparameters on the model
            self.model['model'].set_params(**params)
            
            cv_results = cross_validate(obj_pipe, obj_X, obj_y, 
                                             scoring=scoring, cv=kfolds, return_train_score=return_train_score,n_jobs=n_jobs)
            mean_test_score = -np.mean(cv_results['test_score'])
            if return_train_score:
                mean_train_score = -np.mean(cv_results['train_score'])
            else:
                mean_train_score = None
            return mean_test_score  # We minimize this value

        callbacks = []
        if dx_stopper is not None:
            # Stop optimization process if the change in x is smaller than dx_stopper
            callbacks.append(DeltaXStopper(dx_stopper))

        # Running Bayesian optimization
        res_gp = gp_minimize(objective_function, space, n_calls=n_calls, random_state=self.random_state, 
                             verbose=verbose, n_initial_points=n_initial_points,n_points=n_points,callback=callbacks)

        # Get param names from dimentional search space
        param_names = [dim.name for dim in space]  
        # Best values found for these parameters
        param_values = res_gp.x  
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


    def run_rcv(self,params,scoring,verbose=None, pipe=None, X_train=None, y_train=None, n_jobs=None, 
                n_iter=10, kfolds=5, return_train_score=True):

        """
        Executes RandomizedSearchCV to find the best model parameters.
        """

        rcv_pipe = pipe if pipe else self.model 
        rcv_X = X_train if X_train else self.X_train
        rcv_y = y_train if y_train else self.y_train

        self.rcv_instance = RandomizedSearchCV(rcv_pipe,param_distributions=params,scoring=scoring,
                        return_train_score=return_train_score,n_iter=n_iter,verbose=verbose,cv=kfolds, n_jobs=n_jobs)
        
        self.rcv_instance.fit(rcv_X,rcv_y)
        
        return self.rcv_instance.best_params_
    
    def rcv_metrics(self,rcv_instance=None,X_train=None,y_train=None,val_df=None):

        """
        Calculates and returns the cross-validation and training scores as a DataFrame. If validation dataframe was provided as
        an argument, new validation dataframe is merged
        """
        
        rcv_X = X_train if X_train else self.X_train
        rcv_y = y_train if y_train else self.y_train
        rcv_instance = rcv_instance if rcv_instance is not None else self.rcv_instance
        self.best_estimator_ = rcv_instance.best_estimator_


        self.cv_train_mean = -np.mean(rcv_instance.cv_results_['mean_train_score'])
        self.cv_test_mean = -np.mean(rcv_instance.cv_results_['mean_test_score'])

        val_score_dict = {'Model Name': self.name,
                          'Val Train Score': self.cv_train_mean,
                          'Val Test Score': self.cv_test_mean}
        new_val_df = pd.DataFrame([val_score_dict]) 
        
        if val_df is not None:
            new_val_df = pd.concat([val_df, new_val_df])
            new_val_df.index = range(len(new_val_df))
        
        return new_val_df, self.best_estimator_
    
    
    def test_metrics(self,X_test,y_test,test_pipe=None,X_train=None, y_train=None,test_df=None):

        """
        Evaluates the model on test data and returns various performance metrics (RMSE, MAE, R^2). If test dataframe was
        provided as an argument, new validation dataframe is merged.

        Note: if testing unfitted model or pipe, must include as argument.
        """
        
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