import pandas as pd
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LassoCV

### functions definitions
def RF(X_1, X_2, d_2):
    '''
    Performs Random Forest Classification for predicting propensity scores.
    
    Parameters
    ----------
    X_1 : array-like
        covariates from the subsample for which estimation of propensity scores is desired.
    X_2 : array-like
        covariates from the auxuliary subsample.
    d_2 : list or one-dimensional array
        treatment values from the auxuliary subsample.

    Returns
    -------
    Estimated propensity scores for X_1.

    '''
    
    rfc = RandomForestClassifier(1000, max_depth = 20)
    rfc.fit(X_2, d_2)  
    prop_score = rfc.predict_proba(X_1)
    
    return(prop_score)

def LOGLASSO(X_1, X_2, d_2):
    '''
    Performs Logistic LASSO for predicting propensity scores.

    Parameters
    ----------
    X_1 : array-like
        covariates from the subsample for which estimation of propensity scores is desired.
    X_2 : array-like
        covariates from the auxuliary subsample.
    d_2 : list or one-dimensional array
        treatment values from the auxuliary subsample.

    Returns
    -------
    Estimated propensity scores for X_1.

    '''
    clf = LogisticRegressionCV(cv = 5, penalty = "l1", solver="liblinear").fit(X_2, d_2)
    prop_score = clf.predict_proba(X_1)
    
    return(prop_score)

def Huber(yHat, y, delta=1.):
    return np.where(np.abs(y-yHat) < delta, .5*(y-yHat)**2 ,
                    delta*(np.abs(y-yHat)-0.5*delta))

def WEIGHTS(m_1, m_2, d_a, grid = np.arange(0, 1.026, 0.025), criterion = "class"):
    '''
    Estimates optimal weights for each model to build an Ensemble Learner.
    
    Parameters
    ----------
    m_1 : list or one-dimensional array
        Propensity score estimations from the first model.
    m_2 : list or one-dimensional array
        Propensity score estimations from the first model.
    d_a : list or one-dimensional array
        Treatment variable for the sample for which the models were estimated.
    grid : np.arange
        Grid of weights to test. The number corresponds to weight assinged to 
        the first model.
    criterion : str, default 'class'
        Criterion to select the best model. 'class' selects based on classification
        error, 'mse' selects based on MSE.

    Returns
    -------
    best_weight : float
        Weight of the first model that results in the lowest MSE.
    '''
        
    mse = []
    class_acc = []
    cross_entropy = []
    huber = []
    for w in grid:
        prediction = w*m_1[:, 1] + (1-w)*m_2[:, 1]
        if criterion == "mse":
            mse.append(np.mean([(a-b)**2 for a,b in zip(d_a, prediction)]))
        
        if criterion == "class":
            class_pred = np.round(prediction)
            diff = [a-b for a,b in zip(d_a, class_pred)]
            class_acc.append(diff.count(0)/len(diff))
        
        if criterion == "cross-entropy":
            c_entropy = -d_a*np.log(prediction) - (1-d_a)*np.log(1 - prediction)
            cross_entropy.append(c_entropy.mean())
            
        if criterion == "huber":
            huber.append(Huber(prediction, d_a, 0.5).mean())
            #print(huber)
    
    if criterion == 'class':
        best_weight_class = np.round(grid[class_acc.index(min(class_acc))], 3)
        return best_weight_class
    if criterion == 'mse':
        best_weight_mse = np.round(grid[mse.index(min(mse))], 3)
        return best_weight_mse
    if criterion == "cross-entropy":
        best_weight_entr = np.round(grid[cross_entropy.index(min(cross_entropy))], 3)
        return best_weight_entr
    if criterion == "huber":
        best_weight_huber = np.round(grid[huber.index(min(huber))], 3)
        return best_weight_huber
    
def PSE(X_1, X_2, d_1, d_2, save = False, clnumber = None, weights = 0.5, criterion = "class", savedir = None):
    if (save == True) & (clnumber == None):
        raise ValueError("if save == True, cluster number should be specified")
        
    ps_rf_1 = RF(X_1, X_2, d_2)
    ps_rf_2 = RF(X_2, X_1, d_1)
    
    ps_lasso_1 = LOGLASSO(X_1, X_2, d_2)
    ps_lasso_2 = LOGLASSO(X_2, X_1, d_1)
    
    if weights == True:
    
        w_1 = WEIGHTS(ps_rf_1, ps_lasso_1, d_1, criterion = criterion)
        ps_1 = ps_rf_1[:, 1]*w_1 + ps_lasso_1[:, 1]*(1-w_1)
        
        w_2 = WEIGHTS(ps_rf_2, ps_lasso_2, d_2, criterion = criterion)
        ps_2 = ps_rf_2[:, 1]*w_2 + ps_lasso_2[:, 1]*(1-w_2)
    
    else:
        if type(weights) != float:
            raise TypeError("if weights is not True, then a float weight must be passed")
        ps_1 = ps_rf_1[:, 1]*weights + ps_lasso_1[:, 1]*(1-weights)
        ps_2 = ps_rf_2[:, 1]*weights + ps_lasso_2[:, 1]*(1-weights)
    
    if save == True:
        pd.Series(ps_rf_1[:, 1]).to_csv("{savedir}\\randomforest_predicted_scores_cluster{str(clnumber)}_1.csv", index = False)
        pd.Series(ps_rf_2[:, 1]).to_csv("{savedir}\\randomforest_predicted_scores_cluster{str(clnumber)}_2.csv", index = False)
        
        pd.Series(ps_lasso_1[:, 1]).to_csv("{savedir}\\lasso_predicted_scores_cluster{str(clnumber)}_1.csv", index = False)
        pd.Series(ps_lasso_2[:, 1]).to_csv("{savedir}\\lasso_predicted_scores_cluster{str(clnumber)}_2.csv", index = False)
        
        pd.Series(ps_1).to_csv("{savedir}\\el_predicted_scores_cluster{str(clnumber)}_1.csv", index = False)
        pd.Series(ps_2).to_csv("{savedir}\\el_predicted_scores_cluster{str(clnumber)}_2.csv", index = False)
        
    return(ps_1, ps_2)

def l20(X_1, X_2, d_2, explanatory_variable):
    '''
    Calculates estimated value of l20.
    
    Parameters
    ----------
    X_1 : array-like
        covariates from the subsample for which estimation of l20 is desired.
    X_2 : array-like
        covariates from the auxuliary subsample.
    d_2 : list or one-dimensional array
        treatment values from the auxuliary subsample.

    Returns
    -------
    Estimated values of l20 for X_1.

    '''
    # construct variable (Date - (avg Date))*Y
    lmbda = (X_2["Date"]).mean()
    l20_a = (X_2["Date"] - lmbda) * X_2[explanatory_variable]
    
    # select only observations with D=0
    d_2_d0 = d_2[d_2["D"] == 0]
    x_2_d0 = X_2.loc[d_2_d0.index]
    l20_a_2_d0 = [l20_a[c] for c in d_2_d0.index]
    
    l20_rf_1 = RandomForestRegressor(1000, max_depth = 20)\
        .fit(x_2_d0.iloc[:, :-2], l20_a_2_d0).predict(X_1.iloc[:, :-2])
        
    l20_lasso_1 = LassoCV(cv = 10)\
        .fit(x_2_d0.iloc[:, :-2], l20_a_2_d0).predict(X_1.iloc[:, :-2])
    
    l20_el_1 = np.mean(np.array([l20_rf_1, l20_lasso_1]), axis = 0)
    
    return l20_el_1

def theta_k(X_1, X_2, d_1, d_2, PS_1, l20_1, explanatory_variable):
    '''
    Calculates theta^tilde for subsample X_1.

    Parameters
    ----------
    X_1 : array-like
        covariates from the subsample for which estimation of theta is desired.
    X_2 : array-like
        covariates from the auxuliary subsample.
    d_1 : list or one-dimensional array
        treatment values from the subsample for which estimation of theta is desired.
    d_2 : list or one-dimensional array
        treatment values from the auxuliary subsample.
    PS_1 : list or one-dimensional array
        estimated propensity scores for the subsample for which estimation of theta is desired.
    l20_1 : list or one-dimensional array
        estimated values of l_20 variable for the subsample for which estimation of theta is desired.

    Returns
    -------
    theta_1 : float
        Estimated theta^tilde for subsample X_1.

    '''

    p_1 = float(d_2.mean())
    p_2 = float(d_1.mean())
    
    a_1 = d_1["D"] - PS_1
    
    lmdba_1 = X_2["Date"].mean()
    b_1 = ([p_1]*len(PS_1)) * (1 - PS_1) * ([lmdba_1]*len(PS_1)) * ([1 - lmdba_1]*len(PS_1))
    
    c_1 = (X_1["Date"] - ([lmdba_1]*len(PS_1))) * X_1[explanatory_variable] - l20_1
    
    theta_1 = ((a_1/b_1)*c_1).mean()
    
    return theta_1

def ATET(clnumber:int, x_data:pd.DataFrame, d_data:pd.DataFrame, explanatory_variable:str, 
         save:bool = False, K:int = 2, weights:float = 0.5, criterion:str = "class", 
         savedir:str = None) -> float:
    
    if (save == True) & (savedir == None):
        raise ValueError("if save == True, savedir must be specified")
        
    x = x_data
    d = d_data
    
    cl_1_index = np.random.randint(0, len(x), int(len(x)/K))

    x_cluster_1 = x.loc[x.index.isin(cl_1_index)]
    x_cluster_2 = x.loc[~x.index.isin(cl_1_index)]
    
    d_cluster_1 = d.loc[d.index.isin(cl_1_index)]
    d_cluster_2 = d.loc[~d.index.isin(cl_1_index)]
    
    ps_cl_1, ps_cl_2 = PSE(x_cluster_1.iloc[:, :-2], x_cluster_2.iloc[:, :-2], 
                           d_cluster_1["D"], d_cluster_2["D"], save = save, 
                           clnumber = clnumber, weights = weights, savedir = savedir)
 
    x_cluster_1 = x_cluster_1.reset_index(drop = True)
    d_cluster_1 = d_cluster_1.reset_index(drop = True)
    x_cluster_1 = x_cluster_1.loc[~x_cluster_1.index.isin(np.where((ps_cl_1 < 0.1) | (ps_cl_1 > 0.9))[0])]
    d_cluster_1 = d_cluster_1.loc[~d_cluster_1.index.isin(np.where((ps_cl_1 < 0.1) | (ps_cl_1 > 0.9))[0])]
    
    x_cluster_2 = x_cluster_2.reset_index(drop = True)
    d_cluster_2 = d_cluster_2.reset_index(drop = True)
    x_cluster_2 = x_cluster_2.loc[~x_cluster_2.index.isin(np.where((ps_cl_2 < 0.1) | (ps_cl_2 > 0.9))[0])]
    d_cluster_2 = d_cluster_2.loc[~d_cluster_2.index.isin(np.where((ps_cl_2 < 0.1) | (ps_cl_2 > 0.9))[0])]
    
    mask = np.ones(len(ps_cl_1), bool)
    mask[np.where((ps_cl_1 < 0.1) | (ps_cl_1 > 0.9))] = 0
    ps_cl_1 = ps_cl_1[mask]
    
    mask = np.ones(len(ps_cl_2), bool)
    mask[np.where((ps_cl_2 < 0.1) | (ps_cl_2 > 0.9))] = 0
    ps_cl_2 = ps_cl_2[mask]
    
    l20_el_cl_1 = l20(x_cluster_1, x_cluster_2, d_cluster_2, explanatory_variable)
    l20_el_cl_2 = l20(x_cluster_2, x_cluster_1, d_cluster_1, explanatory_variable)
      
    theta_cl_1 = theta_k(x_cluster_1, x_cluster_2, d_cluster_1, d_cluster_2, 
                         ps_cl_1, l20_el_cl_1, explanatory_variable)
    theta_cl_2 = theta_k(x_cluster_2, x_cluster_1, d_cluster_2, d_cluster_1, 
                         ps_cl_2, l20_el_cl_2, explanatory_variable)
    
    theta_cl = np.mean([theta_cl_1, theta_cl_2])
    
    if save == True:
        pd.Series(l20_el_cl_1).to_csv(f"{savedir}\\el_l20_cluster{str(clnumber)}_1.csv", index = False)
        pd.Series(l20_el_cl_2).to_csv(f"{savedir}\\el_l20_cluster{str(clnumber)}_2.csv", index = False)
      
        with open(f".{savedir}\\theta_cluster{str(clnumber)}.txt", 'w') as f:
            f.write('%d' % theta_cl)
            
    return theta_cl