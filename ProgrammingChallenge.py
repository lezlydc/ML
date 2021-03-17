# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 00:34:14 2021

@author: Lili's laptop
"""
## for data
import pandas as pd
import numpy as np
## for plotting
import matplotlib.pyplot as plt
import seaborn as sns
## for statistical tests
import scipy as sp
import statsmodels.formula.api as smf
import statsmodels.api as sm
## for machine learning
import sklearn as sk
from sklearn import model_selection, preprocessing, feature_selection, ensemble, linear_model, metrics, decomposition, gaussian_process, neural_network, discriminant_analysis, compose
## for explainer
from lime import lime_tabular
from sklearn import naive_bayes, datasets
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.utils.fixes import loguniform
from imblearn.over_sampling import SMOTE, ADASYN
import xgboost as xgb
from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_squared_error

#Functions 

def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):

    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        model_selection.learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)
    
     # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt

#Read data
original = pd.read_csv('TrainOnMe.csv')
original.head()
dtf = original.drop(["Index "], axis=1)

#Preprocessing
    #Delete faulty entries
dtf = dtf.dropna()

    #Enconding labels 
dtf['y'] = preprocessing.LabelEncoder().fit_transform(dtf['y'])

    #Categorical to numerical         
x6_replace = {'x6': {'GMMs and Accordions': 1, 'Bayesian Inference': 0, 'Bayesian Interference': 0}}
labels = dtf['x6'].astype('category').cat.categories.tolist()
replace_map_comp = {'x6' : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
dtf.replace(x6_replace, inplace=True)

x12_replace = {'x12': {'TRUE': 1, 'FALSE': 0, 'Flase': 0}}
labels = dtf['x12'].astype('category').cat.categories.tolist()
replace_map_comp = {'x12' : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
dtf.replace(x12_replace, inplace=True)

    #Outliers
dtf = dtf[np.abs(dtf.x1-dtf.x1.mean()) <= (3*dtf.x1.std())]
dtf = dtf[np.abs(dtf.x1-dtf.x1.mean()) <= (3*dtf.x1.std())]

#    #Correlation between features
corr = np.corrcoef(dtf.drop(['y'],axis=1),rowvar=False)
#dtf = dtf.drop(['x3'], axis=1)

    #Polynomial features
x = dtf.drop(['y'],axis=1)
y = dtf['y'] 
x = preprocessing.PolynomialFeatures(degree = 2, interaction_only = True, include_bias = False).fit_transform(x)

#     #Split data 
# # TODO: Look into missclassifications 
# # TODO: Try more in the voting
  
xtrain,xtest,ytrain,ytest = model_selection.train_test_split(x,y,test_size=0.3,shuffle=True)
   
    #Oversampling minority class
#xtrain,ytrain = SMOTE().fit_resample(xtrain,ytrain)

    #Normalizing features
scaler = preprocessing.RobustScaler()
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.fit_transform(xtest)

# #Algorithms

    #Gradient Boosting
# gradientBoosting = sk.ensemble.GradientBoostingClassifier(subsample=0.7,n_estimators=1500, min_samples_split=10, max_features=6, max_depth=4, learning_rate=0.01).fit(xtrain,ytrain)
# # param_dic = {'learning_rate':[0.15,0.1,0.05,0.01,0.005,0.001],'n_estimators':[100,250,500,750,1000,1250,1500,1750],'max_depth':[2,3,4,5,6,7],'min_samples_split':[2,4,6,8,10,20,40,60,100],'max_features':[2,3,4,5,6,7],'subsample':[0.7,0.75,0.8,0.85,0.9,0.95,1]}
# # random_search = sk.model_selection.RandomizedSearchCV(gradientBoosting,param_distributions=param_dic, n_iter=30, scoring="accuracy").fit(xtrain, ytrain)
# # print("Best Model parameters:", random_search.best_params_)
# # print("Best Model mean accuracy:", random_search.best_score_)
# # gradientBoosting = random_search.best_estimator_
# print("GB Training accuracy", gradientBoosting.score(xtrain, ytrain))
# print("GB Cross val accuracy", model_selection.cross_val_score(gradientBoosting,xtrain,ytrain,cv=10).mean())
# print("GB Testing accuracy", gradientBoosting.score(xtest, ytest), "\n")
# #

#     #Random Forest
# randomForest = sk.ensemble.RandomForestClassifier(n_estimators=1000, min_samples_split=6, max_features=9, max_depth=6).fit(xtrain,ytrain)
# # param_dic = {'n_estimators':[100,250,500,750,1000,1250,1500,1750],'max_depth':[2,3,4,5,6,7],'min_samples_split':[2,4,6,8,10,20,40,60,100],'max_features':[2,3,4,5,6,7,9]}
# random_search = sk.model_selection.RandomizedSearchCV(randomForest,param_distributions=param_dic, n_iter=30, scoring="accuracy").fit(xtrain,ytrain)
# print("Best Model parameters:", random_search.best_params_)
# print("Best Model mean accuracy:", random_search.best_score_)
# randomForest = random_search.best_estimator_
# print("RF Training accuracy", randomForest.score(xtrain, ytrain))
# print("RF Cross val accuracy", model_selection.cross_val_score(randomForest,xtrain,ytrain,cv=10).mean())
# print("RF Testing accuracy", randomForest.score(xtest, ytest), "\n")
# #

    #SVM 
# SVM = sk.svm.SVC()
# param_grid = [
#   {'C': loguniform(1e0, 1e3), 'degree': [2,3,4,5,6,7,9,10],'kernel': ['poly']},
#   {'C': loguniform(1e0, 1e3), 'gamma': loguniform(1e-4, 1e-3), 'kernel': ['rbf']},
#   ]
# random_search = model_selection.RandomizedSearchCV(SVM, param_grid, n_iter = 30).fit(xtrain,ytrain)
# print("SVM Optimized parameters:", random_search.best_params_)
# print("SVM Optimized best validation score:", random_search.best_score_)
# SVM = random_search.best_estimator_ 
# print("SVM poly Training accuracy", SVM.score(xtrain, ytrain))
# print("SVM poly cross val accuracy",model_selection.cross_val_score(SVM,xtrain,ytrain).mean())
# print("SVM poly testing accuracy", SVM.score(xtest, ytest), "\n")

#    # KNN 
# KNN = sk.neighbors.KNeighborsClassifier()
# param_dic = {'n_neighbors':[3,5,10,15,18],'algorithm':["auto", "ball_tree", "kd_tree", "brute"]}
# random_search = sk.model_selection.RandomizedSearchCV(KNN,param_distributions=param_dic, n_iter=10, scoring="accuracy").fit(xtrain,ytrain)
# print("Best Model parameters:", random_search.best_params_)
# print("Best Model mean accuracy:", random_search.best_score_)
# KNN = random_search.best_estimator_
# print("KNN Training accuracy", KNN.score(xtrain, ytrain))
# print("KNN cross val accuracy",model_selection.cross_val_score(KNN,xtrain,ytrain).mean())
# print("KNN Testing accuracy", KNN.score(xtest, ytest),"\n")

#    # Voting Classifier 
# voting = sk.ensemble.VotingClassifier(estimators=[('gB', gradientBoosting), ('rF', randomForest)],voting='hard').fit(xtrain,ytrain)
# for clf, label in zip([gradientBoosting,randomForest], ['Gradient Boosting','Random Forest' ]):
#       scores = sk.model_selection.cross_val_score(clf, xtrain, ytrain, scoring='accuracy', cv=5)
#       print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
# print("Vote Validation accuracy", voting.score(xtest, ytest),"\n")

    #Learning curves 
# title = "Gradient Boosting Val set"
# cv = model_selection.ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
# plot_learning_curve(gradientBoosting, title, xtrain, ytrain, axes=None, ylim=(0, 1.01), cv=cv, n_jobs=4)
# plt.show()

    #XGBoost
dtrain = xgb.DMatrix(xtrain,label=ytrain)
dtest = xgb.DMatrix(xtest,label=ytest)
param = {'max_depth': 2, 'eta': 1}
param['nthread'] = 4
param['eval_metric'] = 'merror'
num_round = 10
evallist = [(dtest, 'test'), (dtrain, 'train')]
bst = xgb.train(param, dtrain, num_round, evallist)

# def xgb_evaluate(max_depth, gamma, colsample_bytree):
#     params = {'eval_metric': 'rmse',
#               'max_depth': int(max_depth),
#               'subsample': 0.8,
#               'eta': 0.1,
#               'gamma': gamma,
#               'colsample_bytree': colsample_bytree}
#     # Used around 1000 boosting rounds in the full model
#     cv_result = xgb.cv(params, dtrain, num_boost_round=100, nfold=3)    
    
#     # Bayesian optimization only knows how to maximize, not minimize, so return the negative RMSE
#     return -1.0 * cv_result['test-rmse-mean'].iloc[-1]

# xgb_bo = BayesianOptimization(xgb_evaluate, {'max_depth': (3, 7), 
#                                              'gamma': (0, 1),
#                                              'colsample_bytree': (0.3, 0.9)})
# xgb_bo.maximize(init_points=3, n_iter=5, acq='ei')
# params = xgb_bo.res['max']['max_params']
# params['max_depth'] = int(params['max_depth'])
# # Train a new model with the best parameters from the search
# model2 = xgb.train(params, dtrain, num_boost_round=250)

# # Predict on testing and training set
# y_pred = model2.predict(dtest)
# y_train_pred = model2.predict(dtrain)

# # Report testing and training RMSE
# print(np.sqrt(mean_squared_error(ytest, y_pred)))
# print(np.sqrt(mean_squared_error(ytrain, y_train_pred)))