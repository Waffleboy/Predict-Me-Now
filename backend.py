# -*- coding: utf-8 -*-

import pandas as pd
from sklearn import metrics,cross_validation
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder,Imputer
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.cross_validation import StratifiedShuffleSplit,train_test_split
from sklearn.linear_model import SGDClassifier
import xgboost as xgb

def loadDataFrame(params,identifier):
    try:
        if identifier == 'train':
            PATH = params['train']
        else:
            PATH = params['test']
        df = pd.read_csv(PATH)
    except OSError:
        raise Exception('Error reading '+identifier+' CSV. Is the path correct?')
    except Exception:
        raise('Unknown exception for '+identifier+'CSV. Check encoding of file or manually change it')
    return df
    
#==============================================================================
#                       Preprocessing Functions
#==============================================================================

# Remove missing values from CSV by dropping or imputation.
# INPUT:
# < pandas dataframe > df: train or test csv
# < boolean >    impute: if True, use impute. else drop.
# OUTPUT:
# < pandas dataframe> df: dataframe with missing values removed
def removeMissingValues(df,impute=False):
    if impute == False: #if not impute, just drop missing rows.
        return df.dropna()
    else:
        pass #settle impute later if needed.

# Magical function to encode everything thats not a numeric column.
# INPUT:
# < pandas datadrame > df: train or test csv
# OUTPUT:
# < pandas datadrame > df: encoded df
def dummyEncode(df):
    columnsToEncode = list(df.select_dtypes(include=['category','object']))
    le = LabelEncoder()
    for feature in columnsToEncode:
        try:
            df[feature] = le.fit_transform(df[feature])
        except:
            print('Error encoding '+feature)
    return df

# TO BE IMPLEMENTED.
def oneHotEncode(df):
    pass

#main preprocessing function to run the above.
# INPUT:
# < pandas datadrame > df: train or test csv
# OUTPUT:
# < pandas datadrame > df: preprocessed df
def runPreprocess(df,params):
    df = removeMissingValues(df,params['imputeMissing'])
    if params['oneHotEncode']:
        df = oneHotEncode(df)
    else:
        df = dummyEncode(df)
    return df

#==============================================================================
#                          Feature Engineering
#==============================================================================

# Insert your own here.

#==============================================================================
#               Splitting to dataset, target and train / test 
#==============================================================================

# Split to target and data columns.
# INPUT:
# < pandas datadrame > df: train or test csv
# < string > targetName: the name of the column thats the target variable
# OUTPUT:
# < pandas datadrame > df: df without target
# < pandas series >  target: target column.
def splitDatasetTarget(df,targetName):
    dataset = df.drop(targetName,axis=1)
    target = df[targetName]
    return dataset,target

# ONLY USED IF DATASET TOO LARGE FOR CROSS - VAL
def splitTrainTest(dataset,target,test_size=0.20):
    trainX,trainY,testX,testY = train_test_split(dataset,target, 
                                                 test_size=test_size, 
                                                 random_state=123)
    return trainX,trainY,testX,testY

#==============================================================================
#                          Sampling (if needed)  
#==============================================================================

#Default: sample is 20% of entire dataset
def stratifiedSampleGenerator(dataset,target,test_size=0.2):
    X_fit,X_eval,y_fit,y_eval= train_test_split(dataset,target,
                                                test_size=test_size,
                                                stratify=target)
    return X_eval.reset_index(drop=True),y_eval.reset_index(drop=True)
    
#==============================================================================
#                               Models
#==============================================================================

def xgBoost():
    clf = xgb.XGBClassifier(max_depth = 8,n_estimators=300,nthread=8,seed=123,
                            silent=1,objective= 'multi:softmax',learning_rate=0.1,
                            subsample=0.9)
    return clf

def randomForest():
    clf = RandomForestClassifier(max_depth=8, n_estimators=500,n_jobs=8,
                                 random_state=123)
    return clf

def extraTrees():
    clf = ExtraTreesClassifier(max_depth=8, n_estimators=500,n_jobs=8,random_state=123)
    return clf

def kNN():
    clf = KNeighborsClassifier(n_neighbors=3,n_jobs=8)
    return clf
    
def SVM():
    clf = SGDClassifier(n_jobs = 8)
    return clf

def getNameFromModel(clf):
    name = str(type(clf))
    name = name[name.rfind('.')+1:name.rfind("'")] #subset from last . to last '
    return name
    
#==============================================================================
#                        XGBoost Specific Functions 
#==============================================================================
    
def xgboostCV(clf, dataset,target ,useTrainCV=True, cv_folds=5, early_stopping_rounds=25):
    print('Running XGBOOST cross validation')
    if useTrainCV:
        xgb_param = clf.get_xgb_params()
        xgb_param['num_class'] = 6 #CHANGE THIS.
        xgtrain = xgb.DMatrix(dataset.values, label=target.values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=clf.get_params()['n_estimators'], nfold=cv_folds,
            metrics=['merror'], early_stopping_rounds=early_stopping_rounds, show_progress=False)
        CV_ROUNDS = cvresult.shape[0]
        print('Optimal Rounds: '+str(CV_ROUNDS))
        clf.set_params(n_estimators=CV_ROUNDS)
    return clf
    
        
#==============================================================================
#                             Misc Functions
#==============================================================================
def accuracyChecker(target,predicted):
    accuracy = metrics.accuracy_score(target,predicted)
    confMat = metrics.confusion_matrix(target,predicted)
    print('Cross val accuracy: '+str(accuracy))
    print('Confusion Matrix:')
    print(confMat)
    
def getSpecifiedClf(params):
    clf_name = params['model'].lower()
    if clf_name == 'xgboost':
        return xgBoost()
    elif clf_name == 'extratrees':
        return extraTrees()
    elif clf_name == 'randomforest':
        return randomForest()
    elif clf_name =='knn':
        return kNN()
    elif clf_name == 'svm':
        return SVM()
    else:
        raise Exception('Incorrect classifier name specified.')
        
#==============================================================================
#                              Run Models
#==============================================================================

def run(params):
    train = loadDataFrame(params,'train')
    if params['test']:
        test = loadDataFrame(params,'test')
        
    train = runPreprocess(train,params)
    clf = getSpecifiedClf(params)
    try:
        dataset,target = splitDatasetTarget(train,params['target'])
    except:
        raise Exception('Target not specified')
    try:
        cross_val = params['cross_validate']
    except:
        cross_val = False
        
    clfName = getNameFromModel(clf)
    if cross_val and clfName != 'XGBClassifier':
        print('Beginning cross validation')
        predicted = cross_validation.cross_val_predict(clf,dataset,target,cv=5,n_jobs=-1)
        accuracyChecker(target,predicted)
        return
        
    if clfName == 'XGBClassifier':
        print('Xgboost CV selected. Beginning to find optimal rounds')
        clf = xgboostCV(clf,dataset,target)
        print('Xgboost Accuracy on 80-20 split (for speed)')
            
    trainX,testX,trainY,testY = splitTrainTest(dataset,target)
    clf.fit(trainX,trainY)
    predicted = clf.predict(testX)
    accuracyChecker(testY,predicted)

        