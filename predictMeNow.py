# -*- coding: utf-8 -*-

import backend

# COMPULSORY SETTINGS
trainDirectory = 'PATH/TO/TRAIN.csv'
testDirectory = '' #replace with test directory if want. to be supported soon.
TARGET = 'OutcomeType'

#Optional Settings

params = {
'oneHotEncode':False,
'imputeMissing':False,
 'model':'xgboost',   # supported: 'extraTrees','randomForest','knn'. Others not supported yet.
'cross_validate':True,
'train':trainDirectory,
 'test':testDirectory,
 'target':TARGET
}

if __name__ == '__main__':
   backend.run(params)
