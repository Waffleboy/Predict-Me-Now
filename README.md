# Predict-Me-Now
### The Instant Ramen of Machine Learning!

## Description

Ever wanted to instantly find out the base accuracy of any dataset? Don't want to go through the hassle of modifying or writing code to know it? Introducing Predict-Me-Now, the library for ~~lazy~~ efficient data scientists!

## How to use

Simply clone the repository, and edit the following variables:

```
#Examples
trainDirectory = '/path/to/train.csv'
testDirectory = '' # only fill this if you want to generate predictions for test set.
TARGET = 'columnName' # this is the name of the column you want to predict
```

## Roadmap

1. Support for regressions (numerical target variables)
2. Generate predictions for test dataset (save to CSV)
3. OneHotEncoding and imputation support
4. Feature importance graph
