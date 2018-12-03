import numpy as np
import csv
import pandas

def readDocument(documentName, columnName, nrows=10):
    df = pandas.read_csv(documentName, nrows=nrows)
    if(columnName == "none"):
        return df
    else:
        return df[columnName]

def writeDocument(documentName):
    df = pandas.DataFrame.to_csv(documentName)
    print(df)

df = pandas.read_csv('training_dataset.csv')
count = df.index

print(len(count)

document = readDocument('training_dataset.csv', "none")
print(document)

