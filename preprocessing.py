import numpy as np
import csv
import pandas

def readDocument():
    df = pandas.read_csv('training_dataset.csv')
    print(df)

readDocument()