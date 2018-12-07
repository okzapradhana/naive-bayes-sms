#Will be executed after Preprocessing

import numpy as np
import preprocessing as pre
import pandas
from sklearn.feature_extraction.text import CountVectorizer

def rawTF(trainingDocument, number):

    sum = 0
    temp = []

    countVectorize = CountVectorizer()

    countVectorize.fit_transform(trainingDocument['Teks'])

    getLabel = trainingDocument[trainingDocument['label'] == number]['Teks']

    fitFeatureStemming = countVectorize.transform(getLabel)

    getFeature = countVectorize.get_feature_names()

    arrayLabel = np.transpose(fitFeatureStemming.toarray())

    for i in range(len(getFeature)):
        for j in range(len(arrayLabel[number])):
            sum += arrayLabel[i].item(j)
        temp.append((getFeature[i], sum))
    return temp

doc = pandas.read_csv('training_stemming.csv')
print(rawTF(doc, 1))