import numpy as np
import pandas
import re
from sklearn.feature_extraction.text import CountVectorizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

#Stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

#Filter
factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()

#Variable initialization
documentName = 'dataset_sms_ori.csv'
totalClass = np.zeros(3)
stemmingArray = []

#Method for Document Parsing
def readDocument(documentName):
    df = pandas.read_csv(documentName)
    return df

def getRowOfEveryClass(totalClass, document):
    for i in range(len(totalClass)):
        totalClass[i] = len(document[document['label'] == i])
    return totalClass

def remove_urls (vTEXT):
    vTEXT = re.sub(r'((http(s)?://|(www?))[0-9a-z\./_+\(\)\$\#\&\!\?]+)', '', vTEXT, flags=re.MULTILINE)
    return(vTEXT)

def remove_emot(vTEXT):
    vTEXT = re.sub(r'[u"\U0001F600-\U0001F64F"u"\U0001F300-\U0001F5FF"]+', '', vTEXT, flags=re.MULTILINE)
    return(vTEXT)

def case_folding(document):
    document = document['Teks'].str.lower()
    return document

def cleansing(document):
    #simbol
    document = document.str.replace('[\\.()?,!""'':;/+=*#%\[\]]','')
    document = document.str.replace('[-_&]',' ')

    #number
    document = document.str.replace('\d','')

    #link/url
    document = document.str.replace('((http(s)?://|(www?))[0-9a-z\./_+\(\)\$\#\&\!\?]+)','')

    #emot
    document = document.str.replace('["\U0001F600-\U0001F64F" | "\U0001F300-\U0001F5FF"]+',' ')

    #fix space
    document = document.str.replace('\s+', ' ')
    return document

def filtering(docTraining):
    filteringArray = []
    for i in range(len(docTraining)):
        filteringResult = docTraining.iloc[i]
        filteringArray.append(stopword.remove(filteringResult))
    return filteringArray

def set_label(docLabel):
    labelArray = []
    for i in range(len(docLabel)):
        labelArray.append(docLabel.iloc[i])
    return labelArray

def stemming(docTraining):
    for i in range(len(docTraining)):
        stemmingResult = filtering(docTraining)[i]
        stemmingArray.append(stemmer.stem(stemmingResult))
    return stemmingArray


#Read Document
document = readDocument(documentName)

#Print Total Rows for Every class
#print('Total row every class ' , getRowOfEveryClass(totalClass, document))

#Filter Based on Document Label
docClass0 = document[document['label'] == 0]
docClass1 = document[document['label'] == 1]
docClass2 = document[document['label'] == 2]

#Create document with 20% of each class for Data Testing
docTest0 = docClass0[-(int(getRowOfEveryClass(totalClass, document)[0] * 0.2) ) :  ]
docTest1 = docClass1[-(int(getRowOfEveryClass(totalClass, document)[1] * 0.2) ) :  ]
docTest2 = docClass2[-(int(getRowOfEveryClass(totalClass, document)[2] * 0.2) ) :  ]


#Create document with 80% of each class
docTraining0 = docClass0[ : (int(getRowOfEveryClass(totalClass, document)[0] * 0.8) ) ]
docTraining1 = docClass1[ : (int(getRowOfEveryClass(totalClass, document)[1] * 0.8) ) ]
docTraining2 = docClass2[ : (int(getRowOfEveryClass(totalClass, document)[2] * 0.8) ) ]

#Data Training Document
docTraining = pandas.concat([docTraining2, docTraining1, docTraining0])
docTraining.to_csv("data_training.csv")

#print('\nTraining Document', docTraining)

#Data Testing Document
docTesting = pandas.concat([docTest2, docTest1, docTest0])
docTesting.to_csv("data_testing.csv")

#print('\nTesting Document', docTesting)

docTrainingLabel = docTraining['label']

#Preprocessing Document Training

#Case Fold
docTraining = case_folding(docTraining)
#print('\nCase Folding\n', docTraining)

#Cleansing
docTraining = cleansing(docTraining)
#print('\nCleansing\n', docTraining)

#Filtering
filteringArray = filtering(docTraining)
labelArray = set_label(docTrainingLabel)

#Stemming
'''stemmingArray = stemming(docTraining)

docStemmingTraining = pandas.DataFrame(data=stemmingArray, columns=['Teks'])
labelDataFrame = pandas.DataFrame(data=labelArray, columns=['label'])

docStemmingTraining = pandas.concat([docStemmingTraining, labelDataFrame], axis=1)
docStemmingTraining.to_csv('training_stemming.csv')'''

#Read Stemming Document
stemmingDocument = readDocument('training_stemming.csv')

#Tokenizing and Get Term
docTraining = stemmingDocument['Teks'].str.split()
stemmingResult = stemmingDocument['Teks']
countVectorize = CountVectorizer()
fitFeature = countVectorize.fit_transform(stemmingResult)
getFeature = countVectorize.get_feature_names()
arrayFeature = fitFeature.toarray()
#print(getFeature)

#Merge with the Label
#docTraining = pandas.concat([docTraining, stemmingDocument['label']], axis=1)
#print('\nDocument Testing\n', docTraining)

#docTraining.to_csv('preprocessing_document.csv')

