import numpy as np
import pandas
import re
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

#Method for Document Parsing
def readDocument(documentName):
    df = pandas.read_csv(documentName)
    return df

def getRows(partOfDocument):
    count = partOfDocument.index
    return len(count)

def getRowOfEveryClass(totalClass, document):
    for i in range(len(totalClass)):
        totalClass[i] = getRows(document[document['label'] == i])
    return totalClass

def lowerCaseDocument(documentName):
    return documentName.str.lower()

def remove_urls (vTEXT):
    vTEXT = re.sub(r'((http(s)?://|(www?))[0-9a-z\./_+\(\)\$\#\&\!\?]+)', '', vTEXT, flags=re.MULTILINE)
    return(vTEXT)

def remove_emot(vTEXT):
    vTEXT = re.sub(r'[u"\U0001F600-\U0001F64F"u"\U0001F300-\U0001F5FF"]+', '', vTEXT, flags=re.MULTILINE)
    return(vTEXT)

#Read Document
document = readDocument(documentName)

#Print Total Rows for Every class
print('Total row every class ' , getRowOfEveryClass(totalClass, document))

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

print('\nTraining Document', docTraining)

#Data Testing Document
docTesting = pandas.concat([docTest2, docTest1, docTest0])
docTesting.to_csv("data_testing.csv")

print('\nTesting Document', docTesting)

#Case Fold
docTesting = docTesting['Teks'].str.lower()
print('\nCase Folding\n', docTesting)

#Cleansing
#simbol
docTesting = docTesting.str.replace('[\\.()?,!""'':;/+=*#%\[\]]','')
docTesting = docTesting.str.replace('[-_&]',' ')
#number
docTesting = docTesting.str.replace('\d','')
#link/url
docTesting = docTesting.str.replace('((http(s)?://|(www?))[0-9a-z\./_+\(\)\$\#\&\!\?]+)','')
docTesting = docTesting.str.replace('\s+', ' ')
#emot
docTesting = docTesting.str.replace('["\U0001F600-\U0001F64F" | "\U0001F300-\U0001F5FF"]+',' ')

print('\nCleansing\n', docTesting)

#for i in range(getRows(docTesting)):
#    regexTesting[i] = remove_urls(docTesting.iloc[i])
#    regexTesting[i] = remove_emot(regexTesting[i])
    
#docTesting = re.sub(r'((http(s)?://|(www?\.))[0-9a-z\./_+\(\)\$\#\&\!\?]+)', '', (docTesting.iloc[i] for i in range(getRows(docTesting))), flags=re.MULTILINE)

#print('\nRegex\n', regexTesting)



#Filtering
for j in range(getRows(docTesting)):
    filteringResult = docTesting.iloc[j]
    output1 = stopword.remove(filteringResult)
    print('\nFiltering\n', output1)

docTesting.to_csv("test_prep.csv")
#Stemming
#for i in range(getRows(docTesting)):
 #   stemmingResult = docTesting.iloc[i]
 #   output   = stemmer.stem(stemmingResult)
 #   print('\nStemming\n', output)

#Tokenizing
#docTesting = docTesting.str.split()
#print('\nTokenizing\n', docTesting)