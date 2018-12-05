import numpy as np
import pandas

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

print()
#Data Testing Document
docTesting = pandas.concat([docTest2, docTest1, docTest0])
docTesting.to_csv("data_testing.csv")

print('\nTesting Document', docTesting)

#Lowercase
#print(docClass0_20['Teks'].str.lower())
#print('Test', docTest0[docTest0['Teks'].str.contains('([a-zA-Z]+)', regex=True)])