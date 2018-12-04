import numpy as np
import pandas

#Variable initialization
documentName = 'dataset_sms_ori.csv'
totalRowClass = np.zeros(3)

#Method for Document Parsing
def readDocument(documentName):
    df = pandas.read_csv(documentName)
    return df

def getRows(partOfDocument):
    count = partOfDocument.index
    return len(count)

def getRowOfEveryClass(totalRowClass, document):
    for i in range(len(totalRowClass)):
        totalRowClass[i] = getRows(document[document['label'] == i])
    return totalRowClass

def lowerCaseDocument(documentName):
    return documentName.str.lower()

#Read Document
document = readDocument(documentName)

#Print Total Rows for Every class
print('Total row every class ' , getRowOfEveryClass(totalRowClass, document))

#Filter Based on Document Label
docClass0 = document[document['label'] == 0]
docClass1 = document[document['label'] == 1]
docClass2 = document[document['label'] == 2]

#Create document with 20% of each class for Data Testing
docClass0_20 = docClass0[-(int(getRowOfEveryClass(totalRowClass, document)[0] * 0.2) ) :  ]
docClass1_20 = docClass1[-(int(getRowOfEveryClass(totalRowClass, document)[0] * 0.2) ) :  ]
docClass2_20 = docClass2[-(int(getRowOfEveryClass(totalRowClass, document)[0] * 0.2) ) :  ]


#Create document with 80% of each class
docClass0_80 = docClass0[ : (int(getRowOfEveryClass(totalRowClass, document)[0] * 0.8) ) ]
docClass1_80 = docClass1[ : (int(getRowOfEveryClass(totalRowClass, document)[1] * 0.8) ) ]
docClass2_80 = docClass2[ : (int(getRowOfEveryClass(totalRowClass, document)[2] * 0.8) ) ]

#Data Training Document
docTraining = pandas.concat([docClass2_80, docClass1_80, docClass0_80])
docTraining.to_csv("data_training.csv")

print('\nTraining Document', docTraining)

#Data Testing Document
docClass0_20.to_csv("datauji_0.csv")
docClass1_20.to_csv("datauji_1.csv")
docClass2_20.to_csv("datauji_2.csv")

#Lowercase
#print(docClass0_20['Teks'].str.lower())
print('Test', docClass0_20[docClass0_20['Teks'].str.contains('([a-zA-Z]+)', regex=True)])