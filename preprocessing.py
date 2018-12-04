import numpy as np
import pandas

#Variable initialization
documentName = 'training_dataset.csv'
totalRowClass = np.zeros(3)

#Method for Document Parsing
def readDocument(documentName):
    df = pandas.read_csv(documentName)
    return df

def getDocumentTotalRows(documentName):
    doc = readDocument(documentName)
    count = doc.index
    return len(count)

def getRows(partOfDocument):
    count = partOfDocument.index
    return len(count)

def getRowOfEveryClass(totalRowClass):
    for i in range(len(totalRowClass)):
        totalRowClass[i] = getRows(document[document['label'] == i])
    return totalRowClass

#Read Document
document = readDocument(documentName)

#Print Total Rows for Every class
print('Total row every class ' , getRowOfEveryClass(totalRowClass))

#Get 80% of document of each Class
docClass0 = document[document['label'] == 0]
print(docClass0[: (int(getRowOfEveryClass(totalRowClass)[0] * 0.8) ) ])

docClass1 = document[document['label'] == 1]
print(docClass1[: (int(getRowOfEveryClass(totalRowClass)[1] * 0.8) ) ])

docClass2 = document[document['label'] == 2]
print(docClass2[: (int(getRowOfEveryClass(totalRowClass)[2] * 0.8) ) ])        