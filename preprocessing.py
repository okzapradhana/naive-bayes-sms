import numpy as np
import pandas

documentName = 'training_dataset.csv'
totalRowClass = np.zeros(3)

def readDocument(documentName, columnName = "none"):
    df = pandas.read_csv(documentName)
    if(columnName == "none"):
        return df
    else:
        return df[columnName]

def getDocumentTotalRows(documentName):
    doc = readDocument(documentName)
    count = doc.index
    return len(count)

def getRows(partOfDocument):
    count = partOfDocument.index
    return len(count)

document = readDocument(documentName)
print('CSV ' , document)

totalRows = getDocumentTotalRows(documentName)
print('Total row', totalRows)

for i in range(0, 3):
    print("Nilai i", i)
    totalRowClass[i] = getRows(document.loc[document['label'] == i])

print('Total row every class ' , totalRowClass)

rowsOfPartDoc = getRows(document.loc[document['label'] == 0])
print(rowsOfPartDoc)
