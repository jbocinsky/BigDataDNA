import os
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import xlwt
from time import gmtime, strftime
import xlrd
import csv

#*************************************************************
def sequenceToNum(originalSequence, numSequences):
	#length of a sequence
	numChars = len(originalSequence[0])

	xSequenceNum = np.zeros((numSequences,numChars*4))

	for seq in range(0, numSequences):
		# print("This is row %d" % (seq))
		sequence = originalSequence[seq]
		sequenceNum = np.zeros((14,4))
		for s in range(0, numChars):
			thisChar = sequence[s]
			# print("Character %d:" % (s), thisChar)
			if(thisChar == 'A'):
				sequenceNum[s,:] = [0,0,0,1]
			elif(thisChar == 'C'):
				sequenceNum[s,:] = [0,0,1,0]
			elif(thisChar == 'G'):
				sequenceNum[s,:] = [0,1,0,0]
			elif(thisChar == 'T'):
				sequenceNum[s,:] = [1,0,0,0]

		# print('This sequence is:', sequenceNum)
		sequenceNum = sequenceNum.ravel()
		xSequenceNum[seq] = sequenceNum

	# print('My Sequences:', xSequenceNum)
	return xSequenceNum

#***************************************************************

def shuffleData(a, b):
	assert len(a) == len(b)
	p = np.random.permutation(len(a))
	return a[p], b[p]

#***************************************************************

def createModel():
	inputDim = 14*4

	#Define Model
	model = Sequential()
	# model.add(Conv2D(7, (3, 3), activation='relu', input_shape=(14, 4, 1)))
	# model.add(MaxPooling2D(pool_size=(2, 2)))
	
	# model.add(Flatten())
	model.add(Dense(70, input_dim=inputDim, init='random_uniform', activation='relu'))
	model.add(Dropout(0.3))
	model.add(Dense(80, activation='relu'))
	model.add(Dropout(0.3))
	model.add(Dense(60, activation='relu'))
	model.add(Dropout(0.3))
	model.add(Dense(30, activation='relu'))
	model.add(Dropout(0.3))
	model.add(Dense(20, activation='relu'))
	model.add(Dropout(0.3))
	model.add(Dense(15, activation='relu'))
	model.add(Dropout(0.3))
	model.add(Dense(10, activation='relu'))
	model.add(Dropout(0.3))
	model.add(Dense(5, activation='relu'))
	model.add(Dropout(0.3))
	# model.add(Dense(1, activation='sigmoid'))
	model.add(Dense(2, activation='softmax'))

	#Compile Model
	adam = optimizers.Adam(lr=0.0001)
	# model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
	model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

	return model

#***************************************************************

def trainModel(model, data, labels, numEpochs=100, batchSize=10):
	#*****************Old Training************************
	#Fit the model
	# validationSet = np.vstack((data[900:999], 


	model.fit(data, labels, validation_split=0.05, epochs=numEpochs, batch_size=batchSize, shuffle=True, verbose=2)
	# model.fit(data, labels, epochs=numEpochs, batch_size=batchSize, shuffle=True, verbose=2)


	#Evaluate the model
	scores = model.evaluate(data, labels)
	print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

	#****************New Training*************************
	#fix random seed for reproducibility
	# seed = 7
	# np.random.seed(seed)

	# estimator = KerasClassifier(build_fn=createModel, epochs=numEpochs, validation_split=0.1, batch_size=batchSize, verbose=2)
	# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
	# results = cross_val_score(estimator, data, labels, cv=kfold)
	# print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

	return model

#***************************************************************

def documentResults(predictions):
	numPredictions = len(predictions)+1
	#initilze a workbook
	book = xlwt.Workbook(encoding="utf-8")

	#Add sheet to workbook
	sheet1 = book.add_sheet("Python Sheet 1")
	
	#Go to results folder
	resultsPath = "./../../results/"
	os.chdir(resultsPath)

	sheet1.write(0, 0, 'id')
	sheet1.write(0, 1, 'prediction')

	for index in range(1, numPredictions):
		sheet1.write(index, 0, index-1)
		sheet1.write(index, 1, predictions[index-1])

	excelFileName = strftime("%m-%d_%H-%M_results.xls", gmtime())
	book.save(excelFileName)

	print('Wrote to file:', excelFileName)

#***************************************************************


#file names:
trainFile = "train.csv"
testFile = "test.csv"

#data path
dataPath = "./../data/KaggleDNACompetition"


#get current working directory
cwd = os.getcwd()

#Go to where data is
os.chdir(dataPath)


#Get training data
trainData = pd.read_csv(trainFile)
xID = trainData['id']
xSequence = trainData['sequence']
yLabel = trainData['label']


#Get test data
testData = pd.read_csv(testFile)
testID = testData['id']
testSequence = testData['sequence']

# #Shuffle Training Data
# xSequence, yLabel = shuffleData(xSequence, yLabel)

#Number of training DNA sequences
numSequences = len(trainData.index)
#Transform characters to numbers
xSequenceNum = sequenceToNum(xSequence, numSequences)

#Number of testing DNA sequences
numSequences = len(testData.index)
#Transform characters to numbers
testSequenceNum = sequenceToNum(testSequence, numSequences)


# print(xSequenceNum[1:5])
# print(np.vstack((xSequenceNum[1:4],xSequenceNum[4:5])))


#Define Model
model = createModel()


#Train model
numEpochs = 100
batchSize = 30
model = trainModel(model, xSequenceNum, yLabel, numEpochs, batchSize)


#Calculate predictions of test data
predictions = model.predict(testSequenceNum)

# print(predictions)

#Get max from softmax
# predictions = [round(x[0]) for x in predictions]
classPredictions = [np.argmax(x) for x in predictions]


for ind in range(0,len(classPredictions)):
	classPredictions[ind] = int(classPredictions[ind])

print(classPredictions)

documentResults(classPredictions)


#Calculate predictions of original data
predictions = model.predict(xSequenceNum)

# print(predictions)

#Get max from softmax
# classPredictions = [round(x[0]) for x in predictions]
classPredictions = [np.argmax(x) for x in predictions]

for ind in range(0,len(classPredictions)):
	classPredictions[ind] = int(classPredictions[ind])


print(classPredictions)



