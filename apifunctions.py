"""API Prediction Function"""

def FunctionGeneratePrediction(inp_carat , inp_x, inp_y, inp_z):
  import pandas as pd
  # Creating a data frame for the model input
  SampleInputData=pd.DataFrame(
    data=[[inp_carat , inp_x, inp_y, inp_z]],
    columns=['carat', 'x', 'y', 'z'])

  # Calling the function defined above using the input parameters
  Predictions=FunctionPredictResult(InputData= SampleInputData)

  # Returning the predictions
  return(Predictions.to_json())


"""**Python Function**"""

from re import IGNORECASE

def FunctionPredictResult(InputData):
  import pandas as pd
  Num_Inputs=InputData.shape[0]

  # Append new data w Training data
  DataForML=pd.read_pickle('DataForML.pkl')
  # Append input data
  InputData = pd.concat([InputData, DataForML], ignore_index=True)

  # In same order as model training
  Predictors=['carat', 'x', 'y', 'z']
  # Generating the input values to the model
  X=InputData[Predictors].values[0:Num_Inputs]

  # Generating the normalised values of X since it was done while model training
  # from sklearn.preprocessing import Normalizer
  # PredictorScaler=Normalizer()
  # PredictorScalerFit=PredictorScaler.fit(X)
  # X=PredictorScalerFit.transform(X)

    # Loading the Function from pickle file
  import pickle
  with open('Final_SVM_Model.pk1', 'rb') as fileReadStream:
    PredictionModel=pickle.load(fileReadStream)
    # Don't forget to close the filestream!
    fileReadStream.close()

    # Genrating Predictions
  Prediction=PredictionModel.predict(X)
  PredictionResult=pd.DataFrame(Prediction, columns=['Prediction'])
  return(PredictionResult)

def TrainModel():
  import pandas as pd
  import numpy as np

  #read dataset
  DiamondData=pd.read_csv('diamonds.csv', encoding="latin")

  #removing duplicates if any
  print("Shape before deleting duplicate values:", DiamondData.shape)
  DiamondData=DiamondData.drop_duplicates()
  print("Shape after deleting duplicate values", DiamondData.shape)

  DiamondData = DiamondData.drop(["Unnamed: 0"], axis =1)


  # remove faulty data in x, y, z
  DiamondData = DiamondData.drop(DiamondData[DiamondData["x"]==0] .index)
  DiamondData = DiamondData.drop(DiamondData[DiamondData["y"]==0] .index)
  DiamondData = DiamondData.drop(DiamondData[DiamondData["z"]==0] .index)

  # final selection of columns to be used in ML model
  SelectedColumns = ['carat', 'x', 'y', 'z']
  DataForML=DiamondData[SelectedColumns]
  DataForML.head()

  DataForML.head()

  # Saving final data subset for reference during deployment
  DataForML.to_pickle('DataForML.pkl')

  # Adding target var to data
  DataForML['price']=DiamondData['price']
  print (DataForML.columns)

  TargetVariable ='price'
  Predictors = SelectedColumns

  X=DataForML[Predictors].values
  y=DataForML[TargetVariable].values

  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=428)

  TargetVariable='price'
  Predictors=['carat', 'x', 'y', 'z']

  X=DataForML[Predictors].values
  y=DataForML[TargetVariable].values

  # Normalise dataset
  from sklearn.preprocessing import Normalizer
  PredictorScaler=Normalizer()
  PredictorScalerFit=PredictorScaler.fit(X)

  X=PredictorScalerFit.transform(X)

  # Check for data loss
  print(X.shape)
  print(y.shape)

  from sklearn import svm
  RegModel= svm.SVR(C=50, kernel='rbf', gamma=0.01)

  # Retrain model with 100% available data
  Final_SVM_Model=RegModel.fit(X,y)


  import pickle
  import os

  with open('Final_SVM_Model.pk1', 'wb') as fileWriteStream:
    pickle.dump(Final_SVM_Model, fileWriteStream)
    fileWriteStream.close()

  print('pickle file of Predictive Model located:'),os.getcwd()
