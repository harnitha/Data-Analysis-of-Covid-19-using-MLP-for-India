import pandas as pd
import prediction
from csv import writer
from numpy import mean



def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)
        

def addRecord():
  data=pd.read_csv('https://api.covid19india.org/csv/latest/case_time_series.csv')
  print(data.head())
  d=data['Total Confirmed']
  d.values

  # data split
  n_test = 12
  # define config[n_input, n_nodes, n_epochs, n_batch ]
  config = [24, 500, 200, 110]
  # grid search
  result= prediction.repeat_evaluate(d.values, config, n_test)
  # summarize scores
  scores=result[0]

  accuracy=((abs(scores[1][-1][0]-d.values[-1])/d.values[-1])*100)
  newRow=[data.Date.values[-1],scores[1][-1][0],d.values[-1],(100-accuracy)]
  todayDate=data.Date.iloc[-1]

  data1=pd.read_csv('predictionCovid.csv')
  if data1.Date.iloc[-1]==todayDate:
    pass
  else:
    append_list_as_row('predictionCovid.csv',newRow)
