# Data-Analysis-of-Covid-19-using-MLP-for-India
TARP project that consists of a dashboard and prediction model for covid cases in India

TARP is part my college curriculum.

Coronavirus has made the world to a standstill by affecting billions of people worldwide. 
Doctors, nurses and other frontline workers are struggling hard to handle the situation. 
This project aims to implement the future scope of the research paper “Yadav, R. S. (2020). Data analysis of COVID-2019 epidemic using machine learning methods: a case study of India. International Journal of Information Technology, 1-10.” 
The future scope of the paper is to create a regression based neural network model for prediction of total COVID-19 cases in India the next day. 
The project aims to develop a regression based artificial neural network model that automatically estimate the number of total cases for the next day in India. 
The uniqueness in this project is that we combine real time analytics and prediction. 
It is mentioned real-time because this project uses API to get data that gets updated every day.
Thus, the dashboard and the prediction model updates and provides updated results automatically, without manually updating each day.
The prediction model uses time series data and converts that to supervised learning format, thus converting it to a regression-based problem. 
Validation is performed both on training and testing model. Validation on testing data follows an approach known as walk forward validation. 
Thus, this model is a combination of both supervised based training and time series-based validation on time series data. 
This project also compares the proposed prediction model with different hyper-parameters, activation functions and optimizers so as to get a view of the best performing model from all.
A comparison is also done with the proposed model and the model proposed in the base paper.
A dashboard is also created and deployed in a cloud-based platform Heroku so as to be available to everyone at any time. 
The dashboard illustrates various graphs that help doctors to better analyse and understand the present situation in an easier way.

PROCEDURE FOR RUNNING THE CODE:

DASHBOARD:
Visit the link to view the link (it might take some time to load):
https://india-world-covid.herokuapp.com/

Colab Link:
https://colab.research.google.com/drive/1QEfvvHz2v4qsc88X4Rpe60rtYUpcBiyk?usp=sharing
To see latest results:
go to runtime ->click run all  

PREDICTION:
Proposed model:
https://colab.research.google.com/drive/1uD0cYhJSJR5MdYTegDRfp4oOtlv4jUyN?usp=sharing
To see latest results:
go to runtime ->click run all  

Comparison between base paper and proposed model:
https://colab.research.google.com/drive/1nnUD4OR0KA3Cf-g9thboY6-rscUNLMDe?usp=sharing
Note: dataset used in this notebook is not an API hence load datasets trainBPpdf.csv and testBPpdf.csv into colab before running the cells.
Dataset is present in comaparision dataset folder.
 


