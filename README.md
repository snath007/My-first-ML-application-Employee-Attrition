# My-first-ML-application-Employee-Attrition
The intention of this project is to build a simple API that is in the form of a web application. The application takes various parameters of an employee such as working hours, salary, satisfaction level etc. and the ML algorithm that is present at the back end, predicts whether the employee will stay or leave the company.
Prerequisites (python libraries and frameworks):
Pandas (for Machine Leraning Model), Scikit Learn, Flask (for API)

Project files involved:
Model.py : This file contains the code which fits the input data in a ML algorithm and predicts the class which the employee belongs to. ‘1’ if the employee stays, ‘0’ if the employee leaves the company. The pickle function converts the code into a serialized object.
Flask.py : This file contains the code which first imports the serialized object (converted through pickle previously). The input values taken from the user by requests.form and sends the predicted outcome for rendering to GUI.html.
GUI.html: This file is the frontend html page which acts as an interface to the user where the input parameters are taken and the predicted result is displayed.

The details of the project has been mentioned in the report attached in the repository
