import pickle
import numpy as np
from flask import Flask, render_template, request    #importing Flask and its features
#In Flask, the `Flask` class is used to create a web application instance. 
#The `static_url_path` parameter sets the URL path for serving static files (like CSS and JavaScript), and the `template_folder` parameter specifies the folder where Flask will look for HTML templates.
app = Flask(__name__,static_url_path='/static',template_folder='templates')  

#loading the model from pickle in read format
model=pickle.load(open('model.pkl','rb'))

# Define a route for the home page
#(@app.route('/')) defines a route for the default URL path ('/') 
@app.route('/')
def index():
    return render_template('index.html')  #this will return to index.html page whenever we click on flask url(root url)

# Define a route for handling predictions
@app.route('/predict', methods=['POST','GET']) #this is to handle predict url
#GET renders form or provides info and POST used to submit the form to predicition model and returns the result
def predict():
    float_features=[float(x) for x in request.form.values()] #in for loop all the data from form is collected and is converted to float
    final=[np.array(float_features)] #the data is made as a 2d array
    prediction=model.predict(final)  #model to predict and result stored in a variable predict
    
    return render_template('index.html', pred="The crop that can be grown is: {}".format(prediction[0])) #the result would be displayed on html file 

if __name__ == '__main__':
    app.run(debug=True)
#If the script is the main program (i.e., it's not imported into another script), the code is executed.