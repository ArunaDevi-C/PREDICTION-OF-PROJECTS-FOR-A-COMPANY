from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Step 9: Function to load the model and make predictions
def predict_future_projects(client_name, industry, year):
    # Create a DataFrame for the new input
    input_data = pd.DataFrame([[client_name, industry, year]], columns=['Client Name', 'Industry', 'Year'])
    
    # Load the saved model
    loaded_pipeline = joblib.load('trained_linear_regression_model.pkl')
    
    # Make predictions
    predicted_projects = loaded_pipeline.predict(input_data)
    
    # Return the predicted number of projects as an integer
    return int(predicted_projects[0])

# Step 11: Flask routes

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get form input
        client_name = request.form['client_name']
        industry = request.form['industry']
        year = int(request.form['year'])  # Convert year to integer
        
        # Get the prediction
        predicted_projects = predict_future_projects(client_name, industry, year)
        
        # Display the result
        return render_template('result.html', client_name=client_name, industry=industry, year=year, prediction=predicted_projects)
    
    # If method is GET, display the prediction form
    return render_template('predict.html')

# Step 12: Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
