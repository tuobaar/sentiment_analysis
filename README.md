# Sentiment Analysis of YouTube Comments
A sentiment analysis web application that fetches and processes YouTube video comments and then classifies them as 
positive or negative. When analyzing comments, the user can choose from a list of machine learning models/classifiers such as Logistic Regression, 
XGBoost, Random Forest, Support Vector Machine, Multinomial Naive Bayes and LSTM Deep Learning.  Built with Python,
Flask, and integrated with the YouTube API, the app provides a user-friendly interface to visualize sentiment reports. 


## Technologies and modules:

- python3
- flask
- joblib
- tensorflow
- xgboost
- scikit-learn

Some modules are already part of the python3 install package. Other modules/packages can be installed through `pip install package_name` or `pip3 install package_name` .


## Project features:

1. Sentiment Classification - Classifies comments into positive or negative sentiments using multiple machine learning models.
2. Custom Model Selection - Allows users to choose from different classifiers for sentiment analysis, including traditional machine learning and deep learning options.
3. Comment Preprocessing - Cleans and preprocesses YouTube comments by removing URLs, emojis, mentions, and other noise.
4. Real-Time Sentiment Summary - Provides an aggregated report showing the overall sentiment based on the comments fetched.
5. Error Handling - Manages API request errors, network issues, and invalid video URLs to ensure a smooth user experience.
6. UI Customization Options - Includes options like dark mode, customizable comment display limit, and display summary for a user-friendly interface.

## Installation:
1. Download or clone the repository from GitHub:

   '>>> git clone https://github.com/tuobaar/sentiment_analysis.git'

    Or [download](https://github.com/tuobaar/sentiment_analysis/archive/refs/heads/main.zip) the zip file and extract it to your preferred location.

2. Open `command console/prompt` or `terminal`, navigate to the location of the extracted folder/files and install all required packages in `requirements.txt`:

   `>>> pip install -r requirements.txt`   
   `>>> pip3 install -r requirements.txt`


## Setting up environment variables
1. Rename `.env.example` to `.env`.
2. Open `.env` and set up the required credentials, including your YouTube API Key.
3. Save the `.env` file to load the environment variables when the application runs.


## Usage:

1. The Flask app can be launched by entering the following in a `command console/prompt` or `terminal`:

   `>>> python app.py`  
   `>>> python3 app.py`

2. Access the Web Application

   Open a web browser and go to http://localhost:5000 to access the application interface.

3. Using the Interface
   
- Enter a YouTube video URL to fetch comments.
- Select the machine learning model you want to use for classification.
- Click Analyze to generate a sentiment report based on the chosen model.

## Additional Features

- Error Handling: The app handles invalid URLs, network errors, and API errors to ensure stability.
- UI Display Limits: To avoid bloating the interface, the app limits the number of displayed comments while maintaining an accurate summary.
- Dark Mode: An optional dark mode is available for improved user experience.

## Troubleshooting
If the application doesn't start:

- Ensure all dependencies are installed from requirements.txt.
- Check the YouTube API Key in the `.env` file is valid and has the necessary permissions.

## License
This project is licensed under the MIT License.