# Sentiment Analysis of YouTube Comments
A sentiment analysis web application that fetches and processes YouTube video comments and then classifies them as 
positive or negative. When analyzing comments, the user can choose from a list of machine learning models/classifiers such as Logistic Regression, 
XGBoost, Random Forest, Support Vector Machine, Multinomial Naive Bayes and LSTM Deep Learning.  Built with Python,
Flask, and integrated with the YouTube API, the app provides a user-friendly interface to visualize sentiment reports. 

## Objective

This project aims to demonstrate the implementation of multiple machine learning models and a deep learning model in a 
sentiment analysis pipeline, rather than to compare their performance exhaustively. The goal is to showcase how various 
algorithms can be integrated into a single application to give users the flexibility to choose a model based on their 
preference or specific use case. It is an educational tool for understanding the deployment of models in real-world 
applications like sentiment analysis on YouTube comments.

## Technologies and modules:

- python3
- flask
- joblib
- tensorflow
- xgboost
- scikit-learn

Some modules are already part of the python3 install package. 
Other modules/packages can be installed through `pip install package_name` or `pip3 install package_name` .


## Project features:

1. Sentiment Classification - Classifies comments into positive or negative sentiments using multiple machine learning models.
2. Custom Model Selection - Allows users to choose from different classifiers for sentiment analysis, including traditional machine learning and deep learning options.
3. Comment Preprocessing - Cleans and preprocesses YouTube comments by removing URLs, emojis, mentions, and other noise.
4. Real-Time Sentiment Summary - Provides an aggregated report showing the overall sentiment based on the comments fetched.
5. Error Handling - Manages API request errors, network issues, and invalid video URLs to ensure a smooth user experience.
6. UI Customization Options - Includes options like dark mode, customizable comment display limit, and display summary for a user-friendly interface.

## Installation:
1. Download or clone the repository from GitHub:

```bash
   git clone https://github.com/tuobaar/sentiment_analysis.git
```
- Alternatively, you can [download the zip file from here](https://github.com/tuobaar/sentiment_analysis/archive/refs/heads/main.zip) and extract it to your preferred location.

2. Open `command console/prompt` or `terminal`, navigate to the location of the extracted folder/files and install all required packages in `requirements.txt`:

```bash
   pip install -r requirements.txt  # use pip3 if needed
````

## Setting up environment variables
1. Rename `.env.example` to `.env`.
2. Open `.env` and set up the required credentials, including your YouTube API Key.
3. Save the `.env` file to load the environment variables when the application runs.

## Setting the API URL for the UI
1. Open the `templates` directory of the project and edit `frontend.html`.
2. Set line 223 to: const apiUrl = 'http://127.0.0.1:5000';

## Usage:

1. The Flask app can be launched by entering the following in a `command console/prompt` or `terminal`:
```bash
   python app.py  # use python3 if needed
```

2. Accessing the Web Application (Locally)

- Open a web browser and go to http://localhost:5000 to access the application interface.

3. Using the Interface Locally
   
- Enter a YouTube video URL / ID to fetch comments. You can also enter your own comments/sentences.
- Select the machine learning model you want to use for classification.
- Click `Analyze` to generate a sentiment report based on the chosen model.

4. Using the Deployed App on [Render](https://www.render.com)

   The app is also deployed on Render and can be accessed online. 
   To use the app, simply click on this [link](https://sentiment-analysis-4nt1.onrender.com/).

- Enter a YouTube video URL to fetch comments.
- Choose a machine learning model for sentiment analysis.
- Click Analyze to get the sentiment report based on your selected model.


      Performance Note on Render:
      When there is inactivity on the app for some time,Render may take longer to load the app initially. 
      This is due to the app being "sleeped" when idle. Once the app is loaded, it will function normally, 
      and you can continue using it for sentiment analysis.`

- The deployed version offers the same functionality as the local version, with the added benefit of being accessible 
from anywhere with any internet enabled device without having to run it locally

## Additional Features

- Error Handling: The app handles invalid URLs, network errors, and API errors to ensure stability.
- UI Display Limits: To avoid bloating the interface, the app limits the number of displayed comments while maintaining an accurate summary.
- Dark Mode: An optional dark mode is available for improved user experience.

## Troubleshooting
If the application doesn't start:

- Ensure all dependencies are installed from `requirements.txt`.
- Check the YouTube API Key in the `.env` file is valid and has the necessary permissions.

## License
This project is licensed under the MIT License.

## Extras

In the `extras` directory, users can explore code and notebooks for training the models and preprocessing the data. 
Since most of the work was done on [Kaggle](https://www.kaggle.com), some of the model training made use of GPUs. 
The models were trained on the [Sentiment140 dataset](https://www.kaggle.com/datasets/kazanova/sentiment140), which consists of positive and negative sentiment-labeled Twitter comments. 
This directory allows users to see the training configurations and preprocessing applied and modify the code for their 
own model training if they wish.
