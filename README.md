# Text Emotion Classifier
This project implements a Text Emotion Classifier using Streamlit, a Python library for building interactive web applications, and a Logistic Regression model trained on preprocessed text data.

### Overview
The Text Emotion Classifier allows users to input text and predicts the emotion associated with the input text. Emotions recognized by the classifier include anger, disgust, fear, happy, joy, neutral, sad, sadness, shame, and surprise.

### Installation
1. Clone the repository:
  `git clone <repository-url>`
2. Navigate to the project directory:
  `cd text-emotion-classifier`
3. Install the required Python packages:
   `pip install -r requirements.txt`
4. Run the Streamlit application:
   `streamlit run app.py`

### Usage
Once the application is running, users can access it via a web browser. They can input text into the provided text area and click the "Predict" button to obtain the predicted emotion along with the associated probability distribution.

### Model
The model used in this project is a Logistic Regression classifier trained on preprocessed text data. The preprocessing steps include data cleaning and feature extraction. The model is trained to classify text into one of the predefined emotion categories.

### Files
* `app.py`: The main Python script containing the Streamlit application.
* `models/text_emotion.pkl`: Serialized model file containing the trained Logistic Regression classifier.
* `requirements.txt`: List of required Python packages for installation.

