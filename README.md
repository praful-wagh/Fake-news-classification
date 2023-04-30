## Project Overview

This project contains three versions of the fake news classification model:


- Version 1: Uses MultinomialNB model with bag of words, achieving 90% accuracy.
- Version 2: Uses deep learning techniques with tensorflow, keras, nltk libraries, and Sequential neural network and LSTM, achieving 92% accuracy.
- Version 3: Uses LGBMClassifier model with TfidfVectorizer module, achieving 98% accuracy.


### version 1

# Fake News Classification

This project aims to classify news articles as either real or fake using the Multinomial Naive Bayes model. The model has achieved a classification accuracy of 90% using bag of words and one hot encoder techniques.

## Dataset

The dataset used in this project is a collection of news articles that have been labeled as fake or real. The dataset contains a total of 20,000 articles, evenly split between fake and real news articles.

## Preprocessing

The first step in the preprocessing stage was to clean the text data by removing any unnecessary characters and symbols. The text was then tokenized into words and transformed into a bag of words representation. One hot encoder was also used to represent the text data in binary form.

## Model

The Multinomial Naive Bayes model was used for classification, which is a probabilistic algorithm that works by calculating the probability of each feature in the data belonging to a particular class. The model was trained on the preprocessed data and achieved an accuracy of 90%.

## Results

The results of the model show that it is able to classify news articles with 90% of accuracy, which is promising for future applications in the field of fake news detection.

## Future Work

In the future, more advanced techniques such as deep learning algorithms and natural language processing (NLP) techniques can be explored to improve the accuracy of the model. Additionally, the model can be trained on larger datasets to make it more robust and applicable in real-world scenarios.




### version 2

# Fake News Classification with Deep Learning

The goal of this project is to classify news articles as real or fake using deep learning techniques. We will be using TensorFlow, Keras, and NLTK libraries to build a Sequential Neural Network (SNN) with Long Short-Term Memory (LSTM) to classify the news articles.

## Dataset

We used a publicly available dataset which contains a collection of real and fake news articles. This dataset has approximately 20,000 news articles. The dataset was split into 80% for training and 20% for testing.

## Preprocessing

The data was preprocessed using the NLTK library. The text data was tokenized, cleaned and converted to lowercase. We also removed stop words, punctuation and numbers. We then used the GloVe word embeddings to convert the text data into vectors.

## Model

We used a Sequential Neural Network (SNN) with LSTM layers to classify the news articles. The model was trained on the preprocessed data using binary cross-entropy loss function and Adam optimizer.

## Results

The model achieved an accuracy of 92% on the testing data. The model was able to classify the news articles with high precision and recall. This model outperformed the MultinomialNB model that was built using bag of words technique which achieved 90% accuracy.

## Future Work

In the future, we plan to experiment with different pre-processing techniques and different deep learning architectures to improve the performance of the model. We also plan to deploy the model to a web application to classify news articles in real-time.





### version 3

# Fake News Classification using LGBMClassifier

This project is focused on classifying fake news using the LGBMClassifier model. We have achieved an accuracy of 98% with this model. The project involves preprocessing the data and extracting features using the TfidfVectorizer module from sklearn. We have also used the nltk module for tokenizing and stemming the text.

## Dataset

The dataset used in this project is a collection of news articles that have been labeled as fake or real. The dataset contains a total of 20,000 articles, evenly split between fake and real news articles.

## Methodology

The project involves the following steps:

1. Data Preprocessing: The raw data is cleaned and preprocessed by removing stop words, stemming, and tokenizing.

2. Feature Extraction: The TfidfVectorizer module is used to extract features from the preprocessed data. This module converts text into numerical vectors, which can be used as input for machine learning models.

3. Model Training: The LGBMClassifier model is trained on the preprocessed and feature extracted data to classify fake news articles.

4. Model Evaluation: The model is evaluated using the test data, and the accuracy score is calculated.

## Libraries Used

- LGBMClassifier
- nltk
- sklearn
- TfidfVectorizer

## Conclusion

We have successfully classified fake news articles with an accuracy of 98% using the LGBMClassifier model. The project demonstrates the importance of feature engineering and the effectiveness of the LGBMClassifier model in text classification tasks.
