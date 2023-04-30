## Project Overview

This project contains two versions of the fake news classification model:

- Version 1: Uses deep learning techniques with tensorflow, keras, nltk libraries, and Sequential neural network and LSTM, achieving 92% accuracy.
- Version 2: Uses MultinomialNB model with bag of words, achieving 90% accuracy.



### version 1
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




### version 2
# Fake News Classification

This project aims to classify news articles as either real or fake using the Multinomial Naive Bayes model. The model has achieved a classification accuracy of 90% using bag of words and one hot encoder techniques.

## Dataset

The dataset used for this project is a collection of news articles obtained from various sources. The dataset contains a total of 20,000 news articles, out of which 5,000 are real news articles and the other 5,000 are fake news articles.

## Preprocessing

The first step in the preprocessing stage was to clean the text data by removing any unnecessary characters and symbols. The text was then tokenized into words and transformed into a bag of words representation. One hot encoder was also used to represent the text data in binary form.

## Model

The Multinomial Naive Bayes model was used for classification, which is a probabilistic algorithm that works by calculating the probability of each feature in the data belonging to a particular class. The model was trained on the preprocessed data and achieved an accuracy of 90%.

## Results

The results of the model show that it is able to classify news articles with 90% of accuracy, which is promising for future applications in the field of fake news detection.

## Future Work

In the future, more advanced techniques such as deep learning algorithms and natural language processing (NLP) techniques can be explored to improve the accuracy of the model. Additionally, the model can be trained on larger datasets to make it more robust and applicable in real-world scenarios.
