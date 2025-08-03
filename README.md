Project Overview

This repository provides a MATLAB implementation of a sentiment analysis classifier for tweets using the k-Nearest Neighbors (k-NN) algorithm. The goal is to classify tweets as either positive or negative based on their textual content.

The project demonstrates a complete classical NLP workflow:

Data Preprocessing: Cleaning raw text data by removing URLs, hashtags, mentions, and stop words, followed by text stemming.

Feature Extraction: Converting the cleaned text into a numerical format using the Term Frequency-Inverse Document Frequency (TF-IDF) method.

Model Training: Training a k-NN classifier on the processed data.

Evaluation: Assessing the model's performance using metrics like accuracy and a confusion matrix.

How to Run

Open the project folder in Python.

Ensure the Twitter_Data.csv file is in the same directory.

Open the file.

Click the "Run" button or press F5.

The script will process the dataset, train the k-NN model, and output the classification accuracy on the test set. It will also display a confusion matrix to visualize the model's performance.
