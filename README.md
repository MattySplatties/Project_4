# Project_4: Evaluating reviews of businesses using ML

# Background
The scope of our project is to use a machine learning model to assess the language used by humans when reviewing businesses. These reviews are ran through a trained model and the goal was to see whether or not the model could accurately assess the review as either a positive or negative review, based on the language used by the reviewer.

# Software/Toools:

* Python
* Bokeh
* Pandas
* Excel
* Google/Yelp API
* Numpy
* Nltk
* Pickle
* SKlearn

# Intructions
The assignment is broken down into the following:

1. Obtain Data to train model
    *Initial trail emotion dataset
    * https://www.kaggle.com/datasets/nelgiriyewithana/emotions 
    * Twitter emotion dataset
    * https://www.kaggle.com/code/mayank404/twitter-sentimental-analysis
    * Yelp Review Dataset
    * https://www.kaggle.com/datasets/yacharki/yelp-reviews-for-sa-finegrained-5-classes-csv

2. Initializes, trains and evaluates model
    * Training of MultiModelVector and LRTrained2 model
    * Cleaning of the training Dataset
    * ![alt text](image-1.png)
    * Setting up vectorizer 
    * ![alt text](image.png)
    * Function used to asses the accuracy of model and show confusion matrix 
    * ![alt text](image-2.png)

3. Extract Data, Clean and Normalize
    * Using google and yelp API to extract the Data used to evauluate the model
    * ![alt text](image-6.png)
    * Limitations faced while pulling data from API's

4. Model Accuracy / Optimization
    * Below is the accuracy of the model used on the large Dataset it was trained on
    * ![alt text](image-3.png)
    * Below is an instance of the model being used to evaluate reviews at the local Chili's here in Nashville on West End
    * ![alt text](image-4.png)
    * ![alt text](image-5.png)
    Findings: When using a real life example to evaluate the model, it predicited the correct sentiment of the review (positive or negative) with a 90% accuracy when looking at the Chili's on West End
    * Tailoring what stopwords where included to better increase accuracy of model (n't words)

# Ethics
Within this assignment, the only ethical dilemmas that were faced were pertaing to using the API's from google and yelp to obtain reviews. We were intially hoping for a rather large set of data to evaluate the model all at once, however both yelp and google limit the number of reviews one can access. We thought of trying to obtain more reviews through web scraping, however both websites outlaw against scraping reviews for any use. 

#Refrences:
1. https://towardsdatascience.com/basics-of-countvectorizer-e26677900f9c

2. https://www.geeksforgeeks.org/xgboost-for-regression/

3. https://scikit-learn.org/stable/modules/generat.ed/sklearn.feature_extraction.text.CountVectorizer.html

4. https://docs.getml.com/latest/api/getml.predictors.XGBoostClassifier.html

5. https://stackoverflow.com/questions/22851316/what-is-the-inverse-of-regularization-strength-in-logistic-regression-how-shoul

6. https://www.statology.org/logistic-regression/

7. https://www.geeksforgeeks.org/introduction-to-stemming/

8. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

9. https://stackoverflow.com/questions/38640109/logistic-regression-python-solvers-definitions

10. https://www.geeksforgeeks.org/naive-bayes-classifiers/

11. https://www.britannica.com/science/multinomial-distribution

12. https://towardsdatascience.com/everything-about-svm-classification-above-and-beyond-cc665bfd993e

Data:

1. https://www.kaggle.com/datasets/kazanova/sentiment140

2. https://www.kaggle.com/datasets/yacharki/yelp-reviews-for-sa-finegrained-5-classes-csv

3. https://www.kaggle.com/datasets/nelgiriyewithana/emotions


Tutorials:

1. https://www.youtube.com/watch?v=4YGkfAd2iXM (Twitter Sentiment Analysis By GeeksforGeeks)

Plotting:

#https://matplotlib.org/stable/tutorials/pyplot.html#sphx-glr-tutorials-pyplot-py

https://www.kaggle.com/code/nelgiriyewithana/an-introduction-to-emotions-dataset

