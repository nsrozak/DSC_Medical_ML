# Purpose of this Project

I used machine learning to predict whether a patient has had a stroke, and I used results from my models and exploratory data analysis to determine which variables are important predictors. My motivation for doing this project was to apply the skills I learned in my machine learning course at UCSB and learn new topics related to data cleaning. Note that I did not use this project in any of my courses at UCSB.

# Steps to Complete This Project

1. Clean data
2. Visualize training data
3. Implement classification machine learning models
  - Logistic regression
  - Ridge regression
  - Lasso regression
  - K nearest neighbors
  - Decision tree
  - Bagged tree
  - Random forest
  - Adaboost
  - Support vector machine
4. Analyze results

# Conclusions

My Adaboost model had the lowest misclassification error rate, so I recommend using this model to predict whether a patient has had a stroke. Important predictors for determining whether someone has had a stroke include a person's age, whether they have had heart disease, their average glucose level, and their BMI. 

# Ideas to Improve the Project

The models I built had misclassification rates around 28%, and I think I can decrease this rate if I improve my data cleaning techniques. This was my first time working with imbalanced classes, and I used a simple method of downsampling my data to correct this issue. However, a better technique is creating artificial data to increase the sample size of stroke patients.
