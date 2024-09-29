# DIABETES PREDICTION USING MACHINE LEARNING
# Overview
This project focuses on predicting the likelihood of diabetes in individuals using various machine learning algorithms. Diabetes is a chronic condition that affects millions of people worldwide, and early detection is critical for managing and preventing complications. By leveraging machine learning techniques such as Support Vector Machines (SVM), Random Forest, and XGBoost, this project aims to create a predictive model that can accurately classify whether a patient is likely to have diabetes based on their medical and lifestyle data.
# Objective

The primary objective of this project is to build a robust machine-learning model that can predict whether a person is likely to develop diabetes based on a set of input features, such as glucose levels, insulin levels, BMI, age, and other medical factors. Specifically, the project aims to:

1. Preprocess the dataset by handling missing values, normalizing features, and selecting the most relevant ones.
2. Train multiple machine learning models (SVM, Random Forest, and XGBoost) and evaluate their performance.
3. Compare the models based on metrics like accuracy, precision, and recall.

# Technologies Used
This project utilizes a variety of tools and libraries to build and evaluate machine learning models for diabetes prediction:

Python : The main programming language used for data analysis and model building.

Pandas : For data manipulation and preprocessing.

NumPy : For efficient numerical computations.

Scikit-learn : For implementing machine learning algorithms and evaluation metrics.

Matplotlib & Seaborn : For data visualization and plotting the results.

# Models Used
In this project, the following machine learning models were trained and compared to predict diabetes:

1. Support Vector Machine (SVM): A powerful classification algorithm that works well for high-dimensional spaces.
2. Random Forest Classifier: An ensemble learning method that combines multiple decision trees to improve predictive performance.
3. XGBoost: An optimized gradient boosting algorithm known for high accuracy and speed, especially for structured/tabular data.

# Methodology
The development of the diabetes prediction model follows a structured approach to ensure accuracy and robustness. Below are the key steps involved in the project:

# 1. Data Preprocessing:

- The Dataset was loaded and checked for missing or erroneous values.
- Features such as glucose levels, BMI, and insulin levels were normalized to ensure consistency across the dataset.
- Feature Selection : To improve model performance, statistical methods like F-scores and p-values were used to select the most relevant features that strongly contribute to predicting diabetes, reducing the dimensionality of the data and avoiding overfitting.

# 2. Splitting the Data :

- The dataset was split into training and testing sets to validate the performance of the models.

- K-Fold Cross-Validation was applied to ensure the models were robust and not overfitting to the training data.
  
# 3. Model Training:

Multiple machine learning algorithms were trained on the dataset, including:
- Support Vector Machine (SVM)
-  Random Forest Classifier
-   XGBoost
Each model was trained using the selected features and evaluated based on metrics like accuracy, precision, recall, and AUC-ROC.

# 4. Hyperparameter Tuning:

To optimize the performance of each model, GridSearchCV was used for hyperparameter tuning. This process involved searching for the best combination of hyperparameters (e.g., regularization strength in SVM, number of estimators in Random Forest, learning rate in XGBoost) that maximized the model's predictive accuracy.

# 5. Model Evaluation:

The models were evaluated on the test dataset, with metrics such as accuracy, precision, and recall.
A comparison of all models was done to determine which performed best based on overall classification performance.

# Conclusion
Thus among the 3 algorithms, the Random Forest algorithm provides the highest accuracy of 97.83 %.
