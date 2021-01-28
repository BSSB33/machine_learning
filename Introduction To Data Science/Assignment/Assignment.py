import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans

#Clearing console
clear = lambda: os.system('cls')
clear()

# Part 1
# Reading in the Data set
dataFrame = pd.read_csv("project_data.csv", 1, ";")

# Analyzing input
print(dataFrame.describe())

# isolating the target column from the dataset
target = dataFrame["Y"]
features = dataFrame.drop("Y", axis='columns')

# Counts the target Good and Bad customers
count_target = dataFrame.values[:,-1]
counter = Counter(count_target)
for group, count in counter.items():
	percentage = count / len(count_target) * 100
	print('Class=%d, Count=%d, Percentage=%.3f%%' % (group, count, percentage))
print('\n')

# categorizing data into two groups: The numerical data, and categorical data. With this two set we can make different operations
numerical_attributes = ["X02", "X08", "X11", "X13", "X16", "X18", "X05"]
categorical_attributes = ["X01", "X03", "X04", "X06", "X07", "X09", "X10", "X12", "X14", "X15", "X17", "X19", "X20"]

numerical_data = features[numerical_attributes]
categorical_data = features.drop(numerical_attributes, axis=1)

# Creating an ordinal encoder instance and using it to encode the categorical data. What this means is that for each different category it introduces an integer number and replaces it.
categorical_encoded_data = OrdinalEncoder().fit_transform(categorical_data)
dataFrame["Y"].replace({1 : 0, 2 : 1}, inplace=True)

# assembling pipeline for numerical data
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('min_max', MinMaxScaler())
])

# assembling pipeline to feed the data
full_pipeline = ColumnTransformer([
    ("num", numerical_pipeline, numerical_attributes),
    ("cat", OrdinalEncoder(), categorical_attributes),
])

# preparing pipeline
delay_prepared = full_pipeline.fit_transform(features)

# Training data
X_train, X_test, y_train, y_test = train_test_split(delay_prepared, target, test_size=0.3) #0.3

# Print the scores of the training
def printScores(name, model, X_train, X_test, y_train, y_test):
    print("================ ", name," ================")
    print("\nTrained score: ", model.score(X_train, y_train))
    print("Test score: ", model.score(X_test, y_test))

# Print the Reports of the prediciton
def printReport(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    # print("\nConfusion Matrix:")
    # print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("========================================================")

# Running for different models for comparison
def runModels():
    print("=============================== Running Models ===============================")
    # KNN - K Nearest Neighbors
    name_KNN = "K Nearest Neighbors"
    knn = KNeighborsClassifier(n_neighbors=50, leaf_size=300)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    printScores(name_KNN, knn, X_train, X_test, y_train, y_test)
    printReport(name_KNN, knn, X_test, y_test)

    # SVM - Support Vector Model
    name_SVM = "Support Vector Model"
    svc = SVC(C=50, gamma=10, kernel='linear')
    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_test)
    printScores(name_SVM, svc, X_train, X_test, y_train, y_test)
    printReport(name_SVM, svc, X_test, y_test)

    # LR - Logistic Regression
    name_LR = "Logistic Regression"
    lr = LogisticRegression(solver='liblinear')
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    printScores(name_LR, lr, X_train, X_test, y_train, y_test)
    printReport(name_LR, lr, X_test, y_test)

    # GPC - Gaussian Process Classifier
    name_GPC = "Gaussian Process Classifier"
    gpc = GaussianNB()
    gpc.fit(X_train, y_train)
    y_pred = gpc.predict(X_test)
    printScores(name_GPC, gpc, X_train, X_test, y_train, y_test)
    printReport(name_GPC, gpc, X_test, y_test) 

# Based on the above comparisons the best model is the SVM
def runGridSearchCV():
    print("=============================== Running SVC Variations ===============================")
    kernels = ['Polynomial', 'RBF', 'Sigmoid','Linear']
    def getClassifier(ktype):
        if ktype == 0:
            # Polynomial kernel
            return SVC(kernel='poly', degree=8, gamma="auto")
        elif ktype == 1:
            # Radial Basis Function kernel
            return SVC(kernel='rbf', gamma="auto")
        elif ktype == 2:
            # Sigmoid kernel
            return SVC(kernel='sigmoid', gamma="auto")
        elif ktype == 3:
            # Linear kernel
            return SVC(kernel='linear', gamma="auto")

    for i in range(4):
        svclassifier = getClassifier(i) 
        svclassifier.fit(X_train, y_train)
        y_pred = svclassifier.predict(X_test)
        print("Evaluation: ", kernels[i], " kernel")
        print(classification_report(y_test,y_pred))


    #If we take a even closer look we can use GridSearchCV to optimize and check for different parameters with different kernels.
    grid_params =  {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001],'kernel': ['rbf', 'poly', 'sigmoid']} #linear

    grid = GridSearchCV(SVC(), grid_params, verbose=1)
    grid.fit(X_train,y_train)

    print("Best parameters set found on development set:")
    print(grid.best_params_)

    print("Detailed classification report:\n")
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    y_true, y_pred = y_test, grid.predict(X_test)
    print(classification_report(y_true, y_pred))

    print("\nTrained score: ", grid.score(X_train, y_train))
    print("Test score: ", grid.score(X_test, y_test))

    # Calculating accuracy
    scores = cross_val_score(svclassifier, X_train, y_train, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    scores = cross_val_score(svclassifier, X_train, y_train, cv=5, scoring='f1_macro')
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


## Part 2
def runElbowMethod():
    print("=============================== Running Elbow Method ===============================")
    # Elbow method to determine the recommended amount of clusters
    # Calculate distortion for a range of number of cluster
    distortions = []
    for i in range(1, 15):
        km = KMeans(n_clusters = i, init = 'random', n_init = 10, max_iter = 300, tol = 1e-04, random_state = 0)
        km.fit(X_train)
        distortions.append(km.inertia_)
    # plot
    plt.plot(range(1, 15), distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.grid(axis="both")
    plt.show()

def printPlots():
    # Plot Number 1: Plotting of Bad and Good customers based on Age and Money in the bank
    # Conclusion: Right now we can't really see anyting important before clustering.
    ax = sns.scatterplot(data = dataFrame, x="X13", y="X05", sizes=(20,6), palette="tab10", hue=target)
    ax.set(xlabel='Age of the Customer', ylabel='Money in the bank')

    # Plot Number 2: Plotting of Bad and Good customers based on Age and Money in the bank
    # Conclusion: Customers are more likely to be judged as a bad customer if they have little reserves.
    ax = sns.scatterplot(data = dataFrame, x="X06", y="X05", sizes=(20,6), palette="tab10", hue=target)
    ax.set(ylabel='Money in the bank', xlabel='Category of the customer based on their money in the bank')

    # Plot Number 3: Plotting of Bad and Good customers based on whether they are foregin customers or not
    # Conclusion: Foregin workers often have more money in the bank.
    ax = sns.swarmplot(data = dataFrame, x="X05", y="X20", sizes=(20,6), palette="tab10", hue=target)
    ax.set(xlabel='Money in the bank', ylabel='Not Foregine Worker | Foregine worker')

    # Plot Number 4: Plotting of Bad and Good customers based on the Credit a customer has in the bank and the Duration of the credit (requested by the customer from the bank)
    # Conclusion: The less money a customer have, the less time they ask for a loan. (payback time)
    ax = sns.scatterplot(data = dataFrame, x="X05", y="X02", sizes=(20,6), palette="tab10", hue=target)
    ax.set(xlabel='Money in the bank', ylabel='Duration of the requested loan')


def runKMeansClustering1():
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(X_train)

    kmeans = KMeans(n_clusters = 2, init = 'random', n_init = 10, max_iter = 300, tol= 1e-04, random_state = 0)
    kmeans.fit(X_train)

    ax = sns.scatterplot(data=features.iloc[:700], x="X13", y="X05", sizes=(20,6), hue=kmeans.labels_, palette="tab10")
    ax.set(xlabel='Age', ylabel='Money in the Bank')
    ax.show()

def runKMeansClustering2():
    scaler = StandardScaler()
    scaled_separated = scaler.fit_transform(features[['X05', 'X13']])

    kmeans = KMeans(n_clusters = 4, init = 'random', n_init = 10, max_iter = 300, tol= 1e-04, random_state = 0)
    kmeans.fit(scaled_separated)

    ax = sns.scatterplot(data=features, x="X13", y="X05", sizes=(20,6), hue=kmeans.labels_, palette="tab10")
    ax.set(xlabel='Age', ylabel='Money in the Bank')

    
def frequentPatternMining():
    from fpgrowth_py import fpgrowth

    categorical_data = dataFrame[["X01", "X03", "X04", "X06", "X07", "X09", "X10", "X12", "X14", "X15", "X17", "X19", "X20", "Y"]]

    itemSetList = categorical_data.to_numpy()
    freqItemSet, rules = fpgrowth(itemSetList, minSupRatio=0.5, minConf=0.5)
    with open("outfile.txt", "w") as outfile:
        for line in rules:
            outfile.write("".join(str(line) + "\n"))
    print("Output Generated!")

if __name__ == "__main__":
    #runModels()
    #runGridSearchCV()
    #printPlots()
    #runElbowMethod()
    runKMeansClustering1()
    #frequentPatternMining()

### Resources

"""
- K Nearest Neighbors Project from Practice classes
- http://www.tryanalyticsblog.com/svm-classification-credit/
- https://towardsdatascience.com/k-means-clustering-of-university-data-9e8491068778
- https://www.vebuso.com/2020/03/svm-hyperparameter-tuning-using-gridsearchcv/
- https://financetrain.com/case-study-german-credit-steps-to-build-a-predictive-model/
- https://machinelearningmastery.com/imbalanced-classification-of-good-and-bad-credit/?fbclid=IwAR3rzTGDEqAy-ufDhDXXb_igidiK_k976MBGtuUEZgLc1CBlOjhrPlaX1FQ
- https://www.kaggle.com/hendraherviawan/predicting-german-credit-default 
- https://machinelearningmastery.com/one-hot-encoding-for-categorical-data/
- https://medium.com/@stallonejacob/data-science-scaling-of-data-in-python-ec7ad220b339
- https://towardsdatascience.com/fp-growth-frequent-pattern-generation-in-data-mining-with-python-implementation-244e561ab1c3"""
