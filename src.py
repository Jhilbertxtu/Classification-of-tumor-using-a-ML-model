from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import random

knn = KNeighborsClassifier(n_neighbors=3)
breast_cancer_dataset = load_breast_cancer()
X_train,X_test,y_train,y_test = train_test_split(breast_cancer_dataset['data'],breast_cancer_dataset['target'],random_state=0)
knn.fit(X_train,y_train)

def showData():
    choice = 0
    while(choice != 7):
        print("Select the data to be displayed ...")
        print("1.Keys")
        print("2.Data")
        print("3.Target Classes")
        print("4.Target Names")
        print("5.Description Of The Dataset")
        print("6.Feature Names")
        print("7.Exit")
        choice = int(input("Enter your choice - "))
        if choice == 1:
            print("\n\n\n")
            print("KEYS - {}".format(breast_cancer_dataset.keys()))
            print("\n\n\n")
        elif choice == 2:
            print("\n\n\n")
            print("DATA - {}".format(breast_cancer_dataset['data']))
            print("\n\n\n")
        elif choice == 3:
            print("\n\n\n")
            print("TARGET CLASSES - {}".format(breast_cancer_dataset['target_names']))
            print("\n\n\n")
        elif choice == 4:
            print("\n\n\n")
            print("TARGET NAMES - {}".format(breast_cancer_dataset['target_names']))
            print("\n\n\n")
        elif choice == 5:
            print("\n\n\n")
            print("DESCRIPTION - {}".format(breast_cancer_dataset['DESCR']))
            print("\n\n\n")
        elif choice == 6:
            print("\n\n\n")
            print("FEATURE NAMES - {}".format(breast_cancer_dataset['feature_names']))
            print("\n\n\n")
        elif choice == 7:
            break

def determineSeverity(cancer):
    severity = knn.predict(cancer)
    print("The tumor in consideration is as follows...")
    for i in range(0,30):
        print("%s is of size %s"%(breast_cancer_dataset['feature_names'][i],cancer[0,i]))
        print("\n")
    print("Prediction class of the severity of the tumor - {}".format(severity))
    print("Prediction target name - {}".format(breast_cancer_dataset['target_names'][severity]))
def showAccuracy():
   test = knn.predict(X_test)
   print("Test Prediction Class - {}".format(test))
   print("\n")
   print("Test Prediction Target Name - {}".format(breast_cancer_dataset['target_names'][test]))
   print("\n")
   print("Test Prediction Accuracy - {:.2f}".format(np.mean( test == y_test)))
def main():
    choice = 0
    unknown_cancer = np.array([[random.uniform(6.981,28.11),random.uniform(9.71,39.28),
                                random.uniform(43.79,188.5),random.uniform(143.5,2501.0),
                                random.uniform(0.053,0.163),random.uniform(0.019,0.345),
                                random.uniform(0.0,0.427),random.uniform(0.0,0.201),
                                random.uniform(0.106,0.304),random.uniform(0.05,0.097),
                                random.uniform(0.112,2.873),random.uniform(0.36,4.885),
                                random.uniform(0.757,21.98),random.uniform(6.802,542.2),
                                random.uniform(0.002,0.031),random.uniform(0.002,0.135),
                                random.uniform(0.0,0.396),random.uniform(0.0,0.053),
                                random.uniform(0.008,0.079),random.uniform(0.001,0.03),
                                random.uniform(7.93,36.04),random.uniform(12.02,49.54),
                                random.uniform(50.41,251.2),random.uniform(185.2,4254.6),
                                random.uniform(0.071,0.223),random.uniform(0.027,1.058),
                                random.uniform(0.0,1.252),random.uniform(0.0,0.291),
                                random.uniform(0.156,0.664),random.uniform(0.055,0.208)]])
    while(choice != 4):
        print("***MAIN MENU***")
        print("1.Show Data-Set")
        print("2.Determine the severity of the cancer")
        print("3.Display accuracy of the ML model")
        print("4.Exit")
        choice = int(input("Enter your choice - "))
        if choice == 1:
            showData()
        elif choice == 2:
            ch = int(input("Do you want to... \n1. Enter your own data\n2.Let the computer randomly choose features\nEnter your choice - "))
            if ch == 1:
                user_cancer = input("Enter the features - ")
                determineSeverity(user_cancer)
            elif ch == 2:
                determineSeverity(unknown_cancer)
        elif choice == 3:
            showAccuracy()
        elif choice == 4:
            break
if __name__=="__main__":
    main()
