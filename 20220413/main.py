from sklearn.datasets import fetch_lfw_people
import os
from rich.console import Console
from sklearn.model_selection import train_test_split,GridSearchCV
from utils import *
from sklearn.decomposition import PCA
import time
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from rich.table import Table

console=Console()
console.rule("[bold red] Download data")
if not os.path.exists("dataset"):
    console.log("You don't have dir! So we create one dir for you!")
    os.mkdir("./dataset/")

download=True

if os.path.exists("./dataset/lfw_home"):
    console.log("Oh! You have data yeah. Geart! :smiley:")
    download=False

lfw_people = fetch_lfw_people(data_home="./dataset/", min_faces_per_person=100,download_if_missing=download)
console.rule("[bold red] View the data")
console.log("Let's Look the information of the data o.O :vampire:")
n,h,w = lfw_people.images.shape
hw = lfw_people.data.shape
lc=lfw_people.target.shape
target_names = lfw_people.target_names
table = make_table(target_names,target_names)
console.log(table)
# console.log("the target name is {}".format(target_names))
n_classes = target_names.shape[0]
console.log("the num of the class is {}".format(n_classes))
console.log("data shape is {} label shape is {} :raccoon:".format(hw,lc))

console.rule("[bold red] Data split")
X_train, X_test, y_train, y_test = train_test_split(lfw_people.data, lfw_people.target, test_size=0.25)
console.log("the shape of the train dataset is {}".format(len(X_train)))
console.log("the shape of the test dataset is {}".format(len(X_test)))
console.rule("[bold red] Draw picture")
t0 = time.time()
plot_img(X_train[0:12],y_train[0:12],ing_name="first")
console.log("Look the photo in dir")
console.log("done in %0.3fs" % (time.time()- t0))
console.rule("[bold red] Data preprocessing")
pca = PCA(n_components=150, svd_solver='randomized',
          whiten=True).fit(X_train)
eigenfaces = pca.components_.reshape((150, h, w))
console.log("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time.time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
console.log("done in %0.3fs" % (time.time()- t0))
console.rule("[bold red] Start train SVM")
console.log("Fitting the classifier to the training set")
t0 = time.time()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf = clf.fit(X_train_pca, y_train)
console.log("done in %0.3fs" % (time.time() - t0))
console.log("Best estimator found by grid search:")
console.log(clf.best_estimator_)
console.rule("[bold red] Start test")
console.log("Names Prediction")
t0 = time.time()
y_pred = clf.predict(X_test_pca)
console.log("Action completed in %0.3fs" % (time.time() - t0))
console.log(classification_report(y_test, y_pred, target_names=target_names))
console.log(confusion_matrix(y_test, y_pred, labels=range(n_classes)))
console.rule("[bold red] plot the gallery of the most significative eigenfaces")
t0 = time.time()
eigenface_titles = ["Eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_img(eigenfaces, eigenface_titles,ing_name="second")
console.log("make the most significative eigenfaces picture!")
console.log("done in %0.3fs" % (time.time()- t0))
console.rule("[bold red] Look the result")
t0 = time.time()
prediction_titles = [title(y_pred, y_test, target_names, i)
                     for i in range(y_pred.shape[0])]
plot_img(X_test, prediction_titles,ing_name="thrid")
console.log("make result picture!")
console.log("done in %0.3fs" % (time.time()- t0))
# plt.show()