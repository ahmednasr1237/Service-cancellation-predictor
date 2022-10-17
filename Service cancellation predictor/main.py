import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression
dataset = pd.read_csv('D:\\AI Project\\Service cancellation predictor\\CustomersDataset.csv')
dataset = dataset.replace({
    'gender': {'Male': 0, 'Female': 1},
     'Partner': {'Yes':0,'No':1},
    'Dependents': {'Yes': 0, 'No': 1},
   'PhoneService' : {'Yes': 0, 'No': 1},
   'MultipleLines': {'Yes': 0, 'No': 1,'No phone service':2},
   'InternetService': {'DSL': 0, 'Fiber optic': 1,'No':2},
   'OnlineSecurity': {'Yes': 0, 'No': 1,'No internet service':2},
   'OnlineBackup': {'Yes': 0, 'No': 1,'No internet service':2},
   'DeviceProtection': {'Yes': 0, 'No': 1,'No internet service':2},
   'TechSupport': {'Yes': 0, 'No': 1,'No internet service':2},
   'StreamingTV': {'Yes': 0, 'No': 1,'No internet service':2},
   'StreamingMovies': {'Yes': 0, 'No': 1,'No internet service':2},
   'Contract': {'Month-to-month': 0, 'One year': 1,'Two year':2},
   'PaperlessBilling': {'Yes': 0, 'No': 1},
   'PaymentMethod': {'Electronic check': 0, 'Mailed check': 1,'Bank transfer (automatic)':2,'Credit card (automatic)':3},
    'Churn': {'Yes': 0, 'No': 1},

})

"""
dataset.drop('gender',axis=1,inplace=True)
dataset.drop('SeniorCitizen',axis=1,inplace=True)
dataset.drop('Partner',axis=1,inplace=True)
dataset.drop('Dependents',axis=1,inplace=True)
dataset.drop('tenure',axis=1,inplace=True)
dataset.drop('MultipleLines',axis=1,inplace=True)
dataset.drop('OnlineBackup',axis=1,inplace=True)
dataset.drop('DeviceProtection',axis=1,inplace=True)
dataset.drop('StreamingTV',axis=1,inplace=True)
dataset.drop('StreamingMovies',axis=1,inplace=True)
dataset.drop('Contract',axis=1,inplace=True)
dataset.drop('PaperlessBilling',axis=1,inplace=True)
dataset.drop('PaymentMethod',axis=1,inplace=True)

x = dataset.iloc[:, 0:6].values
y = dataset.iloc[:, 6].values



from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression
thismodel = LinearRegression()

FeatureSelection = SelectFromModel(estimator = thismodel, max_features = None)
x = FeatureSelection.fit_transform(x, y)
print(FeatureSelection.get_support())
"""

x = dataset.iloc[:, 0:19].values
y = dataset.iloc[:, 19].values

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
x = scaler.fit_transform(x)


from sklearn.decomposition import PCA
pca =PCA(n_components=5)
pca.fit_transform(x)


from sklearn.model_selection import train_test_split
X_train , X_test,Y_train,Y_test= train_test_split(x,y,test_size=0.2,random_state=50,shuffle=True)


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

def testknn ():
    print("Accurcy of KNN :")
    print(Accurcy)
    print("Confusion Matrix of KNN :")
    print(CM)
    sns.heatmap(CM)
    plt.show()
def trainknn ():
    print("accurcy of training (KNN):")
    print(KNNModel.score(X_train, Y_train))

KNNModel = KNeighborsClassifier(n_neighbors=100, weights='distance', algorithm='brute',leaf_size=20)
KNNModel.fit(X_train, Y_train)
y_pred = KNNModel.predict(X_test)
CM=confusion_matrix(Y_test,y_pred)
Accurcy = accuracy_score(Y_test, y_pred)

def testbayes ():
    print("Accurcy of NaiveBayes :")
    print(Accurcy2)
    print("Confusion+ Matrix of NaiveBayes :")
    print(CM2)
    sns.heatmap(CM2)
    plt.show()
    print("******")

def trainbayes():
    print("accurcy of training (NaiveBayes):")
    print(MultiModel.score(X_train, Y_train))


#TASK NAIVE BAYES
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
MultiModel = MultinomialNB()
MultiModel.fit(X_train, Y_train)
y_pred2=MultiModel.predict(X_test)
CM2=confusion_matrix(Y_test,y_pred2)
Accurcy2=accuracy_score(Y_test,y_pred2)

def testtree():
    print("Accurcy of DecisionTree :" )
    print(Accurcy3)
    CM3 = confusion_matrix(Y_test, y_pred3)
    print("Confusion Matrix of DecisionTree :")
    print(CM3)
    sns.heatmap(CM3)
    plt.show()
def traintree():
    print("accurcy of training (DecisionTree):")
    print(Decitree.score(X_train, Y_train))





from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

Decitree=DecisionTreeClassifier(criterion='gini', max_depth=20, random_state=50, min_samples_split=5,max_features=6,max_leaf_nodes=10)
Decitree.fit(X_train,Y_train)
y_pred3=Decitree.predict(X_test)
Accurcy3=accuracy_score(Y_test,y_pred3)

def trainlog():
    print("accurcy of training (LogisticRegression):")
    print(LogisticRegressionModel.score(X_train, Y_train))


def testlog():
    print("Accurcy of LogisticRegression :" )
    print(Accurcy4)
    CM4 = confusion_matrix(Y_test, y_pred4)
    print("Confusion Matrix of LogisticRegression :")
    print(CM4)
    sns.heatmap(CM4)
    plt.show()

from sklearn.linear_model import LogisticRegression
LogisticRegressionModel = LogisticRegression(penalty='l1', solver='liblinear', C=50,tol=0.001, max_iter=1000000000)
LogisticRegressionModel.fit(X_train, Y_train)
y_pred4 = LogisticRegressionModel.predict(X_test)
Accurcy4=accuracy_score(Y_test,y_pred4)

def trainsvm():
    print("accurcy of training (SVM):")
    print(SVCModel.score(X_train, Y_train))
def testsvm() :
    print("Accurcy of SVM :")
    print(Accurcy5)
    CM5 = confusion_matrix(Y_test, y_pred5)
    print("Confusion Matrix of SVM :")
    print(CM5)
    sns.heatmap(CM5)
    plt.show()

from sklearn.svm import SVC
SVCModel = SVC(kernel='linear', max_iter=1000000, C=5)
SVCModel.fit(X_train, Y_train)
y_pred5 = SVCModel.predict(X_test)
Accurcy5=accuracy_score(Y_test,y_pred5)



def trainforest():
    print("accurcy of training (Random Forest):")
    print(RandomForestModel.score(X_train,Y_train))


def testforest():
    print("Accurcy of Random Forest :")
    print(Accurcy6)
    CM6 = confusion_matrix(Y_test, y_pred6)
    print("Confusion Matrix of Random Forest:")
    print(CM6)
    sns.heatmap(CM6)
    plt.show()

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
RandomForestModel = RandomForestClassifier(criterion = 'gini',n_estimators=100,max_depth=10)
RandomForestModel.fit(X_train, Y_train)
y_pred6=RandomForestModel.predict(X_test)
CM6=confusion_matrix(Y_test,y_pred6)
Accurcy6=accuracy_score(Y_test,y_pred6)


from tkinter import *
from tkinter import ttk
root=Tk()
root.title('Service Cancellation Predictor')
root.geometry('800x400')
root.maxsize(900,800)
root.config(bg='#00818A')
root.minsize(400,200)
fr1=Frame(root)
def test():    #1cond
    if (var1.get() == 1 ):
        testlog()



    elif (var1.get() == 2 ):
        testsvm()



    elif (var1.get() == 3 ):
        testtree()

    elif (var1.get() == 4 ):
        testknn()


    elif (var1.get() == 5 ):
        testbayes()


    elif (var1.get() == 6 ):
        testforest()



def train():
    if (var1.get() == 1):
        trainlog()



    elif (var1.get() == 2):
        trainsvm()



    elif (var1.get() == 3):
        traintree()



    elif (var1.get() == 4):
        trainknn()



    elif (var1.get() == 5):
        trainbayes()


    elif (var1.get() == 6):
        trainforest()


def check():
    v=var7.get()
    if (v==1):
     test()
    elif (v==2):
        train()

def algo():
    t2 = e2.get()
    if t2 == 'Male':
        t2 = 0
    elif t2 =='Female':
        t2 = 1
    Label(root, text=t2)
    t3 = e3.get()
    Label(root, text=t3)
    t4 = e4.get()
    if t4 == 'Yes':
        t4 = 0
    elif t4 == 'No':
        t4 = 1
    Label(root, text=t4)
    t5 = e5.get()
    if t5 == 'Yes':
        t5 = 0
    elif t5 == 'No':
        t5 = 1
    Label(root, text=t5)
    t6 = e6.get()
    Label(root, text=t6)
    t7 = e7.get()
    if t7 == 'Yes':
        t7 = 0
    elif t7 == 'No':
        t7 = 1
    Label(root, text=t7)
    t8 = e8.get()
    if t8 == 'Yes':
        t8 = 0
    elif t8 == 'No':
        t8 = 1
    elif t8 == 'No phone service':
        t8 = 2
    Label(root, text=t8)
    t9 = e9.get()
    if t9 == 'DSL':
        t9 = 0
    elif t9 == 'Fiber optic':
        t9 = 1
    elif t9 == 'No':
        t9 = 2
    Label(root, text=t9)
    t10 = e10.get()
    if t10 == 'Yes':
        t10 = 0
    elif t10 == 'No':
        t10 = 1
    elif t10 == 'No internet service':
        t10 = 2
    Label(root, text=t10)
    t11 = e11.get()
    if t11 == 'Yes':
        t11 = 0
    elif t11 == 'No':
        t11 = 1
    elif t11 == 'No internet service':
        t11 = 2
    Label(root, text=t11)
    t12 = e12.get()
    if t12 == 'Yes':
        t12 = 0
    elif t12 == 'No':
        t12 = 1
    elif t12 == 'No internet service':
        t12 = 2
    Label(root, text=t12)
    t13 = e13.get()
    if t13 == 'Yes':
        t13 = 0
    elif t13 == 'No':
        t13 = 1
    elif t13 == 'No internet service':
        t13 = 2
    Label(root, text=t13)
    t14 = e14.get()
    if t14 == 'Yes':
        t14 = 0
    elif t14 == 'No':
        t14 = 1
    elif t14 == 'No internet service':
        t14 = 2
    Label(root, text=t14)
    t15 = e15.get()
    if t15 == 'Yes':
        t15 = 0
    elif t15 == 'No':
        t15 = 1
    elif t15 == 'No internet service':
        t15 = 2
    Label(root, text=t15)
    t16 = e16.get()
    if t16 == 'Month-to-month':
        t16 = 0
    elif t16 == 'One year':
        t16 = 1
    elif t16 == 'Two year':
        t16 = 2
    Label(root, text=t16)
    t17 = e17.get()
    if t17 == 'Yes':
        t17 = 0
    elif t17 == 'No':
        t17 = 1
    Label(root, text=t17)
    t18 = e18.get()
    if t18 == 'Electronic check':
        t18 = 0
    elif t18 == 'Mailed check':
        t18 = 1
    elif t18 == 'Bank transfer (automatic)':
        t18 = 2
    elif t18 == 'Credit card (automatic)':
        t18 = 3
    Label(root, text=t18)
    t19 = e19.get()
    Label(root, text=t19)
    t20 = e20.get()
    Label(root, text=t20)
    arr = [[t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20]]

    def predlog():
        print('the prediction by LogisticRegression is :')
        y1=LogisticRegressionModel.predict(arr)
        """  if y1 == 0:
           print('yes')
        elif y1==1:
            """
        print(y1)

    def predsvm():
        print('the prediction by SVM is :')
        y2=SVCModel.predict(arr)
        """if y2 == 0:
            print('yes')
        elif y2 == 1:
        """
        print(y2)

    def predtree():
        Decitree.predict(arr)
        print('the prediction by DecisionTree is :')
        y3=Decitree.predict(arr)
        """if y3 == 0:
            print('yes')
        elif y3== 1:
        """
        print(y3)

    def predknn():
       KNNModel.predict(arr)
       print('the prediction by KNN is :')
       y4=KNNModel.predict(arr)
       """ if y4 == 0:
           print('yes')
       elif y4 == 1:
           """
       print(y4)

    def predbayes():
        print('the prediction by NaiveBayes is :')
        y5=MultiModel.predict(arr)
        """if y5 == 0:
            print('yes')
        elif y5 == 1:
        """
        print(y5)
    def predforest():
        print('the prediction by Random Forest is :')
        y6=RandomForestModel.predict(arr)
        """if y6==0:
           print('yes')
        elif y6==1:
        """
        print(y6)



        # 1cond
    if (var1.get() == 1):
        predlog()



    elif (var1.get() == 2):
        predsvm()



    elif (var1.get() == 3):
        predtree()



    elif (var1.get() == 4):
        predknn()



    elif (var1.get() == 5):
        predbayes()


    elif (var1.get() == 6):
        predforest()


var1 = IntVar()

radiobutton3=Radiobutton(root,text='Logistic Regression',bg='#00818A',fg='#000000' ,variable=var1,value=1).grid(row=1,column=1)
radiobutton4=Radiobutton(root, text='SVM',bg='#00818A',fg='#000000',variable=var1,value=2).grid(row=1,column=2)
radiobutton5=Radiobutton(root, text='ID3',bg='#00818A',fg='#000000', variable=var1,value=3).grid(row=1,column=3)
radiobutton6=Radiobutton(root, text='KNN',bg='#00818A',fg='#000000', variable=var1,value=4).grid(row=1,column=4)
radiobutton7=Radiobutton(root, text='Naive Bayes',bg='#00818A',fg='#000000', variable=var1,value=5).grid(row=1,column=5)
radiobutton8=Radiobutton(root,text='Random Forest',bg='#00818A',fg='#000000',variable =var1,value=6).grid(row=1,column=6)


var7 = IntVar()
radiobutton1=Radiobutton(root,text='Test',variable=var7,bg='#00818A',fg='#000000' ,value=1).grid(row=22,column=2)
radiobutton2=Radiobutton(root,text='Train',variable=var7,bg='#00818A',fg='#000000' ,value=2).grid(row=22,column=4)
button1=Button(root,text='Display',bg='#FEC260',fg='#000000',width=15,height=1,command=check).grid(row=24,column=3)
button3=Button(root,text='Predict',bg='#FEC260',fg='#000000',width=15,height=1,command=algo).grid(column=3,row=20)

Label(root, text='customerID',bg='#00818A',fg='#000000').grid(row=8,column=1)
Label(root, text='gender',bg='#00818A',fg='#000000').grid(row=8,column=3)
Label(root, text='SeniorCitizen',bg='#00818A',fg='#000000').grid(row=8,column=5)
e1 = Entry(root,bg='#FFF8BC')
e2 = Entry(root,bg='#FFF8BC')
e3 = Entry(root,bg='#FFF8BC')
e2.grid(row=8, column=4)
e1.grid(row=8, column=2)
e3.grid(row=8, column=6)

Label(root, text='Partner',bg='#00818A',fg='#000000').grid(row=9,column=1)
Label(root, text='Dependents',bg='#00818A',fg='#000000').grid(row=9,column=3)
Label(root, text='tenure',bg='#00818A',fg='#000000').grid(row=9,column=5)
e4 = Entry(root,bg='#FFF8BC')
e5 = Entry(root,bg='#FFF8BC')
e6 = Entry(root,bg='#FFF8BC')
e4.grid(row=9, column=2)
e5.grid(row=9, column=4)
e6.grid(row=9, column=6)

Label(root, text='PhoneService',bg='#00818A',fg='#000000').grid(row=10,column=1)
Label(root, text='MultipleLines',bg='#00818A',fg='#000000').grid(row=10,column=3)
Label(root, text='InternetService',bg='#00818A',fg='#000000').grid(row=10,column=5)
e7 = Entry(root,bg='#FFF8BC')
e8 = Entry(root,bg='#FFF8BC')
e9 = Entry(root,bg='#FFF8BC')
e7.grid(row=10, column=2)
e8.grid(row=10, column=4)
e9.grid(row=10, column=6)

Label(root, text='OnlineSecurity',bg='#00818A',fg='#000000').grid(row=11,column=1)
Label(root, text='OnlineBackup',bg='#00818A',fg='#000000').grid(row=11,column=3)
Label(root, text='DeviceProtection',bg='#00818A',fg='#000000').grid(row=11,column=5)
e10 = Entry(root,bg='#FFF8BC')
e11= Entry(root,bg='#FFF8BC')
e12 = Entry(root,bg='#FFF8BC')
e10.grid(row=11, column=2)
e11.grid(row=11, column=4)
e12.grid(row=11, column=6)

Label(root, text='TechSupport',bg='#00818A',fg='#000000').grid(row=12,column=1)
Label(root, text='StreamingTV',bg='#00818A',fg='#000000').grid(row=12,column=3)
Label(root, text='StreamingMovies',bg='#00818A',fg='#000000').grid(row=12,column=5)
e13 = Entry(root,bg='#FFF8BC')
e14= Entry(root,bg='#FFF8BC')
e15 = Entry(root,bg='#FFF8BC')
e13.grid(row=12, column=2)
e14.grid(row=12, column=4)
e15.grid(row=12, column=6)

Label(root, text='Contract',bg='#00818A',fg='#000000').grid(row=13,column=1)
Label(root, text='PaperlessBilling',bg='#00818A',fg='#000000').grid(row=13,column=3)
Label(root, text='PaymentMethod',bg='#00818A',fg='#000000').grid(row=13,column=5)
e16 = Entry(root,bg='#FFF8BC')
e17= Entry(root,bg='#FFF8BC')
e18 = Entry(root,bg='#FFF8BC')
e16.grid(row=13, column=2)
e17.grid(row=13, column=4)
e18.grid(row=13, column=6)

Label(root, text='MonthlyCharges',bg='#00818A',fg='#000000').grid(row=14,column=1)
Label(root, text='TotalCharges',bg='#00818A',fg='#000000').grid(row=14,column=3)
e19 = Entry(root,bg='#FFF8BC')
e20= Entry(root,bg='#FFF8BC')
e19.grid(row=14, column=2)
e20.grid(row=14, column=4)

root.mainloop()