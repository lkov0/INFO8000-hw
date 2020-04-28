from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn import metrics
from io import BytesIO
import base64
sns.set()
app = Flask("microbiome_app",template_folder='templates', static_url_path='/static')
clf = None
df = None
    
def train_rf(endo):
    global df, clf_rf
    X = df.drop(columns=[endo])
    y = np.array(df[endo])
    feature_list = list(X.columns)
    train_features, test_features, train_labels, test_labels = train_test_split(X, y, test_size = 0.25, random_state = 18)
    clf_rf = RandomForestClassifier(n_estimators=10000, random_state=18, max_features = 'sqrt',n_jobs=-1, verbose = 1)
    clf_rf.fit(X,y)
    y_pred = clf_rf.predict(test_features)
    
    # make figure of top 10 features, importances
    feature_imp = pd.Series(clf_rf.feature_importances_,index=feature_list).sort_values(ascending=False)
    sns.barplot(x=feature_imp[:10], y=feature_imp.index[:10])
    # Add labels to your graph
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title("Visualizing Important Features - Random Forest")
    plt.savefig("static/img.png",format='png', bbox_inches='tight')
        
    return metrics.accuracy_score(test_labels,y_pred)

def train_dt(endo):
    global df, clf_dt
    X = df.drop(columns=[endo])
    y = np.array(df[endo])
    train_features, test_features, train_labels, test_labels = train_test_split(X, y, test_size = 0.25, random_state = 18)
    clf_dt = DecisionTreeClassifier()
    clf_dt.fit(X,y)
    y_pred = clf_rf.predict(test_features)
    return metrics.accuracy_score(test_labels,y_pred)

def train_abc(endo):
    global df, clf_abc
    X = df.drop(columns=[endo])
    y = np.array(df[endo])
    train_features, test_features, train_labels, test_labels = train_test_split(X, y, test_size = 0.25, random_state = 18)
    clf_abc = AdaBoostClassifier(n_estimators=1000, learning_rate=1)
    clf_abc.fit(X,y)
    y_pred = clf_abc.predict(test_features)
    return metrics.accuracy_score(test_labels,y_pred)

def train_svm(endo):
    global df, clf_svm
    X = df.drop(columns=[endo])
    y = np.array(df[endo])
    train_features, test_features, train_labels, test_labels = train_test_split(X, y, test_size = 0.25, random_state = 18)
    clf_svm = svm.SVC(kernel='rbf',C=1.91,gamma='scale') # Linear Kernel
    clf_svm.fit(X,y)
    y_pred = clf_svm.predict(test_features)
    return metrics.accuracy_score(test_labels,y_pred)

def train_nb(endo):
    global df, clf_nb
    X = df.drop(columns=[endo])
    y = np.array(df[endo])
    train_features, test_features, train_labels, test_labels = train_test_split(X, y, test_size = 0.25, random_state = 18)
    clf_nb = GaussianNB()
    clf_nb.fit(X,y)
    y_pred = clf_nb.predict(test_features)
    return metrics.accuracy_score(test_labels,y_pred)

def train_nn(endo):
    global df, clf_nn
    X = df.drop(columns=[endo])
    y = np.array(df[endo])
    train_features, test_features, train_labels, test_labels = train_test_split(X, y, test_size = 0.25, random_state = 18)
    clf_nn = MLPClassifier(hidden_layer_sizes=(79,79,79), activation='relu', solver='adam', max_iter=100000)
    clf_nn.fit(X,y)
    y_pred = clf_nn.predict(test_features)
    return metrics.accuracy_score(test_labels,y_pred)

    
def init():
    global df
    np.random.seed(18)
    df = pd.read_csv('Microbiome.csv', delimiter = '\t')
    df.rename(columns = {'345dcc18d51f44572bd67c08e5e95b8b':'Endo1','074e66f75650948b8df12cfe2ffb5f37':'Endo2','060fdbbfa61cbfb4d47350dc2a2019cd':'Endo3','d2208d27b5df4c53eb547f7ac45f4d6b':'Endo4','574d164310944193d8fc13dc10346e58':'Endo5','8cb92babedb9f4ff7bedee4ac4f47370':'Endo6','3e00a33b844a56c2e00acedeffc43b5e':'Endo7','0f5f7693288de84f4ade2e6abaa2440f':'Endo8','561ed5d9dab98c645f731a40b7b63fa4':'Endo9','a6d4742d8c802171498b62b6d79b1764':'Endo10'}, inplace = True)
    df = df.drop(['PlantA_Or_B','Notes','GA_Microbiome','Living_Mulch','Endophyte','R1_Fastq_Name','R2_Fastq_Name','Sample_or_Control','Sampling_Number','Plate','Row','Column','Well','Soil_Test_Number','Maize_Sample','Living_Mulch_Treatment','reads','quant_reading','Concentration'], axis =1)
    df = df.dropna(axis=0, how='any')
    df = pd.get_dummies(df)

init()

# show an interface to add/test data, which will hit test
@app.route("/")
def main():
    return render_template("main.html")

# this function adds a row to the dataset and retrains
@app.route("/run_observation",methods=["POST"])
def run_classification():
    global df
    global clf_rf
    global clf_dt
    global clf_abc
    global clf_svm
    global clf_nb
    global clf_nn
    try:
        endo = request.values.get('endo','Endo1')
        is_test = request.values.get("test","no")
    except: 
        return "Error parsing entries"
    
    if is_test != "no":
        rf_accu = str(train_rf(endo))
        dt_accu = str(train_dt(endo))
        abc_accu = str(train_abc(endo))
        svm_accu = str(train_svm(endo))
        nb_accu = str(train_nb(endo))
        nn_accu = str(train_nn(endo))
        
        #output = endo + " ACCURACIES<br><br>Random forest: " + rf_accu + "<br>Decision Tree: " + dt_accu + "<br>Ada Boost classifier: " + abc_accu + "<br>Support vector machine: " + svm_accu + "<br>Naive Bayes: " + nb_accu + "<br>Neural network: " + nn_accu

        return render_template("result.html", endo = endo, rf_accu = rf_accu, dt_accu = dt_accu, abc_accu = abc_accu, svm_accu = svm_accu, nb_accu = nb_accu, nn_accu = nn_accu)
        
    return "not implemented"
if __name__ == "__main__":
    app.run()
