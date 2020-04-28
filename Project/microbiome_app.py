from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle
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
app = Flask("microbiome_app",template_folder='templates')
clf = None
df = None

#userful function that gets the current figure as a base 64 image for embedding into websites
def getCurrFigAsBase64HTML():
    im_buf_arr = BytesIO()
    plt.gcf().savefig(im_buf_arr,format='png')
    im_buf_arr.seek(0)
    b64data = base64.b64encode(im_buf_arr.read()).decode('utf8');
    return render_template('img.html',img_data=b64data) 
    
def train_rf():
    global df, clf
    X = df.drop(columns=['endo'])
    y = np.array(df['Endo1'])
    clf
    clf
    return clf

def train_dt():
    global df, clf
    X = df.drop(columns=['endo'])
    y = np.array(df['endo'])
    clf
    clf
    
    return clf

def train_abc():
    global df, clf
    X = df.drop(columns=['endo'])
    y = np.array(df['endo'])
    clf
    clf

    return clf

def train_svm():
    global df, clf_svm
    X = df.drop(columns=['endo'])
    y = np.array(df['endo'])
    clf_svm
    clf_svm
    
    return clf_svm

def train_nb():
    global df, clf
    X = df.drop(columns=['endo'])
    y = np.array(df['endo'])
    clf
    clf

    return clf

def train_nn():
    global df, clf
    X = df.drop(columns=['endo'])
    y = np.array(df['endo'])
    clf
    clf

    return clf

    
def init():
    global df
    np.random.seed(18)
    df = pd.read_csv('Microbiome.csv')
    df.rename(columns = {'345dcc18d51f44572bd67c08e5e95b8b':'Endo1','074e66f75650948b8df12cfe2ffb5f37':'Endo2','060fdbbfa61cbfb4d47350dc2a2019cd':'Endo3','d2208d27b5df4c53eb547f7ac45f4d6b':'Endo4','574d164310944193d8fc13dc10346e58':'Endo5','8cb92babedb9f4ff7bedee4ac4f47370':'Endo6','3e00a33b844a56c2e00acedeffc43b5e':'Endo7','0f5f7693288de84f4ade2e6abaa2440f':'Endo8','561ed5d9dab98c645f731a40b7b63fa4':'Endo9','a6d4742d8c802171498b62b6d79b1764':'Endo10'}, inplace = True)
    df = df.drop(['PlantA_Or_B','Notes','GA_Microbiome','Living_Mulch','Endophyte','R1_Fastq_Name','R2_Fastq_Name','Sample_or_Control','Sampling_Number','Plate','Row','Column','Well','Soil_Test_Number','Maize_Sample','Living_Mulch_Treatment','reads','quant_reading','Concentration'], axis =1)
    df = df.dropna(axis=0, how='any')
    df = pd.get_dummies(df)
    train_rf()
    train_dt()
    train_abc()
    
    
    

init()

# show an interface to add/test data, which will hit test
@app.route("/")
def main():
    return render_template("main.html")

# this function adds a row to the dataset and retrains
@app.route("/run_observation",methods=["POST"])
def run_classification():
    global df
    global clf
    global clf
    global clf
    global clf_svm
    global clf
    global clf
    try:
        endo = request.values.get('endo','Endo1')
        is_test = request.values.get("test","no")
    except: 
        return "Error parsing entries"
    
    if is_test != "no":
        # run tests here and store in df?
        
        return endo
        
    return "not implemented"
if __name__ == "__main__":
    app.run()