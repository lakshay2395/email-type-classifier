
# coding: utf-8

# In[26]:


import pandas

from sklearn import model_selection, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer 

import matplotlib.pyplot as plt

import re


lem = WordNetLemmatizer()

data = pandas.read_csv("mails.csv",sep=",",parse_dates=['Date'])


#Extracting Sender Email from sender and creating sender column
data["Sender Email"] = data["Sender"].str.split(" <").str.get(1).str.split(">").str.get(0)
data.loc[data["Sender Email"].isnull(),'Sender Email'] = data["Sender"].str.replace("<","").str.replace(">","")

#Setting NaN Subject to first part of sender column before '<'
data.loc[data["Subject"].isnull(),'Subject'] = data["Sender"].str.split(" <").str.get(0)

#Processing Dates
data["Date"] = data["Date"].str.replace("Sun, ","").str.replace("Mon, ","").str.replace("Tue, ","").str.replace("Wed, ","").str.replace("Thu, ","").str.replace("Fri, ","").str.replace("Sat, ","")
data["Date"] = pandas.to_datetime(data["Date"],errors='coerce')
data = data.drop(data[data["Date"].isnull()].index)

#removing stopwords

stopWords = set(stopwords.words('english'))

def text_without_stopwords(text):
    words = word_tokenize(str(text).lower())
    wordsFiltered = []
    for w in words:
        if w not in stopWords:
            x = lem.lemmatize(w)
            if x == w:
                x = lem.lemmatize(w,"v")
            wordsFiltered.append(x)
    return re.sub(r'[^a-zA-Z ]',r' '," ".join(wordsFiltered))

data["Subject"] = data["Subject"].apply(text_without_stopwords)

def clean_email_text(text):
    x = str(text).strip().split(".")
    y = x[0].split("@")
    y.append(x[1])
    return re.sub(r'[^a-zA-Z ]',r' '," ".join(y))

data['Sender Email'] = data['Sender Email'].apply(clean_email_text)


# In[27]:


def get_training_test_validation_sets(X,y,training_size=0.6,random_state=None):
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y, test_size=(1-training_size), random_state=random_state)
    X_test, X_val, y_test, y_val = model_selection.train_test_split(X_test, y_test, test_size=0.9, random_state=random_state)
    return X_train,X_test,X_val,y_train,y_test,y_val

def train_model(classifier,X_train,y_train,X_val,y_val):
    classifier.fit(X_train,y_train)
    return classifier.score(X_val, y_val),classifier

encoder = preprocessing.LabelEncoder()
data["Type"] = encoder.fit_transform(data["Type"])

def get_train_test_validate_data(training_size=0.6,random_state=None):
    X_train,X_test,X_val,y_train,y_test,y_val = get_training_test_validation_sets(data[["Subject","Sender Email"]],data["Type"],training_size=training_size,random_state=random_state)
    tfidf_vect = TfidfVectorizer(sublinear_tf=True, min_df=3, norm='l1', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
    tfidf_vect.fit(X_train["Subject"]) 
    
    train_Subject_tfidf = pandas.DataFrame(tfidf_vect.transform(X_train["Subject"]).todense(),columns=tfidf_vect.get_feature_names())
    val_Subject_tfidf = pandas.DataFrame(tfidf_vect.transform(X_val["Subject"]).todense(),columns=tfidf_vect.get_feature_names())
    test_Subject_tfidf = pandas.DataFrame(tfidf_vect.transform(X_test["Subject"]).todense(),columns=tfidf_vect.get_feature_names())

    train_Sender_Email_tfidf = pandas.DataFrame(tfidf_vect.transform(X_train["Sender Email"]).todense(),columns=tfidf_vect.get_feature_names())
    val_Sender_Email_tfidf = pandas.DataFrame(tfidf_vect.transform(X_val["Sender Email"]).todense(),columns=tfidf_vect.get_feature_names())
    test_Sender_Email_tfidf = pandas.DataFrame(tfidf_vect.transform(X_test["Sender Email"]).todense(),columns=tfidf_vect.get_feature_names())

    train = pandas.concat([train_Subject_tfidf,train_Sender_Email_tfidf],axis=1)
    test = pandas.concat([test_Subject_tfidf,test_Sender_Email_tfidf],axis=1)
    val = pandas.concat([val_Subject_tfidf,val_Sender_Email_tfidf],axis=1) 
    return train,test,val,y_train,y_test,y_val


# In[32]:


models = {
    'naive_bayes.MultinomialNB' : naive_bayes.MultinomialNB,
    'linear_model.LogisticRegression' : linear_model.LogisticRegression,
    'svm.LinearSVC' : svm.LinearSVC,
    'ensemble.RandomForestClassifier' : ensemble.RandomForestClassifier
}

train,test,val,y_train,y_test,y_val = get_train_test_validate_data(training_size=0.66)

for key in models.keys():
    print("****** ",key," STARTED ******")
    accuracy,model = train_model(models[key](),train, y_train, val,y_val)
    print("Accuracy(val): ",accuracy)
    accuracy = model.score(test, y_test)
    print("Accuracy(test): ",accuracy)
    print("****** ",key," ENDEDED ******")
    print("\n")

