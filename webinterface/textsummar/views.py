from urllib import request
from django.shortcuts import render
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import words
from nltk.stem import LancasterStemmer
import os
from pathlib import Path

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('words')

stop_words=stopwords.words("english")
lemmatizer = WordNetLemmatizer()
dic = words.words()
ls = LancasterStemmer()


def clean_text(text):
  for punctuation in string.punctuation:
    text = text.replace(punctuation, '')
  text = " ".join([i for i in word_tokenize(text) if i not in stop_words])
  text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])
  return text

from math import ceil
def get_thematic_keywords(text):
  w = nltk.word_tokenize(text)
  dic = nltk.FreqDist(w)
  return list(dict(dic.most_common(ceil(len(w)*0.1))))

def get_thematic_score(text):
  text = set(nltk.word_tokenize(text))
  cnt = sum([1 for word in st if word in text])
  return (cnt/len(st))

dataset=pd.read_csv("D:\downloads\duc_2001 (1).csv")
dataset = dataset.iloc[:,1:]
single_dataset = pd.read_csv("D:\downloads\single_dataset.csv")
clean_dataset = pd.read_csv("D:\downloads\duc_2001_clean.csv")
reference_dataset = pd.read_csv("D:\downloads\webinterfce.csv")
# reference_dataset = pd.read_csv("D:\downloads\reference_summary.csv")
# Create your views here.

def index(request):
    return render(request,"index.html")

def generate(request):
    if request.method=="POST":
        dir=request.POST.get("dirno")
        doc=request.POST.get("docno")
        d=list(dataset[(dataset['Docno'] == doc) & (dataset['Dir'] == dir)]['Text'])[0]
        # ct = clean_text(d)
        global st
        st = single_dataset[(single_dataset['Docno'] == doc) & (single_dataset['Dir'] == dir)]['Text'].iloc[0][0]
        tw = get_thematic_keywords(st)
        cd = clean_dataset[(clean_dataset['Docno'] == doc) & (clean_dataset['Dir'] == dir)]
        cd['thematic_score'] = cd['Stemmed_text'].apply(get_thematic_score)
        cd = cd.sort_values(by = 'thematic_score', ascending = False)
        rs = ''
        cnt = 0
        for i in cd['sentence']:
          if cnt + len(i.split()) <= 100:
            cnt += len(i.split())
            rs += (i+' ')

        rd = list(reference_dataset[(reference_dataset['Docno'] == doc) & (reference_dataset['Dir'] == dir)]['ref_summary'])[0]
    return render(request,"generate.html",{"ok":d, "gs": rs, 'rd':rd})
