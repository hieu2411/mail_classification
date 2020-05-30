from flask import *
import codecs
import csv
import os
import sys

import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB


app = Flask(__name__)


def write_emails(data=[]):
    fw = codecs.open('emails.csv', 'w', 'utf-8')
    writer = csv.writer(fw)
    writer.writerows(data)

def read_comments(path='./data_train/train/emails.csv'):
    global row, index
    csv.field_size_limit(sys.maxsize)
    rw = csv.reader(codecs.open('emails.csv', 'rU', 'utf-8'))
    result = []
    index =0
    try:
        for row in rw:
            index +=1
            if row is not None:
                result.append(row)
        return result
    except Exception as e:

        print(row, 'index', index, e)
        return result

def write_data_to_csv():
    train_dir_spam = './enron1/spam/'
    files = os.listdir(train_dir_spam)
    emails = []
    for file in files:
        temp = (str(open(train_dir_spam+ str(file)).read()), 0)
        emails.append(temp)

    train_dir_ham = './enron1/ham/'
    files = os.listdir(train_dir_ham)
    for file in files:
        temp = (str(open(train_dir_ham + str(file)).read()), 1)
        emails.append(temp)
    emails = remove_special_char(emails)
    write_emails(data=emails)

def remove_special_char(emails):
    global line
    processed = []
    try:
        for line in emails:
            if line is not None:
                email = line[0]
                sentiment = line[1]
                email = email.replace('\n', ' ')
                email = email.replace('_', ' ')
                email = email.replace(',', '')
                email = email.replace('.', '')
                email = email.replace('!', '')
                email = email.replace('(', '')
                email = email.replace(')', '')
                email = email.replace('/', '')
                email = email.replace(':', '')
                email = email.replace('$', '')
                email = email.replace('`', '')
                email = email.replace('\'', '')
                email = email.replace('^', '')
                email = email.replace('-', '')
                email = email.replace('   ', ' ')
                email = email.replace('  ', ' ')
                email = email.replace('\0', '')


                processed.append((email, sentiment))
        return processed
    except:
        print(line)

def format_email(email):
    email = email.replace('\n', ' ')
    email = email.replace('_', ' ')
    email = email.replace(',', '')
    email = email.replace('.', '')
    email = email.replace('!', '')
    email = email.replace('(', '')
    email = email.replace(')', '')
    email = email.replace('/', '')
    email = email.replace(':', '')
    email = email.replace('$', '')
    email = email.replace('`', '')
    email = email.replace('\'', '')
    email = email.replace('^', '')
    email = email.replace('-', '')
    email = email.replace('   ', ' ')
    email = email.replace('  ', ' ')
    email = email.replace('\0', '')
    return email

def load_trained_data():
    try:
        loaded_model = joblib.load('model.pkl')
        loaded_vectorizer = joblib.load('vectorizer.pkl')

        return  loaded_model, loaded_vectorizer
    except:
        return None, None

def classify(document):
    label = {0: 'spam mail', 1: 'ham mail'}
    X = loaded_vectorizer.transform([document])
    y = loaded_model.predict(X)[0]
    proba = np.max(loaded_model.predict_proba(X))
    return label[int(y)], proba

loaded_model, loaded_vectorizer = load_trained_data()

if loaded_model is None or loaded_vectorizer is None:
    write_data_to_csv()
    emails = read_comments()

    df = pd.DataFrame(emails, columns=['review', 'sentiment'])

    x_train = df.loc[:, 'review'].values
    y_train = df.loc[:, 'sentiment'].values


    vectorizer = CountVectorizer()
    counts = vectorizer.fit_transform(x_train)

    model = MultinomialNB()
    model.fit(counts, y_train)

    print('Score on training data is: ' + str(model.score(counts, y_train)))

    joblib.dump(model, 'model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')
    loaded_model, loaded_vectorizer = load_trained_data()





@app.route('/', methods = ['get', 'post'])
def index():

    return render_template('index.html', title = 'Mail classification')


@app.route('/results', methods=['get', 'post'])
def results():
    if request.method == 'POST':
        mail = request.form['mail']
        file_content = ''
        if 'file' not in request.files:
            return render_template('index.html', title='Mail classification')
        else:
            file = request.files['file']
            filename, file_extension = os.path.splitext(file.filename)
            if file_extension == '.txt':  # allowed
                # save because it's easier to read, format
                file.save(filename)
                rw = csv.reader(codecs.open(filename, 'rU', 'utf-8'))
                content = ''
                for row in rw:
                    if row is not None:
                        content += row[0] + ' '
                file_content = format_email(content)
                # delete file
                os.remove(filename)

        result_mail, mail_proba = classify(mail)
        result_file, file_proba = classify(file_content)
        data = {'mail': mail,
                'result_mail': result_mail,
                'mail_proba': mail_proba * 100,
                'file': file_content,
                'result_file': result_file,
                'file_proba':file_proba * 100}
        return render_template('results.html', title = 'Results' ,data = data)
    return render_template('results.html', title='Results', data = None)


if __name__ == '__main__':
    app.run()