#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals, print_function
from flask import Flask, request, jsonify

from pathlib import Path

import spacy
from spacy.util import minibatch, compounding

import plac
import json
import spacy
import sys

import textacy
import textacy.keyterms

# from spacy.tokenizer import Tokenizer
# from spacy.lang.en import English
from collections import defaultdict

import random
import binascii
import base64
import os
from datetime import datetime
from flask import Flask, render_template, jsonify, redirect, url_for, request

from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
import numpy as np
import cv2
import json
import pandas as pd
import datetime
import itertools
from scipy import stats

main_dir = "."

app = Flask(__name__)
app.config.from_object(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg', 'gif']
messagelists = []
textcount = 0
imagecount = len(os.listdir(os.path.join(main_dir,"faces")))
lastimageno = 0

@app.route('/api/', methods=["POST"])
def main_interface():
    global textcount
    global messagelists
    global imagecount
    global lastimageno
    response = request.get_json()
    newmess = response["journal"]
    inputlist = []

    if textcount != 0:
        with open("messagelist.json", 'r') as f1:
            datastore = json.load(f1)
            for item in datastore:
                inputlist.append(item)
    inputlist.append({"results":newmess})
    textcount += 1
    with open("messagelist.json", 'w+') as f2:
        json.dump(inputlist, f2)

    img_data = response["image"]
    imgname_file = str(imagecount) + ".jpeg"
    imgname = os.path.join("./faces", imgname_file)
    with open(imgname, "wb") as fh:
        fh.write(base64.decodebytes(img_data.encode('utf-8')))
    imagecount += 1

    return jsonify({"results": "Success"})


@app.after_request
def add_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    return response

@app.route('/uploadajax/', methods=['POST'])
def upload():
    global imagecount
    global lastimageno
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            now = datetime.now()
            #os.path.join(main_dir,"faces"
            filename = os.path.join("./faces", "%s.%s" % (imagecount, file.filename.rsplit('.', 1)[1]))
            file.save(filename)
            imagecount += 1
            return jsonify({"success":True})
        else:
            return jsonify({"success":False})

@app.route('/getresults/', methods=['GET'])
def sendbackresults():
    global imagecount
    global lastimageno
    global textcount
    global messagelists

    mess = []
    try:
        textcount = 1 #DEBUG LINE!!!!!
        if textcount != 0:
            with open("messagelist.json", 'r') as f1:
                datastore = json.load(f1)
                for item in datastore:
                    mess.append(item["results"])

        clustering_results = derdesdemden(input=mess)
        facial_results = facial_recog()
        textpredictor = textclasser()

        result1 = clustering_results.split(",")
        #result_text = "Your most common key expressions in your journals: " + clustering_results + "\n"

        #rawscore = []
        facial_numbers = []
        #with open("imageresults.json", 'r') as f2:
        #    json.dump(allresults, f2)
        #    print("Testing data of size {} has been generated.".format(len(allresults)))

        with open("imageresults_posneg.json", 'r') as f3:
            datastore = json.load(f3)
            for item in datastore:
                facial_numbers.append(item["score"])

        facial_numbers = np.array(facial_numbers)
        text_numbers = np.array(textpredictor)
        print(facial_numbers)
        for i in range(len(facial_numbers)):
            if facial_numbers[i] == 0:
                # False negative case. Use the text score as emotive score.
                facial_numbers[i] = text_numbers[i]
            elif facial_numbers[i] * text_numbers[i] < 0:
                # Misclassification case (False positive) - penalise in opposite direction by halving the value.
                facial_numbers[i] /= -2.0
        print(facial_numbers)
        allemo = ""
        with open("imageresults.json", 'r') as f2:
            datastore = json.load(f2)
            for item in datastore:
                allemo += item["emotion"] + ","
        allemo = allemo[:-1]

        result2 = allemo.split(",")

        hrv_res, hr_sleep = healthdata()
        result_text3 = ""
        result_text4 = ""
        result_text5 = ""
        for numwer in hrv_res:
            result_text3 += str(numwer) + ","
        result_text3 = result_text3[:-1]
        result3 = result_text3.split(",")
        for numwer in hr_sleep:
            result_text4 += str(numwer) + ","
        result_text4 = result_text4[:-1]
        result4 = result_text4.split(",")
        for numwer in facial_numbers:
            result_text5 += str(numwer) + ","
        result_text5 = result_text5[:-1]
        result5 = result_text5.split(",")
        #print(len(result3), len(result4))

        facial_score = sum(facial_numbers)
        #result6 = "Your weekly SLFYCare emotion score total: " + str(facial_score) + "\n"
        result6 = [str(facial_score)]

        final_json = [{"keywords":result1},{"emotions":result2},{"heartratev":result3},{"sleephrs":result4},{"emotive_score":result5},{"weekly_tally":result6}]

        return jsonify(final_json)
    except Exception as e:
        print(e)
        return("Nope, nothing to see here!")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def derdesdemden(input=None, output_dir="newout_updated", file_format="",
        algorithm="s", n_key=10, n_key_float=0.0, n_grams="1,2,3,4",
        runmode="t", daily_output="daily_keyword.json", cutoff=10,
        monthly_output="monthly_keywords.txt", common_keyword_file=None,
        threshold=0.5):
    if runmode != "t" and runmode != "m" and runmode != "d":
        return("Set runmode to 't', 'm', or 'd' only!")

    if algorithm != "t" and algorithm != "s":
        return("Specify an algorithm! (t)extrank or (s)grank")

    if input is None:
        return("Specify input file with -i")


    alldata = []

    for curline in input:
        alldata.append(curline)

    # Preprocessing data and de-duplicating only applies to daily runs. - NOT ANYMORE
    # if runmode == 'd':
    # Preprocess data by removing garbage keywords
    alldata = clean_data(alldata)

    # Remove duplicates via MinHash algorithm
    #alldata = de_dupe(alldata, threshold)

    # the cummulative tally of common keywords
    word_keyterm_cummula = defaultdict(lambda: 0)
    # the mapping of journals to the common keywords
    word_keyterm_journals = defaultdict(lambda: [])

    en = textacy.load_spacy_lang("en_core_web_sm", disable=("parser",))
    for item in alldata:
        #print(item)
        msgid = item.split(' ')[0]
        curline = item.replace(msgid, '').strip()
        #print(curline)
        curdoc = textacy.make_spacy_doc(curline.lower(), lang=en)
        curdoc_ranks = []
        if algorithm == "t":
            if n_key_float > 0.0 and n_key_float < 1.0:
                curdoc_ranks = textacy.keyterms.textrank(curdoc,
                    normalize="lemma", n_keyterms=n_key_float)
            else:
                curdoc_ranks = textacy.keyterms.textrank(curdoc,
                    normalize="lemma", n_keyterms=n_key)
        elif algorithm == "s":
            ngram_str = set(n_grams.split(','))
            ngram = []
            for gram in ngram_str:
                ngram.append(int(gram))
            if n_key_float > 0.0 and n_key_float < 1.0:
                curdoc_ranks = textacy.keyterms.sgrank(curdoc,
                    window_width=1500, ngrams=ngram, normalize="lower",
                    n_keyterms=n_key_float)
            else:
                curdoc_ranks = textacy.keyterms.sgrank(curdoc,
                    window_width=1500, ngrams=ngram, normalize="lower",
                    n_keyterms=n_key)

        for word in curdoc_ranks:
            word_keyterm_cummula[word[0]] += 1
            if runmode == 'd':
                word_keyterm_journals[word[0]].append((msgid, word[1]))
                if len(word_keyterm_journals[word[0]]) > 10:
                    newlist = []
                    min_tuple = word_keyterm_journals[word[0]][0]
                    for tuple in word_keyterm_journals[word[0]]:
                        if tuple[1] < min_tuple[1]:
                            min_tuple = tuple
                    for tuple in word_keyterm_journals[word[0]]:
                        if tuple[0] != min_tuple[0]:
                            newlist.append(tuple)
                    word_keyterm_journals[word[0]] = newlist

    word_keyterm_cummula_sorted = sorted(word_keyterm_cummula.items(),
        key=lambda val: val[1], reverse=True)

    cutoff = 10
    if runmode == 't':
        # Test mode. Print out every five words in the top 100 highest
        # ranked key phrases. The total count is only used in test mode to
        # print the results.
        total_count = 0.0
        for entry in word_keyterm_cummula_sorted:
            total_count += entry[1]

        quint = 0
        quint_printout = ""
        for entry in word_keyterm_cummula_sorted[:cutoff]:
            quint_printout += entry[0] + ","
            #quint_printout += entry[0] + ", " + \
            #    "{0:.5f}".format(float(entry[1]) / total_count) + " | "
            quint += 1
        quint_printout = quint_printout[:-1]
        #print(quint_printout)
        #quint_printout += "\n"
        #quint = 0
        #quint_printout = ""
        print(quint_printout)
        return quint_printout


"""
    Preprocessing function that removes excessive punctuations, any floating
    punctuations, any file extensions, and unneccessary entities.
"""


def clean_data(journal_list):
    nlp = spacy.load('en_core_web_sm')  # make sure to use larger model!

    fine_data = []
    # Delete any occurrences of these but keep the words attached to them.
    garbage_punc = ['...', '....', '.....', '///', '////', '/////', '---',
        '----', '-----']
    # Remove any files with these extensions
    file_exts = [".html", "[/url", ".xxx", ".jpg", ".jpeg", ".png", ".gif",
        ".txt", ".doc", ".docx", ".pdf"]
    # Look for any words which contain these X's as substrings, remove them.
    xs = ['xxx', 'xxxx', 'xxxxx']
    # Delete any occurrences of these if they occur as a single token
    punctuations = ['!', '?', '_', '/', '-', '+', '=', '>', '|', '[', ']',
        '{', '}', '(', ')', ',', '#', "\"", "\'"]
    remove_these_entities = ['DATETIME', 'ORDER_NUMBER', 'ADDRESS_STREET_2',
        'ADDRESS_STREET_1', 'ADDRESS_ZIP', 'MONEY']

    for curline in journal_list:
        # Separate the journal ID from the message, then remove all non-ascii
        # characters
        msgid = curline.split(' ')[0]
        curline = curline.replace(msgid, '').strip()
        curline = remove_non_ascii(curline).strip()

        # Get rid of gibberish - remove any excessive punctuations.
        for garb in garbage_punc:
            curline = curline.replace(garb, '')

        # Tokenize the sentence to further prune the sentences.
        doc = nlp(curline)
        strtok = ""
        for token in doc:
            if token.ent_type_ not in remove_these_entities:
                strtok += token.text + " "

        # Remove all punctuation marks.
        for char in strtok:
            if char in punctuations:
                if strtok[0:2] == char + ' ':
                    strtok = strtok[2:]
                elif strtok[-2:] == ' ' + char:
                    strtok = strtok[:-2]
                else:
                    strtok = strtok.replace(' ' + char + ' ', ' ')

        stringtoanalyze = strtok.strip()
        removal_dump = []

        """
            Go through the string and prune the following:

            1. Any non-English words.
            2. Any word greater than 20 characters in length.
            3. Any Base64 encryptions and file names.
            4. Any words with lots of 'x' in it.
        """
        for word in stringtoanalyze.split():
            if not isEnglish(word):
                removal_dump.append(word)
                continue
            if len(word) > 20:
                removal_dump.append(word)
                continue
            if word[-4:] in file_exts or word[-5:] in file_exts or \
                    word[-2:] == "==":
                removal_dump.append(word)
                continue
            wordlw = word.lower()
            if "xxxx" in wordlw or "xxx" in wordlw or wordlw[:4] == "xxxx" or \
                    wordlw[-4:] == "xxxx" or wordlw[:3] == "xxx" or \
                    wordlw[-3:] == "xxx":
                removal_dump.append(word)
                continue
            for exes in xs:
                if exes in wordlw:
                    removal_dump.append(word)

        for rem in removal_dump:
            if stringtoanalyze == rem:
                stringtoanalyze = ""
            elif stringtoanalyze[:len(rem)] == rem:
                stringtoanalyze = stringtoanalyze[len(rem):]
            elif stringtoanalyze[(-1 * len(rem)):] == rem:
                stringtoanalyze = stringtoanalyze[:(-1 * len(rem))]
            else:
                stringtoanalyze = stringtoanalyze.replace(' ' + rem + ' ', ' ')

        # If all the pruning results in a nonempty string of length greater
        # than 1, it is safe to use for clustering.
        stringtoanalyze = stringtoanalyze.strip()
        if len(nlp(stringtoanalyze)) > 1:
            fine_data.append(msgid + ' ' + stringtoanalyze + '\n')

    print("Done with cleaning data.")

    return fine_data


"""
    Functions to remove any non-English words and emojis from journals for
    preprocessing purposes.
"""
def remove_non_ascii(s):
    for char in s:
        if len(char.encode('utf-8')) > 3:
            s = s.replace(char, '')
    return s


def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True


"""
    credits to Catherine for this code
"""
def facial_recog():
    global lastimageno
    global imagecount
    util_dir = "./utility"
    # loading the model
    json_file = open(os.path.join(util_dir, "fer_model.json"), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(os.path.join(util_dir, "fer.h5"))
    print("Loaded model from disk")
    # setting image resizing parameters
    WIDTH = 48
    HEIGHT = 48
    x=None
    y=None
    labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    neg_emotions = ['Angry', 'Disgust', 'Fear', 'Sad', 'Surprise']
    pos_emotions = ['Happy', 'Neutral']
    # loading image

    image_dir = "test/"
    #images = ["RadBoudFACES/Rafd090_73_Moroccan_male_happy_right.jpg", "RadBoudFACES/Rafd090_57_Caucasian_female_angry_right.jpg","RadBoudFACES/Rafd090_52_Moroccan_male_fearful_frontal.jpg", 'RadBoudFACES/Rafd090_45_Moroccan_male_sad_left.jpg']
    images = os.listdir(os.path.join(main_dir,"faces"))
    allresults = []
    allresults_posneg = []
    #print(images)

    #for image in range(imagecount):
    for image in range(len(images)):
        full_size_image = cv2.imread(os.path.join(main_dir, "faces", images[image]))
        print("Image Loaded")
        gray = cv2.cvtColor(full_size_image, cv2.COLOR_RGB2GRAY)
        face = cv2.CascadeClassifier(os.path.join(util_dir,'haarcascade_frontalface_default.xml'))
        faces = face.detectMultiScale(gray, 1.3  , 10)
        if len(np.array(faces)) == 0:
            allresults_posneg.append({"score": 0})
            allresults.append({"emotion": "Undefined"})
            continue
        # detecting faces
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
            cv2.rectangle(full_size_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
            # predicting the emotion
            yhat= loaded_model.predict(cropped_img)
            cv2.putText(full_size_image, labels[int(np.argmax(yhat))], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
            #print("True label: "+image)
            #print(int(np.argmax(yhat)))
            emotion = labels[int(np.argmax(yhat))]
            labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
            if emotion == "Angry":
                gen_emot = "Negative"
                allresults_posneg.append({"score": -5})
            elif emotion == "Disgust":
                gen_emot = "Negative"
                allresults_posneg.append({"score": -1})
            elif emotion == "Fear":
                gen_emot = "Negative"
                allresults_posneg.append({"score": -3})
            elif emotion == "Happy":
                gen_emot = "Negative"
                allresults_posneg.append({"score": 6})
            elif emotion == "Sad":
                gen_emot = "Negative"
                allresults_posneg.append({"score": -5})
            elif emotion == "Surprise":
                gen_emot = "Positive"
                allresults_posneg.append({"score": 1})
            else:
                gen_emot = "Positive"
                allresults_posneg.append({"score": 2})
            allresults.append({"emotion": emotion})
            print("Emotion: "+emotion)
            print("Generalized: "+gen_emot)


    with open("imageresults.json", 'w+') as f2:
        json.dump(allresults, f2)
        print("Testing data of size {} has been generated.".format(len(allresults)))
    with open("imageresults_posneg.json", 'w+') as f3:
        json.dump(allresults_posneg, f3)
        print("Testing data of size {} has been generated.".format(len(allresults_posneg)))

    #cv2.imshow('Emotion', full_size_image)
    #cv2.waitKey()


def textclasser(model="samplemodel", input_dir="messagelist.json"):
    nlp = spacy.load(model)  # load existing spaCy model
    print("Loaded model '%s'" % model)

    if "textcat" not in nlp.pipe_names:
        textcat = nlp.create_pipe(
            "textcat", config={"exclusive_classes": True, "architecture": "simple_cnn"}
        )
        nlp.add_pipe(textcat, last=True)
    else:
        textcat = nlp.get_pipe("textcat")

    textcat.add_label("POSITIVE")
    textcat.add_label("NEGATIVE")

    # predict everything!
    predictions = []
    with open(input_dir, 'r') as f1:
        datastore = json.load(f1)
        for item in datastore:
            currmess = item["results"]
            doc = nlp(currmess)
            if doc.cats["POSITIVE"] >= doc.cats["NEGATIVE"]:
                predictions.append(2)
            else:
                predictions.append(-2)
    return(predictions)

"""
    credits to Catherine for this code
"""
def healthdata():
    # load in data

    apple_watch_dir = './healthdata'
    hrv = pd.read_csv(os.path.join(apple_watch_dir, "HeartRateVariabilitySDNN.csv"))
    sleep = pd.read_csv(os.path.join(apple_watch_dir, "SleepAnalysis.csv"))
    sleep = sleep.loc[sleep["value"] == "HKCategoryValueSleepAnalysisAsleep"]
    workouts = pd.read_csv(os.path.join(apple_watch_dir, "Workout.csv"))

    today = datetime.datetime.today().date()
    get_date = lambda x: '{}-{:02}-{:02}'.format(x.year, x.month, x.day)

    images = os.listdir(os.path.join(main_dir,"faces"))

    days_back = len(images)
    #print(days_back)

    # convert dates in data to date objects
    hrv['creationDate'] = pd.to_datetime(hrv['creationDate'])
    # make compatible to compare with dates
    for date_i in range(len(hrv)):
        if hrv['creationDate'].iloc[date_i].month < 10:
            month_str = "0" + str(hrv['creationDate'].iloc[date_i].month)
        else:
            month_str = str(hrv['creationDate'].iloc[date_i].month)
        if hrv['creationDate'].iloc[date_i].day < 10:
            day_str = "0" + str(hrv['creationDate'].iloc[date_i].day)
        else:
            day_str = str(hrv['creationDate'].iloc[date_i].day)
        hrv.at[date_i, 'date'] = str(hrv['creationDate'].iloc[date_i].year)+"-"+month_str+"-"+day_str


    workouts['creationDate'] = pd.to_datetime(workouts['creationDate'])
    for date_i in range(len(workouts)):
        if workouts['creationDate'].iloc[date_i].month < 10:
            month_str = "0"+str(workouts['creationDate'].iloc[date_i].month)
        else:
            month_str = str(workouts['creationDate'].iloc[date_i].month)
        if workouts['creationDate'].iloc[date_i].day < 10:
            day_str = "0"+str(workouts['creationDate'].iloc[date_i].day)
        else:
            day_str = str(workouts['creationDate'].iloc[date_i].day)
        workouts.at[date_i, 'date'] = str(workouts['creationDate'].iloc[date_i].year)+"-"+month_str+"-"+day_str

    workouts['z_score_duration'] = stats.zscore(workouts['duration'])

    sleep['creationDate'] = pd.to_datetime(sleep['creationDate'])
    sleep['date'] = sleep['creationDate'].map(get_date)
    sleep['startDate'] = pd.to_datetime(sleep['startDate'])
    sleep['endDate'] = pd.to_datetime(sleep['endDate'])
    sleep['timeDiff'] = (sleep['endDate'] - sleep['startDate'])

    # generate the last week's dates

    date_generator = (pd.Timestamp.today() - pd.Timedelta(days=i) for i in itertools.count())
    last_week_dates = itertools.islice(date_generator, days_back)
    last_week_dates = list(last_week_dates)
    #print(last_week_dates)
    # convert to something that can compare
    last_week_str = []
    for last_week in last_week_dates:
        if last_week.month < 10:
            month_str = "0"+str(last_week.month)
        else:
            month_str = str(last_week.month)
        if last_week.day < 10:
            day_str = "0"+str(last_week.day)
        else:
            day_str = str(last_week.day)
        last_week_str.append(str(last_week.year)+"-"+month_str+"-"+day_str)


    # calculate minutes of sleep a night

    sleep_by_day = sleep.groupby('date')[["timeDiff"]].sum()
    sleep_by_day = sleep_by_day.reset_index()

    for date_i in range(len(sleep_by_day)):
        sleep_by_day.at[date_i,'hours'] = sleep_by_day.at[date_i, 'timeDiff'].total_seconds()/60/60
    sleep_by_day['z_score'] = stats.zscore(sleep_by_day['hours'])

    # average heart rate variability by day

    hrv_by_day = hrv.groupby('date').mean()
    hrv_by_day = hrv_by_day.reset_index()
    hrv_by_day['z_score'] = stats.zscore(hrv_by_day['value'])
    #print(len(hrv.groupby('date')))
    # calculate overall mean of measures

    mean_hrv = hrv[["value"]].mean()
    mean_workout_dur = workouts[["duration"]].mean()
    mean_hr_sleep = sleep_by_day[["timeDiff"]].mean()
    # select out the recent period dates from the data set
    #print(hrv_by_day["date"].isin(last_week_str)[195:])
    #print(last_week_str)
    #last_week_hrv = hrv_by_day.loc[hrv_by_day["date"].isin(last_week_str)]
    last_week_hrv = hrv_by_day.loc[hrv_by_day["date"].isin(last_week_str)]
    last_week_workout = workouts[workouts["date"].isin(last_week_str)]
    last_week_workout = last_week_workout[['date', 'duration', 'z_score_duration']]
    last_week_hr_sleep = sleep_by_day.loc[sleep_by_day["date"].isin(last_week_str)]
    #print(len(last_week_hrv), len(last_week_hr_sleep))


    # get recent mean

    #last_week_mean_hrv = last_week_hrv[["value"]].mean()
    #last_week_mean_workout_dur = last_week_workout[["duration"]].mean()
    last_week_mean_hr_sleep = last_week_hr_sleep[["timeDiff"]].mean()

    hrv_res = np.array(last_week_hrv['value'])
    hr_sleep = np.array(last_week_hr_sleep['hours'])
    #print(len(hrv_res), len(hr_sleep))
    return(hrv_res, hr_sleep)




if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0',port='8000')
