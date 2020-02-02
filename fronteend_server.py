from flask import Flask, request, Response,render_template
import json
import time

PATH_TO_TEST_IMAGES_DIR = './images'

app = Flask(__name__,template_folder='.')

@app.route('/journal',methods=['GET'])
def journal():

	#return render_template('journal.html')
    return Response(open('journal.html').read(), mimetype="text/html")
@app.route('/')
def login():
	return render_template('login.html')

@app.route('/result')
def result():

	return render_template('result.html')

@app.route('/getresults')
def getresults():
	data = json.dumps([
  {
    "keywords": [
      "images",
      "sharp sacrifice",
      "hot sacrifice",
      "cold sacrifice",
      "saddest bunch",
      "man",
      "breaking point",
      "bug"
    ]
  },
  {
    "emotions": [
      "Undefined",
      "Neutral",
      "Neutral",
      "Neutral",
      "Sad",
      "Sad",
      "Neutral",
      "Neutral",
      "Neutral",
      "Neutral",
      "Neutral",
      "Neutral",
      "Sad",
      "Neutral",
      "Sad"
    ]
  },
  {
    "heartratev": [
      "77.58956",
      "70.37764000000001",
      "83.06665",
      "100.91186666666665",
      "77.94065",
      "86.5041",
      "86.89715",
      "87.69796",
      "76.68482",
      "70.30373333333334",
      "64.9392",
      "171.347",
      "79.963075",
      "56.4209"
    ]
  },
  {
    "sleephrs": [
      "9.266666666666667",
      "7.216666666666667",
      "9.833333333333334",
      "6.916666666666667",
      "8.633333333333333",
      "7.916666666666667",
      "9.833333333333334",
      "8.533333333333333",
      "8.35",
      "9.283333333333333",
      "7.35",
      "7.9",
      "8.5",
      "4.833333333333333"
    ]
  },
  {
    "emotive_score": [
      "-2",
      "-1",
      "-1",
      "-1",
      "2",
      "-5",
      "-1",
      "2",
      "2",
      "2",
      "2",
      "-1",
      "-5",
      "2",
      "-5"
    ]
  },
  {
    "weekly_tally": [
      "-10"
    ]
  }
])
	return data

@app.route('/test/', methods=['POST'])
def test():
	data = request.json
	print(data)
	return 'success'
# save the image as a picture
@app.route('/image', methods=['POST'])
def image():

    i = request.files['image']  # get the image
    f = ('%s.jpeg' % time.strftime("%Y%m%d-%H%M%S"))
    i.save('%s/%s' % (PATH_TO_TEST_IMAGES_DIR, f))

    return Response("%s saved" % f)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0',port='5000')
