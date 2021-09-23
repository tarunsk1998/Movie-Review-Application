from flask import Flask, send_file, render_template, request, redirect
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
#nltk.download('punkt')

app = Flask(__name__, template_folder="templates")


@app.route('/', methods=['GET', 'POST'])
@app.route('/writereview', methods=['GET', 'POST'])
def home():
    if request.method == "GET":
        return render_template("index.html")
    else:
        movie_name = request.form.get("MovieName")
        movie_review = request.form.get("MovieReview")

        model = load_model('model/movie_review_model.h5')

        cleaned = re.compile(r'<.*?>')
        test_review_corpus = []
        sent_list = []
        test_sentences = nltk.sent_tokenize(movie_review)
        test_sentences = [''.join(re.sub(cleaned, '', sentence)) for sentence in test_sentences]
        test_sentences = [''.join(re.sub('[^a-zA-Z]', ' ', sentence)) for sentence in test_sentences]
        test_sent_list = []
        for sentence in test_sentences:
            sentence = sentence.lower()
            words = sentence.split()
            # words = [stemmer.stem(word) for word in words if not word in stopwords.words('english')]
            sent_list.append(' '.join(words))
        test_review_corpus.append(''.join(sent_list))

        vocabulary_size = 500
        test_one_hot_representation = [one_hot(words, vocabulary_size) for words in test_review_corpus]

        sent_len = 1000
        test_embeded_docs = pad_sequences(test_one_hot_representation, padding="pre", maxlen=sent_len)

        print("predicted value : ", model.predict(test_embeded_docs)[0])
        if (model.predict(test_embeded_docs)[0] <= 0.5):
            print("Negative review")
            review = "Negative"
        else:
            print("Positive review")
            review = "Positive"

        return render_template("result.html", review_result=review)


@app.route('/cancel', methods=['GET', 'POST'])
def cancel():
    return redirect("/")


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
