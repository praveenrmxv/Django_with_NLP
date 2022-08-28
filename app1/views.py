from django.shortcuts import render
import pandas as  pd
from sklearn import metrics
from django.views.generic import TemplateView
import sklearn
def result(request):
    # Importing the dataset
    #dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)
    dataset = pd.read_csv('static/Restaurant_Reviews.tsv',delimiter='\t', quoting=3)

    # Cleaning the texts
    import re
    import nltk
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    corpus = []
    for i in range(0, 1000):
        review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        review = ' '.join(review)
        corpus.append(review)

    # Creating the Bag of Words model
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(max_features=1500)
    X = cv.fit_transform(corpus).toarray()
    y = dataset.iloc[:, 1].values

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

    # Fitting Naive Bayes to the Training set
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)

    from sklearn.metrics import accuracy_score
    accuracy=accuracy_score(y_test, y_pred, normalize=False)

    #d={'i':accuracy,'j':cm}
    d = {'i': metrics.accuracy_score(y_test, y_pred), 'j': metrics.confusion_matrix(y_test, y_pred)}
    return render(request,'restaurant.html',context=d)

###################################################################################
class Home(TemplateView):
    template_name = 'home.html'