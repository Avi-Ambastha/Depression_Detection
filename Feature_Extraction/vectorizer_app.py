from sklearn.feature_extraction.text import TfidfVectorizer
class custom_vectorization:
    def __init__(self):
        self.vectorizer=TfidfVectorizer()
    def vectorize(self,series):
        return self.vectorizer.fit_transform(series)
