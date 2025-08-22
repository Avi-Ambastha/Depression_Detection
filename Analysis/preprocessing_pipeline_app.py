import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
class custom_preprocessing:
    def __init__(self):
        self.pos_map={
            'J':wordnet.ADJ,
            'N':wordnet.NOUN,
            'V':wordnet.VERB,
            'R':wordnet.ADV
        }
        self.stop_words=set(stopwords.words('english'))
        self.lemmatizer=WordNetLemmatizer()
        self.vectorizer=TfidfVectorizer()
    def clean_text(self,text):
        clean=re.sub(r'\b\w*(\w)\1{2,}\w*\b', '',text)
        return re.sub(r"[^a-zA-Z\s]","",clean).lower()
    def tokenize(self,text):
        return word_tokenize(text)
    def remove_stop_words(self,text):
        return [w for w in text if w not in self.stop_words] 
    def lemmatize(self,text):
        pos_tags=nltk.pos_tag(text)
        return [self.lemmatizer.lemmatize(word,self.pos_map.get(tag[0],wordnet.NOUN)) for word,tag in pos_tags]
    def preprocess(self,text):
        cleantext=self.clean_text(text)
        tokens=self.tokenize(cleantext)
        stop_words_removed=self.remove_stop_words(tokens)
        lemmatized_tokens=self.lemmatize(stop_words_removed)
        return " ".join(lemmatized_tokens)
    def vectorize(self,series):
        return self.vectorizer.fit_transform(series)
    def transform(self,array):
        return self.vectorizer.transform(array)
        
