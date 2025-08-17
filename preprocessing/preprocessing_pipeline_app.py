import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk
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
    def clean_text(self,text):
        return re.sub(r"[^a-zA-Z\s]","",text).lower()
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
        return lemmatized_tokens
