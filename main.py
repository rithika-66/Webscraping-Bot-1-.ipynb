
import nltk
nltk.download('punkt')
import numpy as np
import random
import string
import bs4 as bs
import urllib.request
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def scrape_website(url):
    try:
        link = urllib.request.urlopen(url)
        link = link.read()
        data = bs.BeautifulSoup(link, 'lxml')
        data_paragraphs = data.find_all('p')
        data_text = ''
        for para in data_paragraphs:
            data_text += para.text
        return data_text
    except Exception as e:
        print(f"Error fetching data from the URL: {e}")
        return None


def preprocess_text(data_text):
   
    data_text = data_text.lower()
    
    data_text = re.sub(r'\[[0-9]*\]', ' ', data_text)
    
    data_text = re.sub(r'\s+', ' ', data_text)
    return data_text


wnlem = nltk.stem.WordNetLemmatizer()
def perform_lemmatization(tokens):
    return [wnlem.lemmatize(token) for token in tokens]


punctuation_removal = dict((ord(punctuation), None) for punctuation in string.punctuation)
def get_processed_text(document):
    return perform_lemmatization(nltk.word_tokenize(document.lower().translate(punctuation_removal)))


greeting_inputs = ("hey", "morning", "good morning", "evening", "good evening", "hi", "whatsup")
greeting_responses = ["hey", "I'm good", "Hey, how's you", "*nods*", "hello", "welcome I'm good and you"]

def generate_greeting_response(greeting):
    for token in greeting.split():
        if token.lower() in greeting_inputs:
            return random.choice(greeting_responses)


def generate_response(user_input, sentences):
    bot_response = ''
    sentences.append(user_input)
    word_vectorizer = TfidfVectorizer(tokenizer=get_processed_text, stop_words='english')
    all_word_vectors = word_vectorizer.fit_transform(sentences)
    similarity_vector_values = cosine_similarity(all_word_vectors[-1], all_word_vectors)
    similar_sentence_number = similarity_vector_values.argsort()[0][-2]
    matched_vector = similarity_vector_values.flatten()
    matched_vector.sort()
    vector_matched = matched_vector[-2]

    if vector_matched == 0:
        bot_response = bot_response + "I am sorry, I don't understand."
    else:
        bot_response = bot_response + sentences[similar_sentence_number]
    
    sentences.remove(user_input)
    return bot_response


def chatbot():
   
    url = input("Please enter the URL of the article you want to scrape: ")

    
    scraped_text = scrape_website(url)
    if not scraped_text:
        print("Failed to retrieve or process data from the provided URL.")
        return
    
    
    processed_text = preprocess_text(scraped_text)
    
   
    sentences = nltk.sent_tokenize(processed_text)
    
   
    print("Hello! You can now ask questions based on the content from the URL.")
    continue_flag = True
    while continue_flag:
        user_input = input("Ask your question or type 'bye' to exit: ").lower()

        if user_input == 'bye':
            continue_flag = False
            print("AI Bot says goodbye!")
        elif user_input in ['thanks', 'thank you']:
            continue_flag = False
            print("You're welcome!")
        else:
            if generate_greeting_response(user_input) is not None:
                print("AI Bot:", generate_greeting_response(user_input))
            else:
                print("AI Bot:", generate_response(user_input, sentences))

if __name__ == "__main__":
    chatbot()
