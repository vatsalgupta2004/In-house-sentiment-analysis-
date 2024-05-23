from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples, stopwords, gutenberg
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier
import re, string, random
from prettytable import PrettyTable

def try_again():
    global content_try_store
    def remove_noise(tweet_tokens, stop_words = ()):

        cleaned_tokens = []

        for token, tag in pos_tag(tweet_tokens):
            token = re.sub("(@[A-Za-z0-9_]+)","", token)

            if tag.startswith("NN"):
                pos = 'n'
            elif tag.startswith('VB'):
                pos = 'v'
            else:
                pos = 'a'

            lemmatizer = WordNetLemmatizer()
            token = lemmatizer.lemmatize(token, pos)

            if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
                cleaned_tokens.append(token.lower())
        return cleaned_tokens

    def get_all_words(cleaned_tokens_list):
        for tokens in cleaned_tokens_list:
            for token in tokens:
                yield token

    def get_tweets_for_model(cleaned_tokens_list):
        for tweet_tokens in cleaned_tokens_list:
            yield dict([token, True] for token in tweet_tokens)

    def preprocess_gutenberg():
        stop_words = stopwords.words('english')
        
        positive_gutenberg_text = gutenberg.raw('austen-emma.txt')
        negative_gutenberg_text = gutenberg.raw('melville-moby_dick.txt')
        
        positive_sentences = sent_tokenize(positive_gutenberg_text)
        negative_sentences = sent_tokenize(negative_gutenberg_text)
        
        positive_cleaned_tokens_list = []
        negative_cleaned_tokens_list = []

        for sentence in positive_sentences:
            tokens = word_tokenize(sentence)
            positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

        for sentence in negative_sentences:
            tokens = word_tokenize(sentence)
            negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

        return positive_cleaned_tokens_list, negative_cleaned_tokens_list
    
    if __name__ == "__main__":

        stop_words = stopwords.words('english')

        positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
        negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')

        positive_cleaned_tokens_list = []
        negative_cleaned_tokens_list = []

        for tokens in positive_tweet_tokens:
            positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

        for tokens in negative_tweet_tokens:
            negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

        gutenberg_positive_cleaned_tokens, gutenberg_negative_cleaned_tokens = preprocess_gutenberg()

        positive_cleaned_tokens_list.extend(gutenberg_positive_cleaned_tokens)
        negative_cleaned_tokens_list.extend(gutenberg_negative_cleaned_tokens)

        all_pos_words = get_all_words(positive_cleaned_tokens_list)

        freq_dist_pos = FreqDist(all_pos_words)
        print(freq_dist_pos.most_common(20))

        positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
        negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)

        positive_dataset = [(tweet_dict, "Positive")
                            for tweet_dict in positive_tokens_for_model]

        negative_dataset = [(tweet_dict, "Negative")
                            for tweet_dict in negative_tokens_for_model]

        dataset = positive_dataset + negative_dataset

        random.shuffle(dataset)

        train_data = dataset[:7000]
        test_data = dataset[7000:]

        classifier = NaiveBayesClassifier.train(train_data)

        print(" ")
        print("Accuracy is:", classify.accuracy(classifier, test_data))
        print(classifier.show_most_informative_features(10))

        user_message = input("\nEnter your desired sentence to check whether positive or negative: ")

        custom_tokens = remove_noise(word_tokenize(user_message))

        print(f"\nSentence:--> {user_message}")
        print(f"Sentiment Type:--> {classifier.classify(dict([token, True] for token in custom_tokens))}")
        content_try_store.update({user_message:classifier.classify(dict([token, True] for token in custom_tokens))})

logo=''' 
                       _                                             _                        
                  _   (_)                  _                        | |                       
  ___  ____ ____ | |_  _ ____   ____ ____ | |_      ____ ____   ____| |_   _ _____ ____  ____ 
 /___)/ _  )  _ \|  _)| |    \ / _  )  _ \|  _)    / _  |  _ \ / _  | | | | (___  ) _  )/ ___)
|___ ( (/ /| | | | |__| | | | ( (/ /| | | | |__   ( ( | | | | ( ( | | | |_| |/ __( (/ /| |    
(___/ \____)_| |_|\___)_|_|_|_|\____)_| |_|\___)   \_||_|_| |_|\_||_|_|\__  (_____)____)_|    
                                                                      (____/                 
'''
print(logo)
table = PrettyTable(["Sno","sentence","sentiment"])
content_try_store={}
condition= True
while(condition):
    try_check=(input("\nDo you want to continue with the sentiment analyser program (yes/no): ").lower())
    if(try_check=="yes"):
        condition=True
        try_again()
    else:
        condition= False
        print("\nAll your searches and their results from today are displayed below -->")
        num=1
        for sentence,sentiment in content_try_store.items():
            table.add_row([num,sentence,sentiment])
            num+=1
        table.align='l'
        print(table)
        content_try_store={}
        print("\nThank you for visiting hope you are satisfied with our Sentiment Analyser project")
        print("\t\t\t😎 ^_^ ^_^ ^_^ 😎\n")