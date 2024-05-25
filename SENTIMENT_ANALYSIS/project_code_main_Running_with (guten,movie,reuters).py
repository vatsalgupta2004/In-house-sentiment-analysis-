from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples, stopwords, gutenberg, movie_reviews, reuters
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier
import re, string, random
from prettytable import PrettyTable
from project_code_logo import logo
 
def try_again():
    global content_try_store
    def remove_noise(tweet_tokens, stop_words = ()):

        cleansed_tokens = []

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
                cleansed_tokens.append(token.lower())
        return cleansed_tokens

    def get_all_words(cleansed_tokens_list):
        for tokens in cleansed_tokens_list:
            for token in tokens:
                yield token

    def get_tweets_for_model(cleansed_tokens_list):
        for tweet_tokens in cleansed_tokens_list:
            yield dict([token, True] for token in tweet_tokens)

    def preprocess_gutenberg():
        stop_words = stopwords.words('english')
        
        positive_gutenberg_text = gutenberg.raw('austen-emma.txt')
        negative_gutenberg_text = gutenberg.raw('melville-moby_dick.txt')
        
        positive_sentences = sent_tokenize(positive_gutenberg_text)
        negative_sentences = sent_tokenize(negative_gutenberg_text)
        
        positive_cleansed_tokens_list = []
        negative_cleansed_tokens_list = []

        for sentence_st in positive_sentences:
            tokens = word_tokenize(sentence_st)
            positive_cleansed_tokens_list.append(remove_noise(tokens, stop_words))

        for sentence_st in negative_sentences:
            tokens = word_tokenize(sentence_st)
            negative_cleansed_tokens_list.append(remove_noise(tokens, stop_words))

        return positive_cleansed_tokens_list, negative_cleansed_tokens_list
    
    def preprocess_movie_reviews():
        stop_words = stopwords.words('english')
        positive_reviews_data = []
        negative_reviews_data = []

        for fileid in movie_reviews.fileids('pos'):
            tokens = word_tokenize(movie_reviews.raw(fileid))
            positive_reviews_data.append(remove_noise(tokens, stop_words))

        for fileid in movie_reviews.fileids('neg'):
            tokens = word_tokenize(movie_reviews.raw(fileid))
            negative_reviews_data.append(remove_noise(tokens, stop_words))

        return positive_reviews_data, negative_reviews_data
    
    def preprocess_reuters_reviews():
        stop_words = stopwords.words('english')
        positive_reuters_data = []
        negative_reuters_data = []

        positive_reuters_category =["earn","acq","crude"]
        negative_reuters_category =["crude","trade","interest"]

        for category in positive_reuters_category:
            for fileid in reuters.fileids(category):
                tokens = word_tokenize(reuters.raw(fileid))
                positive_reuters_data.append(remove_noise(tokens, stop_words))

        for category in negative_reuters_category:
            for fileid in reuters.fileids(category):
                tokens = word_tokenize(reuters.raw(fileid))
                negative_reuters_data.append(remove_noise(tokens, stop_words))

        return positive_reuters_data, negative_reuters_data

    if __name__ == "__main__":

        stop_words = stopwords.words('english')

        positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
        negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')

        positive_cleansed_tokens_list = []
        negative_cleansed_tokens_list = []

        for tokens in positive_tweet_tokens:
            positive_cleansed_tokens_list.append(remove_noise(tokens, stop_words))

        for tokens in negative_tweet_tokens:
            negative_cleansed_tokens_list.append(remove_noise(tokens, stop_words))

        gutenberg_positive_cleansed_tokens, gutenberg_negative_cleansed_tokens = preprocess_gutenberg()
        movie_positive_cleansed_tokens, movie_negative_cleansed_tokens = preprocess_movie_reviews()
        reuters_positive_cleansed_tokens, reuters_negative_cleansed_tokens = preprocess_reuters_reviews()

        positive_cleansed_tokens_list.extend(gutenberg_positive_cleansed_tokens)
        negative_cleansed_tokens_list.extend(gutenberg_negative_cleansed_tokens)
        positive_cleansed_tokens_list.extend(movie_positive_cleansed_tokens)
        negative_cleansed_tokens_list.extend(movie_negative_cleansed_tokens)
        positive_cleansed_tokens_list.extend(reuters_positive_cleansed_tokens)
        negative_cleansed_tokens_list.extend(reuters_negative_cleansed_tokens)

        all_pos_words = get_all_words(positive_cleansed_tokens_list)

        freq_dist_pos = FreqDist(all_pos_words)
        print(freq_dist_pos.most_common(30))

        positive_tokens_for_model = get_tweets_for_model(positive_cleansed_tokens_list)
        negative_tokens_for_model = get_tweets_for_model(negative_cleansed_tokens_list)

        positive_dataset = [(tweet_dict, "Positive")
                            for tweet_dict in positive_tokens_for_model]

        negative_dataset = [(tweet_dict, "Negative")
                            for tweet_dict in negative_tokens_for_model]

        dataset = positive_dataset + negative_dataset

        random.shuffle(dataset)

        # slice the dataset for faster comparisons as dataset is now smaller
        # dataset=dataset[0:20000]

        train_data = dataset[:29000]
        test_data = dataset[29000:]

        classifier = NaiveBayesClassifier.train(train_data)

        print(" ")
        print("Accuracy is:", classify.accuracy(classifier, test_data))
        (classifier.show_most_informative_features(10))

        user_message = input("\nEnter your desired sentence to check whether positive or negative: ")

        custom_tokens = remove_noise(word_tokenize(user_message))

        print(f"\nSentence:--> {user_message}")
        print(f"Sentiment Type:--> {classifier.classify(dict([token, True] for token in custom_tokens))}")
        content_try_store.update({user_message:classifier.classify(dict([token, True] for token in custom_tokens))})


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
        get_sentiment=input("enter sentiments you want to print table of (positive,negative,all): ")
        print("\nAll your searches and their results from today are displayed below -->")
        num=1
        for sentence,sentiment in content_try_store.items():
            if((get_sentiment=="positive") and ((sentiment.lower())=="positive")):
                table.add_row([num,sentence,sentiment])
            elif((get_sentiment=="negative") and ((sentiment.lower())=="Negative")):
                table.add_row([num,sentence,sentiment])
            elif(get_sentiment == "all"):
                table.add_row([num, sentence, sentiment])
            num+=1
        table.align='l'
        print(table)
        content_try_store={}
        print("\nThank you for visiting hope you are satisfied with our Sentiment Analyser project")
        print("\t\t\t😎 ^_^ ^_^ ^_^ 😎\n")