## Nathan Chen
import math, os, pickle, re
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
import numpy as np

class BayesClassifier:
    """A simple BayesClassifier implementation

    Attributes:
        pos_freqs - dictionary of frequencies of positive words
        neg_freqs - dictionary of frequencies of negative words
        pos_filename - name of positive dictionary cache file
        neg_filename - name of positive dictionary cache file
        training_data_directory - relative path to training directory
        neg_file_prefix - prefix of negative reviews
        pos_file_prefix - prefix of positive reviews
    """

    def __init__(self):
        """Constructor initializes and trains the Naive Bayes Sentiment Classifier. If a
        cache of a trained classifier is stored in the current folder it is loaded,
        otherwise the system will proceed through training.  Once constructed the
        classifier is ready to classify input text."""
        # initialize attributes
        self.pos_freqs: Dict[str, int] = {}
        self.neg_freqs: Dict[str, int] = {}
        self.pos_filename: str = "pos.dat"
        self.neg_filename: str = "neg.dat"
        self.training_data_directory: str = "movie_reviews/"
        self.neg_file_prefix: str = "movies-1"
        self.pos_file_prefix: str = "movies-5"

        # check if both cached classifiers exist within the current directory
        if os.path.isfile(self.pos_filename) and os.path.isfile(self.neg_filename):
            print("Data files found - loading to use cached values...")
            self.pos_freqs = self.load_dict(self.pos_filename)
            self.neg_freqs = self.load_dict(self.neg_filename)
        else:
            print("Data files not found - running training...")
            self.train()

    def train(self) -> None:
        """Trains the Naive Bayes Sentiment Classifier

        Train here means generates `pos_freq/neg_freq` dictionaries with frequencies of
        words in corresponding positive/negative reviews
        """
        # get the list of file names from the training data directory
        # os.walk returns a generator (feel free to Google "python generators" if you're
        # curious to learn more, next gets the first value from this generator or the
        # provided default `(None, None, [])` if the generator has no values)
        _, __, files = next(os.walk(self.training_data_directory), (None, None, []))
        if not files:
            raise RuntimeError(f"Couldn't find path {self.training_data_directory}")

        # files now holds a list of the filenames
        # self.training_data_directory holds the folder name where these files are

        # stored below is how you would load a file with filename given by `filename`
        # `text` here will be the literal text of the file (i.e. what you would see
        # if you opened the file in a text editor
        # text = self.load_file(os.path.join(self.training_data_directory, files[3]))
        # print(text)

        # *Tip:* training can take a while, to make it more transparent, we can use the
        # enumerate function, which loops over something and has an automatic counter.
        # write something like this to track progress (note the `# type: ignore` comment
        # which tells mypy we know better and it shouldn't complain at us on this line):
        
        # Create a list of stopwords
        file = self.load_file("sorted_stoplist.txt")
        # print(file)
        stopwords = self.tokenize(file)
        print(stopwords)
        
        for index, filename in enumerate(files, 1): # type: ignore
            print(f"Training on file {index} of {len(files)}")
        #     <the rest of your code for updating frequencies here>
            text = self.load_file(os.path.join(self.training_data_directory, filename))
            tokens = self.tokenize(text)
            # print(tokens)

            filtered_tokens = [token for token in tokens if token not in stopwords]
            # print(filtered_tokens)

            if filename.startswith(self.pos_file_prefix):
                self.update_dict(filtered_tokens, self.pos_freqs)
            elif filename.startswith(self.neg_file_prefix):
                self.update_dict(filtered_tokens, self.neg_freqs)

        # print(self.pos_freqs["awesome"])
        # print(self.neg_freqs["awesome"])
        # print(self.pos_freqs["great"])
        # print(self.neg_freqs["great"])
        # print(self.pos_freqs["the"])
        # print(self.neg_freqs["the"])

        # we want to fill pos_freqs and neg_freqs with the correct counts of words from
        # their respective reviews
        
        # for each file, if it is a negative file, update (see the Updating frequencies
        # set of comments for what we mean by update) the frequencies in the negative
        # frequency dictionary. If it is a positive file, update (again see the Updating
        # frequencies set of comments for what we mean by update) the frequencies in the
        # positive frequency dictionary. If it is neither a postive or negative file,
        # ignore it and move to the next file (this is more just to be safe; we won't
        # test your code with neutral reviews)
        

        # Updating frequences: to update the frequencies for each file, you need to get
        # the text of the file, tokenize it, then update the appropriate dictionary for
        # those tokens. We've asked you to write a function `update_dict` that will make
        # your life easier here. Write that function first then pass it your list of
        # tokens from the file and the appropriate dictionary
        

        # for debugging purposes, it might be useful to print out the tokens and their
        # frequencies for both the positive and negative dictionaries
        

        # once you have gone through all the files, save the frequency dictionaries to
        # avoid extra work in the future (using the save_dict method). The objects you
        # are saving are self.pos_freqs and self.neg_freqs and the filepaths to save to
        # are self.pos_filename and self.neg_filename
        self.save_dict(self.pos_freqs, self.pos_filename)
        self.save_dict(self.neg_freqs, self.neg_filename)



    def classify(self, text: str) -> str:
        """Classifies given text as positive, negative or neutral from calculating the
        most likely document class to which the target string belongs

        Args:
            text - text to classify

        Returns:
            classification, either positive, negative or neutral
        """
        # TODO: fill me out
        
        # get a list of the individual tokens that occur in text
        tokens = self.tokenize(text)
        print(tokens)

        file = self.load_file("sorted_stoplist.txt")
        stopwords = self.tokenize(file)


        # create some variables to store the positive and negative probability. since
        # we will be adding logs of probabilities, the initial values for the positive
        # and negative probabilities are set to 0
        pos_score = 0
        neg_score = 0

        # get the sum of all of the frequencies of the features in each document class
        # (i.e. how many words occurred in all documents for the given class) - this
        # will be used in calculating the probability of each document class given each
        # individual feature
        pos_total = sum(self.pos_freqs.values())
        # print(pos_total)
        neg_total = sum(self.neg_freqs.values())
        # print(neg_total)
        
        # Creating the entire vocab and finding the size
        vocab = set(self.pos_freqs.keys()).union(self.neg_freqs.keys())
        vocab_size = len(vocab)

        # for each token in the text, calculate the probability of it occurring in a
        # postive document and in a negative document and add the logs of those to the
        # running sums. when calculating the probabilities, always add 1 to the numerator
        # of each probability for add one smoothing (so that we never have a probability
        # of 0)
        for token in tokens:
            if token not in stopwords:
                pos_freqs = self.pos_freqs.get(token, 0) + 1
                neg_freqs = self.neg_freqs.get(token, 0) + 1

                # print(pos_freqs, neg_freqs)

                pos_score += math.log(pos_freqs / (pos_total + vocab_size))
                neg_score += math.log(neg_freqs / (neg_total + vocab_size))

        print(pos_score, neg_score)


        # for debugging purposes, it may help to print the overall positive and negative
        # probabilities
        # print(pos_score, neg_score)

        print(f"Positive Probability: {pos_score}")
        print(f"Negative Probability: {neg_score}")

        # determine whether positive or negative was more probable (i.e. which one was
        # larger)
        if pos_score > neg_score:
            return "positive"
        else:
            return "negative"

        # return a string of "positive" or "negative"

    def load_file(self, filepath: str) -> str:
        """Loads text of given file

        Args:
            filepath - relative path to file to load

        Returns:
            text of the given file
        """
        with open(filepath, "r", encoding='utf8') as f:
            return f.read()

    def save_dict(self, dict: Dict, filepath: str) -> None:
        """Pickles given dictionary to a file with the given name

        Args:
            dict - a dictionary to pickle
            filepath - relative path to file to save
        """
        print(f"Dictionary saved to file: {filepath}")
        with open(filepath, "wb") as f:
            pickle.Pickler(f).dump(dict)

    def load_dict(self, filepath: str) -> Dict:
        """Loads pickled dictionary stored in given file

        Args:
            filepath - relative path to file to load

        Returns:
            dictionary stored in given file
        """
        print(f"Loading dictionary from file: {filepath}")
        with open(filepath, "rb") as f:
            return pickle.Unpickler(f).load()

    def tokenize(self, text: str) -> List[str]:
        """Splits given text into a list of the individual tokens in order

        Args:
            text - text to tokenize

        Returns:
            tokens of given text in order
        """
        tokens = []
        token = ""
        for c in text:
            if (
                re.match("[a-zA-Z0-9]", str(c)) != None
                or c == "'"
                or c == "_"
                or c == "-"
            ):
                token += c
            else:
                if token != "":
                    tokens.append(token.lower())
                    token = ""
                if c.strip() != "":
                    tokens.append(str(c.strip()))

        if token != "":
            tokens.append(token.lower())
        return tokens

    def update_dict(self, words: List[str], freqs: Dict[str, int]) -> None:
        """Updates given (word -> frequency) dictionary with given words list

        By updating we mean increment the count of each word in words in the dictionary.
        If any word in words is not currently in the dictionary add it with a count of 1.
        (if a word is in words multiple times you'll increment it as many times
        as it appears)

        Args:
            words - list of tokens to update frequencies of
            freqs - dictionary of frequencies to update
        """
        # TODO: your work here
        # print("update dict")
        for word in words:
            if word in freqs:
                freqs[word] += 1
            else:
                freqs[word] = 1

    def generate_netflix_logo(self):

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_facecolor('black')
        fig.patch.set_facecolor('black')
        ax.axis('off')

    # Create the red bars that form the stylized 'N'
        bar_width = 0.2

    # Left red bar
        ax.add_patch(plt.Rectangle((0.2, 0.2), bar_width, 0.6, color="#E50914"))

    # Right red bar
        ax.add_patch(plt.Rectangle((0.6, 0.2), bar_width, 0.6, color="#E50914"))

    # Diagonal red bar (the middle slant of the N)
        x = np.array([0.2, 0.4, 0.6])
        y = np.array([0.8, 0.2, 0.8])
        ax.plot(x, y, color="#E50914", linewidth=30, solid_capstyle='butt')

        plt.title("Netflix", fontsize=30, color="white", weight="bold", pad=20)
        plt.show()

    def interactive_classification(self):
        print("Type 'exit' to quit.")
        while True:
            review = input("Enter your review: ")
            if review.lower() == "exit":
                break
            print(f"You typed: {review}")
            prediction = self.classify(review)
            print(f"Prediction: {prediction}")

if __name__ == "__main__":
    # uncomment the below lines once you've implemented `train` & `classify`
    b = BayesClassifier()
    a_list_of_words = ["I", "really", "like", "this", "movie", ".", "I", "hope", \
                       "you", "like", "it", "too"]
    a_dictionary = {}
    b.update_dict(a_list_of_words, a_dictionary)
    assert a_dictionary["I"] == 2, "update_dict test 1"
    assert a_dictionary["like"] == 2, "update_dict test 2"
    assert a_dictionary["really"] == 1, "update_dict test 3"
    assert a_dictionary["too"] == 1, "update_dict test 4"
    print("update_dict tests passed.")

    pos_denominator = sum(b.pos_freqs.values())
    neg_denominator = sum(b.neg_freqs.values())

    print("\nThese are the sums of values in the positive and negative dicitionaries.")
    print(f"sum of positive word counts is: {pos_denominator}")
    print(f"sum of negative word counts is: {neg_denominator}")

    print("\nHere are some sample word counts in the positive and negative dicitionaries.")
    print(f"count for the word 'love' in positive dictionary {b.pos_freqs['love']}")
    print(f"count for the word 'love' in negative dictionary {b.neg_freqs['love']}")
    print(f"count for the word 'terrible' in positive dictionary {b.pos_freqs['terrible']}")
    print(f"count for the word 'terrible' in negative dictionary {b.neg_freqs['terrible']}")
    print(f"count for the word 'computer' in positive dictionary {b.pos_freqs['computer']}")
    print(f"count for the word 'computer' in negative dictionary {b.neg_freqs['computer']}")
    print(f"count for the word 'science' in positive dictionary {b.pos_freqs['science']}")
    print(f"count for the word 'science' in negative dictionary {b.neg_freqs['science']}")
    # print(f"count for the word 'i' in positive dictionary {b.pos_freqs['i']}")
    # print(f"count for the word 'i' in negative dictionary {b.neg_freqs['i']}")
    # print(f"count for the word 'is' in positive dictionary {b.pos_freqs['is']}")
    # print(f"count for the word 'is' in negative dictionary {b.neg_freqs['is']}")
    # print(f"count for the word 'the' in positive dictionary {b.pos_freqs['the']}")
    # print(f"count for the word 'the' in negative dictionary {b.neg_freqs['the']}")

    print("\nHere are some sample probabilities.")
    print(f"P('love'| pos) {(b.pos_freqs['love']+1)/pos_denominator}")
    print(f"P('love'| neg) {(b.neg_freqs['love']+1)/neg_denominator}")
    print(f"P('terrible'| pos) {(b.pos_freqs['terrible']+1)/pos_denominator}")
    print(f"P('terrible'| neg) {(b.neg_freqs['terrible']+1)/neg_denominator}")

    # uncomment the below lines once you've implemented `classify`
    print("\nThe following should all be positive.")
    print(b.classify('I love computer science'))
    print(b.classify('this movie is fantastic'))
    print("\nThe following should all be negative.")
    print(b.classify('rainy days are the worst'))
    print(b.classify('computer science is terrible')) 
    print(b.classify("No way should this have beaten Traffic for best movie."))

    print()

    print("\nTHE FOLLOWING IS TO TEST OUT THE METHOD WITH EACH GROUPS RESPONCSES")
    print(b.classify('Summer break is almost here.  I am super excited and I know that its going to be the best'))
    print(b.classify('This was crazy incredible. Absolutely stunning cast.'))
    print(b.classify('I cant believe how amazing this movie is! Definitely a top 10.'))
    print(b.classify("I am nervous that I won't do well on the AP tests.  I have studied, but I don't think I'll do that well"))
    print(b.classify('I hate this stupid math class.'))
    print(b.classify('I failed the AP test. It was a very bad day.'))
    b.interactive_classification()
    pass