import math, os, pickle, re
from turtle import pos
from typing import Tuple, List, Dict

class BayesClassifier:

    def __init__(self):
        # initialize attributes
        self.pos_freqs: Dict[str, int] = {}
        self.neg_freqs: Dict[str, int] = {}
        self.pos_filename: str = "pos.dat"
        self.neg_filename: str = "neg.dat"
        self.pos_training_data_directory: str = "pos_review.txt"
        self.pos_training_data_directory: str = "neg_review.txt"

        # check if both cached classifiers exist within the current directory
        if os.path.isfile(self.pos_filename) and os.path.isfile(self.neg_filename):
            print("Data files found - loading to use cached values...")
            self.pos_freqs = self.load_dict(self.pos_filename)
            self.neg_freqs = self.load_dict(self.neg_filename)
            # print(self.pos_freqs)
        else:
            print("Data files not found - running training...")
            self.train()

    def train(self) -> None:
        with open("pos_review.txt", "r", encoding="utf-8") as file:
            for line in file:
                text = line
                token = self.tokenize(text)
                self.update_dict(token, self.pos_freqs)
        file.close()

        with open("neg_review.txt", "r", encoding="utf-8") as file:
            for line in file:
                text = line
                token = self.tokenize(text)
                self.update_dict(token, self.neg_freqs)
        file.close()

        self.save_dict(self.pos_freqs, self.pos_filename)
        self.save_dict(self.neg_freqs, self.neg_filename)

    def classify(self, text: str) -> str:
        
        tokens = self.tokenize(text)
        
        pos_prob = 0
        neg_prob = 0
        
        num_pos_words = sum(self.pos_freqs.values())
        num_neg_words = sum(self.neg_freqs.values())
        
        for word in tokens:
            num_pos_app = 1
            if word in self.pos_freqs:
                num_pos_app += self.pos_freqs[word]

            pos_prob += math.log(num_pos_app/num_pos_words)

            num_neg_app = 1
            if word in self.neg_freqs:
                num_neg_app += self.neg_freqs[word]

            neg_prob += math.log(num_neg_app/num_neg_words)


        if pos_prob > neg_prob:
            return "positive"
        else:
            return "negative"
        

    def load_file(self, filepath: str) -> str:
        with open(filepath, "r", encoding='utf8') as f:
            return f.read()

    def save_dict(self, dict: Dict, filepath: str) -> None:
        print(f"Dictionary saved to file: {filepath}")
        with open(filepath, "wb") as f:
            pickle.Pickler(f).dump(dict)

    def load_dict(self, filepath: str) -> Dict:
        print(f"Loading dictionary from file: {filepath}")
        with open(filepath, "rb") as f:
            return pickle.Unpickler(f).load()

    def tokenize(self, text: str) -> List[str]:
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
        for word in words:
            if word in freqs:
                freqs[word] += 1
            else:
                freqs[word] = 1

print("")
print("--------------------Classifier Below--------------------")
print("")

b = BayesClassifier()

print("")

text = input("What do you want to classify?\n")
print(b.classify(text))