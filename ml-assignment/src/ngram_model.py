import random
import re

class TrigramModel:
    def __init__(self):
        self.trigrams = {}
        self.tokens = []

    def _tokenize(self, text):
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        tokens = text.split()
        return tokens

    def fit(self, text):
        # Handle empty input
        if not text or text.strip() == "":
            self.tokens = []
            self.trigrams = {}
            return

        # Tokenize
        self.tokens = self._tokenize(text)

        # Not enough tokens → no trigram possible
        if len(self.tokens) < 3:
            self.trigrams = {}
            return

        # Build trigrams
        self.trigrams = {}
        for i in range(len(self.tokens) - 2):
            key = (self.tokens[i], self.tokens[i+1])
            next_word = self.tokens[i+2]

            if key not in self.trigrams:
                self.trigrams[key] = []

            self.trigrams[key].append(next_word)

    def generate(self, max_words=20):
        # For empty text, return empty string (tests expect this)
        if not self.tokens:
            return ""

        # If no trigram pairs, still return something
        if not self.trigrams:
            return " ".join(self.tokens)

        # Start generation from a random bigram
        current_bigram = random.choice(list(self.trigrams.keys()))
        output_words = [current_bigram[0], current_bigram[1]]

        # Generate words
        for _ in range(max_words):
            next_options = self.trigrams.get(current_bigram)
            if not next_options:
                break

            next_word = random.choice(next_options)
            output_words.append(next_word)

            current_bigram = (current_bigram[1], next_word)

        return " ".join(output_words)
