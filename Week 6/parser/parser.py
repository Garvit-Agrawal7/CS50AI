import nltk
import sys

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatic" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

NONTERMINALS = """
S -> NP VP | S Conj S | S Conj VP
NP -> N | Det N | Det AP N | P NP | NP P NP
VP -> V | Adv VP | V Adv | VP NP | V NP Adv
AP -> Adj | AP Adj      
"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():

    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    """
    Pre-processes sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """
    list_words = nltk.word_tokenize(sentence.lower())
    for word in list_words:
        if not word.isalpha():
            list_words.remove(word)
    return list_words


def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    """
    if isinstance(tree, nltk.Tree):
        if tree.label() == 'NP':
            if not any(sub.label() == 'NP' for sub in tree):
                return [tree]
        np_chunks = []
        for subtree in tree:
            result = np_chunk(subtree)
            if result is not None:
                np_chunks.extend(result)
        return np_chunks


if __name__ == "__main__":
    main()
