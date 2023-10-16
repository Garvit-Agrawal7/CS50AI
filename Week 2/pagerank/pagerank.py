import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    # if len(sys.argv) != 2:
    #     sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(input("enter directory: "))
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    # A dictionary containing all the webpages in the corpus
    probability_dict = {webpage: 0 for webpage in corpus.keys()}
    curr_webpage = corpus[page]

    # If set is empty, probability is equal for all instances
    if not curr_webpage:
        probability_dict.update((k, 1/len(corpus)) for k in probability_dict)
        return probability_dict

    random_probability = (1-damping_factor) / len(corpus)
    links_probability = damping_factor / len(curr_webpage)

    # Updating Probability dictionary
    for k in probability_dict.keys():
        probability_dict[k] += random_probability
        if k in curr_webpage:
            probability_dict[k] += links_probability
    return probability_dict


def sample_pagerank(corpus, damping_factor, n):
    """
        Return PageRank values for each page by sampling `n` pages
        according to transition model, starting with a page at random.

        Return a dictionary where keys are page names, and values are
        their estimated PageRank value (a value between 0 and 1). All
        PageRank values should sum to 1.
        """
    # A dictionary containing all the webpages in the corpus
    pagerank = {webpage: 0 for webpage in corpus.keys()}

    # Choosing a webpage at random and adding 1 ten-thousandth to it
    curr_webpage = random.choice(list(pagerank.keys()))
    pagerank[curr_webpage] += 1 / n

    for _ in range(n - 1):
        model = transition_model(corpus, curr_webpage, damping_factor)
        # Choose the next page based on the transition model probabilities
        curr_webpage = random.choices(list(model.keys()), weights=list(model.values()))[0]

        # Update PageRank for the chosen page
        pagerank[curr_webpage] += 1 / n
    return pagerank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    N = 1/len(corpus)
    prob_dict = {webpage: N for webpage in corpus.keys()}

    # Indifference
    threshold = 0.001
    total_pages = len(corpus)

    while True:
        # Create a copy of the current Probability Dictionary values for comparison
        old_prob_dict = prob_dict.copy()
        for page in corpus:
            new_rank = (1 - damping_factor) / total_pages
            for link in corpus:
                if page in corpus[link]:
                    new_rank += damping_factor * old_prob_dict[link] / len(corpus[link])
                elif not corpus[link]:
                    new_rank += damping_factor * old_prob_dict[link] / total_pages
            prob_dict[page] = new_rank
        # Check if the change in Probability dictionary is less than the threshold
        if all(abs(prob_dict[page] - old_prob_dict[page]) < threshold for page in corpus):
            break
    return prob_dict


if __name__ == "__main__":
    main()
