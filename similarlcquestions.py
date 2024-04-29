from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import defaultdict

lineMap = defaultdict(str)
corpus = []
with open('output.txt', 'r') as file:
    # Iterate over each line in the file
    i = 0
    for line in file:
        i += 1
        # Process each line as needed
        corpus.append(line.strip())
        lineMap[i] = line.strip()

# print(corpus)
previousInserted = False

print("\n")
while True:
    values = defaultdict(float)
    user_input = input("Enter a leetcode question or topic (type 'quit' to exit): ")
    print("\n")
    if user_input.lower() == 'quit':
        print("Thanks for using this tool. Happy Leetcoding!\n")
        break

    print("Related Questions:\n")
    print("Question ID | Question Title | Similarity Score (lowest 0 to 1 highest)\n")

    # corpus.append(user_input)
    # print(corpus[-1])
    if previousInserted == True:
         corpus[0] = user_input
    else:
        corpus.insert(0, user_input)
        previousInserted = True

    vectorizer = CountVectorizer()
    features = vectorizer.fit_transform(corpus).todense()
    # print(vectorizer.vocabulary_)

    i = 0
    for f in features[1:]:
        i += 1
        value = cosine_similarity(np.asarray(features[0]), np.asarray(f))
        values[lineMap[i]] = value[0][0]
        # if values[0][0] < 2.3:
        #     print(lineMap[i], "  ", euclidean_distances(np.asarray(features[-1]), np.asarray(f)))
    sorted_dict = dict(sorted(values.items(), key=lambda item: item[1], reverse=True))
    j = 0
    highestVal = 0
    for key, value in list(sorted_dict.items())[:10]:
        if value != 0.0:
            highestVal = max(highestVal, value)
            print(f'{key}: {value}')
            j += 1
    if j == 0:
        print("No related questions were found. Try to refine the question or topic")
        print("Example: Dynamic Programming -> Longest Common Subsequence")
    elif highestVal < 0.4:
        print("\nThe similarity score is low: Try to refine the question or topic")
        print("\nExample: Segement Trees -> Range Sum Query")
    print("\n")