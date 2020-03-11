# NLPproject for the Information Retrieval course

* Andrea Alfieri - 5128315
* Diego Albo - 5043204
* Tomasz Motyka - 5146844
* Avinash Saravanan - 4993381

# Board Management
* If you want to assign someone to a card, convert the card to an issue.
* To ensure that the board carries out automated actions make sure you associate pull requests with issues.

# Directory Structure (Explanation of Folders and Files)
1. Data
  1. CSV Files
    1. Contains the original emergent data from which we are working. The cleaned file was cleaned to remove any extraneous data and invalid information.
  2. Pickled Features
    1. Contains features that have been extracted. Can be read by using Pandas or Pickle.
  3. PPDB (Paraphrase Database)
2. Data Reading
  1. Contains files used for reading data in from the data folder.
3. Evaluation
  1. Contains files used for classifier training.
4. Feature Extraction
  1. Contains files used to extract features to their respective pickle files.
5. Feature Selection
  1. Contains ablation tests, statistical tests, and forward/backward selection which are used to determine the usefulness of features.
6. Visualization
  1. Contains scripts used to generate graphs and charts from extracted features.


## Useful Links
* [Paper being reproduced](https://www.aclweb.org/anthology/N16-1138/)
* [Paper repo](https://github.com/willferreira/mscproject)
* [Report Link](https://www.overleaf.com/read/ntwnpkpxmxvw)
* [NLTK](https://www.nltk.org/)
* [SciKit](https://scikit-learn.org/)
* [Pickle](https://docs.python.org/3/library/pickle.html)
