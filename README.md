# NLP project for the Information Retrieval course

The goal of the project was to reproduce this [paper](https://www.aclweb.org/anthology/N16-1138/) on *stance classification*. The system has been trained to determine the article headline stance (*for*, *against* or *observing*) with respect to the claim. The team was able to reproduce the entire research and actually improve the model accuracy.

A final report can be found in [this paper]().

* Andrea Alfieri - [@andreaalf97](http://github.com/andreaalf97)
* Diego Albo - [@DiegoStock12](http://github.com/DiegoStock12)
* Tomasz Motyka - [@motykatomasz](http://github.com/motykatomasz)
* Avinash Saravanan - [@asarav](http://github.com/asarav)

# Board Management
* If you want to assign someone to a card, convert the card to an issue.
* To ensure that the board carries out automated actions make sure you associate pull requests with issues.

# Directory Structure (Explanation of Folders and Files)
1. Data
    * CSV Files
        * Contains the original emergent data from which we are working. The cleaned file was cleaned to remove any extraneous data and invalid information.
    * Pickled Features
        * Contains features that have been extracted. Can be read by using Pandas or Pickle.
    * PPDB (Paraphrase Database)
2. Data Reading
    * Contains files used for reading data in from the data folder.
3. Evaluation
    * Contains files used for classifier training.
4. Feature Extraction
    * Contains files used to extract features to their respective pickle files.
5. Feature Selection
    * Contains ablation tests, statistical tests, and forward/backward selection which are used to determine the usefulness of features.
6. Visualization
    * Contains scripts used to generate graphs and charts from extracted features.


## Useful Links
* [Paper being reproduced](https://www.aclweb.org/anthology/N16-1138/)
* [Paper repo](https://github.com/willferreira/mscproject)
* [Report Link](https://www.overleaf.com/read/ntwnpkpxmxvw)
* [NLTK](https://www.nltk.org/)
* [SciKit](https://scikit-learn.org/)
* [Pickle](https://docs.python.org/3/library/pickle.html)
