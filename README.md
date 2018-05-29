# Fake News Challenge
_Prajwal Rao (5176504) & Julian Blair (3463793) | COMP9417 2018s1_

## Acknowledgement
The Fake News Challenge was hosted in 2017 by a group of academic and industry volunteers. Learn more about the challenge [here][1].

Thanks to the same team for providing a baseline implementation, which was used as a starting point for our project. The GitHub repository for the baseline can be found [here][2]. 

## Install
Python packages can be installed via commandline using:

`pip install [packagename]`

NLTK packages can be installed within Python using:

`import nltk
nltk.download('[packagename]')`

### Prerequisites
* Python 3
* _pip_ packages
   * numpy
   * scipy
   * sklearn
   * tqdm
   * nltk (see below)
* _nltk_ packages
   * punkt
   * wordnet
   * averaged_perceptron_tagger
   
   [1]: http://www.fakenewschallenge.org/ "Fake News Challenge"
   [2]: https://github.com/FakeNewsChallenge/fnc-1-baseline "Baseline FNC implementation"

## Navigation
The _data_ folder contains the CSVs provided for the challenge as training data, testing data, and competition benchmarking data.

The _src_ folder contains the baseline code provided, as well as implementations built upon the baseline by us. 

### Source subfolders
| Subfolder        | Version | Description
| ---------------- | ------- | -----------
| **baseline**     | 0       | The baseline provided.
| **word_overlap** | 1       | Restructures the classificaiton problem from multi-class to multi-tier two-class, and modifies the word_overlap feature to filter common words.
| **paraphrasing** | 2       | **TODO: WRITE DESCRIPTION**
| **final**        | 3       | Adds Naive Bayes classifier and all caps frequency feature.

Each source subfolder contains the following files and folders:

| Item             | Description
| ---------------- | -----------
| **fnc_kfold.py** | The main execution script. Extracts CSVs, generates data splits for training/testing, precomputes features, then fits the hold-out set and test data.
| **feature_engineering.py** | Contains helper functions for feature precomputation.
| **features/**    | Contains precomputed feature files for later use in fitting the provided data.
| **utils/**       | Contains helper functions provided in baseline for dataset generation, test split creation, and scoring.

## Execution
To run the feature generation and scoring process, navigate to any of these source subfolders and run the following commnad:

`python|python3 fnc_kfold.py`
