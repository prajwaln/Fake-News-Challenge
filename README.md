# Fake News Challenge
_Prajwal Rao (5176504) & Julian Blair (3463793) | COMP9417 2018s1_

## Acknowledgement
The Fake News Challenge was hosted in 2017 by a group of academic and industry volunteers. Learn more about the challenge [here][1].

Thanks to the same team for providing a baseline implementation, which was used as a starting point for our project The GitHub repository for the baseline can be found [here][2]. 

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
   
   [1]: http://www.fakenewschallenge.org/ "Fake News Challenge"
   [2]: https://github.com/FakeNewsChallenge/fnc-1-baseline "Baseline FNC implementation"

## Execution
The _data_ folder contains the CSVs provided for the challenge as training data, testing data, and competition benchmarking data.

The _src_ folder contains the baseline code provided, as well as implementations built upon the baseline by us. 

### Source subfolders
| Subfolder        | Description
| ---------------- | -----------
| **baseline**     | The baseline provided.
| **word_overlap** | First implementation. Restructures the classificaiton problem from multi-class to multi-tier two-class, and modifies the word_overlap feature to filter common words.
