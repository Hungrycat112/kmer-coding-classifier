# kmer-coding-classifier

# kmer-coding-classifier
Predicts whether a DNA sequence is coding or noncoding using k-mer frequencies + ML 

This is a bioinformatics tool that classifies short DNA sequences (e.g. 300bp) as coding or noncoding, based on their k-mer frequency profiles. The model is implemented from scratch in Python using only NumPy, and aims to help explore simple, interpretable ML in genomics.

Features
- extract k-mer counts from raw DNA sequences
- vectorizes sequences into feature matrices
- train a naive bayes / logistic regression model 
- predict class (coding vs noncoding) on new sequences

Example use case: 
$ python predict.py --seq ATGGCTTAGGCTACG... --k 4
Prediction: Coding

Goals: 
- practice building interpretable ML model 
- explore signal embedded in raw k-mer frequences
- make bio ML tools more accessible 
