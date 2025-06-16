# kmer-coding-classifier

# kmer-coding-classifier
Predicts whether a DNA sequence is coding or noncoding using k-mer frequencies + ML 

This is a bioinformatics tool that classifies short DNA sequences (e.g. 300bp) as coding or noncoding, based on their k-mer frequency profiles. The model is implemented from scratch in Python using only NumPy, and aims to help explore simple, interpretable ML in genomics.

> The dataset used in this project was sourced from [rokithkumar's DNA-Classification GitHub repository](https://github.com/rokithkumar/dna-classification), which provides labeled sequences for benchmarking ML classifiers.

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

Models used:
1) Multinomial Native Bayes 
- fast and interpretable 
- works well with discrete frequency counts
- assumes independence of k-mers (often too simplistic in genomics)

2) Logistic Regression
- Linear classifier that weights k-mer contributions 
- performs well when patterns are additive
- easily interpretable via feature weights

Results & Key Findings:

The dataset contained 4,380 labeled DNA sequences. After testing k-mer lengths from 3 to 6, the best performance was achieved with:

- Model: Logistic Regression  
- k-mer size: 6  
- Accuracy: **91.1%** on the held-out test set  
- Feature dimension: 4,096 (all possible 6-mers)

Example prediction:
```text
True label: 2  
Predicted label: 2  
Correct: Yes
```

Interpretation:

- Performance increased with larger `k` values, likely due to capturing codon-level or motif-level structure.
- Logistic Regression consistently outperformed Naive Bayes, suggesting linear weighting of k-mer patterns is more effective than naive probabilistic independence.
- Using raw counts of k-mers without any alignment or biological preprocessing still achieved strong performance — demonstrating the value of alignment-free ML.

---

Future Directions:

- Add Random Forest and XGBoost for nonlinear comparison  
- Visualize top predictive k-mers per class  
- Build a Streamlit demo for real-time predictions
