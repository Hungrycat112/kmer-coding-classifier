import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from itertools import product

class KmerClassifier:
    def __init__(self, k=4):
        """
        Initialize k-mer classifier
        
        Args:
            k (int): Length of k-mers to extract
        """
        self.k = k
        self.nb_model = MultinomialNB()
        self.lr_model = LogisticRegression(max_iter=1000)
        self.kmer_vocab = None
        
    def generate_all_kmers(self, k):
        """Generate all possible k-mers of length k"""
        bases = ['A', 'T', 'G', 'C']
        return [''.join(kmer) for kmer in product(bases, repeat=k)]
    
    def extract_kmers(self, sequence):
        """Extract k-mers from a DNA sequence"""
        kmers = []
        for i in range(len(sequence) - self.k + 1):
            kmers.append(sequence[i:i+self.k])
        return kmers
    
    def sequence_to_kmer_vector(self, sequence):
        """Convert DNA sequence to k-mer frequency vector"""
        kmers = self.extract_kmers(sequence)
        kmer_counts = Counter(kmers)
        
        # Create vector based on all possible k-mers
        vector = []
        for kmer in self.kmer_vocab:
            vector.append(kmer_counts.get(kmer, 0))
        
        return np.array(vector)
    
    def prepare_features(self, sequences):
        """Convert list of sequences to feature matrix"""
        # Generate vocabulary of all possible k-mers
        if self.kmer_vocab is None:
            self.kmer_vocab = self.generate_all_kmers(self.k)
        
        # Convert each sequence to feature vector
        feature_matrix = []
        for seq in sequences:
            vector = self.sequence_to_kmer_vector(seq)
            feature_matrix.append(vector)
        
        return np.array(feature_matrix)
    
    def fit(self, sequences, labels):
        """Train both Naive Bayes and Logistic Regression models"""
        print(f"Extracting {self.k}-mers and preparing features...")
        X = self.prepare_features(sequences)
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Number of unique {self.k}-mers: {len(self.kmer_vocab)}")
        
        # Train both models
        print("Training Naive Bayes model...")
        self.nb_model.fit(X, labels)
        
        print("Training Logistic Regression model...")
        self.lr_model.fit(X, labels)
        
        return self
    
    def predict(self, sequences, model='nb'):
        """Make predictions using specified model"""
        X = self.prepare_features(sequences)
        
        if model == 'nb':
            return self.nb_model.predict(X)
        elif model == 'lr':
            return self.lr_model.predict(X)
        else:
            raise ValueError("Model must be 'nb' or 'lr'")
    
    def evaluate(self, sequences, labels, model='nb'):
        """Evaluate model performance"""
        predictions = self.predict(sequences, model)
        accuracy = accuracy_score(labels, predictions)
        
        print(f"\n=== {model.upper()} Model Results ===")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(labels, predictions))
        
        return accuracy, predictions

def main():
    # Load the dataset
    print("Loading DNA dataset...")
    df = pd.read_csv('DNA.csv')  
    # Filter out sequences shorter than the largest k 
    max_k = 6
    df = df[df['sequence'].apply(len) >= max_k].reset_index(drop=True)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Class distribution:\n{df['class'].value_counts()}")
    print(f"\nFirst few sequences:")
    print(df.head())
    
    # Prepare data
    sequences = df['sequence'].values
    labels = df['class'].values
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        sequences, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Try different k values
    k_values = [3, 4, 5, 6]
    results = {}
    
    for k in k_values:
        print(f"\n{'='*50}")
        print(f"Testing with k={k}")
        print(f"{'='*50}")
        
        # Create and train classifier
        classifier = KmerClassifier(k=k)
        classifier.fit(X_train, y_train)
        
        # Evaluate both models
        nb_acc, _ = classifier.evaluate(X_test, y_test, model='nb')
        lr_acc, _ = classifier.evaluate(X_test, y_test, model='lr')
        
        results[k] = {'nb': nb_acc, 'lr': lr_acc}
    
    # Summary of results
    print(f"\n{'='*50}")
    print("SUMMARY OF RESULTS")
    print(f"{'='*50}")
    print("k-value\tNaive Bayes\tLogistic Regression")
    for k, acc in results.items():
        print(f"{k}\t{acc['nb']:.4f}\t\t{acc['lr']:.4f}")
    
    # Find best k and model
    best_k = None
    best_model = None
    best_acc = 0
    
    for k, acc in results.items():
        if acc['nb'] > best_acc:
            best_acc = acc['nb']
            best_k = k
            best_model = 'nb'
        if acc['lr'] > best_acc:
            best_acc = acc['lr']
            best_k = k
            best_model = 'lr'
    
    print(f"\nBest performance: k={best_k}, {best_model.upper()} model, accuracy={best_acc:.4f}")
    
    # Example prediction with best model
    print(f"\n{'='*50}")
    print("EXAMPLE PREDICTION")
    print(f"{'='*50}")
    
    best_classifier = KmerClassifier(k=best_k)
    best_classifier.fit(X_train, y_train)
    
    # Take first sequence from test set as example
    example_seq = X_test[0]
    true_label = y_test[0]
    
    if best_model == 'nb':
        pred_label = best_classifier.predict([example_seq], model='nb')[0]
    else:
        pred_label = best_classifier.predict([example_seq], model='lr')[0]
    
    print(f"Example sequence: {example_seq[:50]}...")
    print(f"True label: {true_label}")
    print(f"Predicted label: {pred_label}")
    print(f"Correct: {'Yes' if pred_label == true_label else 'No'}")

if __name__ == "__main__":
    main()