"""
Hyperparameter optimization for digit classification
"""

import numpy as np
from sklearn import datasets, svm
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, accuracy_score
from utils import flatten_images, print_classification_metrics


def optimize_hyperparameters():
    """
    Perform hyperparameter optimization for SVM classifier.
    
    Returns:
        dict: Best parameters found
    """
    # Load digits dataset
    digits = datasets.load_digits()
    data = flatten_images(digits.images)
    
    # Define parameter grid
    param_grid = {
        'gamma': [0.001, 0.01, 0.1, 1, 'scale', 'auto'],
        'C': [0.1, 1, 10, 100, 1000]
    }
    
    # Test different development set sizes
    dev_sizes = [0.2, 0.3, 0.4, 0.5]
    
    best_overall_score = 0
    best_overall_params = {}
    
    print("=== Hyperparameter Optimization Results ===\n")
    
    for dev_size in dev_sizes:
        print(f"Testing with dev_size: {dev_size}")
        print("-" * 50)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            data, digits.target, test_size=dev_size, shuffle=True, random_state=42
        )
        
        # Create SVM classifier
        svm_classifier = svm.SVC()
        
        # Perform grid search
        grid_search = GridSearchCV(
            svm_classifier, 
            param_grid, 
            cv=5, 
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit grid search
        grid_search.fit(X_train, y_train)
        
        # Get best parameters
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        print(f"Best parameters for dev_size {dev_size}:")
        print(f"  - Gamma: {best_params['gamma']}")
        print(f"  - C: {best_params['C']}")
        print(f"  - Cross-validation score: {best_score:.4f}")
        
        # Test on held-out test set
        best_model = grid_search.best_estimator_
        test_predictions = best_model.predict(X_test)
        test_accuracy = accuracy_score(y_test, test_predictions)
        
        print(f"  - Test accuracy: {test_accuracy:.4f}")
        print()
        
        # Track overall best
        if test_accuracy > best_overall_score:
            best_overall_score = test_accuracy
            best_overall_params = {
                'dev_size': dev_size,
                'gamma': best_params['gamma'],
                'C': best_params['C'],
                'cv_score': best_score,
                'test_accuracy': test_accuracy
            }
    
    return best_overall_params


def print_optimization_summary(best_params):
    """
    Print a summary of the optimization results.
    
    Args:
        best_params (dict): Best parameters found
    """
    print("=" * 60)
    print("OPTIMAL HYPERPARAMETERS FOUND:")
    print("=" * 60)
    print(f"Dev Size (test_size): {best_params['dev_size']}")
    print(f"Gamma: {best_params['gamma']}")
    print(f"C: {best_params['C']}")
    print(f"Cross-validation Score: {best_params['cv_score']:.4f}")
    print(f"Test Accuracy: {best_params['test_accuracy']:.4f}")
    print("=" * 60)


def validate_best_model(best_params):
    """
    Train and validate the model with best parameters.
    
    Args:
        best_params (dict): Best parameters found
    """
    digits = datasets.load_digits()
    data = flatten_images(digits.images)
    
    # Split with optimal dev_size
    X_train, X_test, y_train, y_test = train_test_split(
        data, digits.target, 
        test_size=best_params['dev_size'], 
        shuffle=True, 
        random_state=42
    )
    
    # Train model with best parameters
    best_model = svm.SVC(
        gamma=best_params['gamma'], 
        C=best_params['C']
    )
    best_model.fit(X_train, y_train)
    
    # Make predictions
    predictions = best_model.predict(X_test)
    
    print("\nFINAL MODEL PERFORMANCE:")
    print("-" * 30)
    print_classification_metrics(best_model, y_test, predictions)


if __name__ == "__main__":
    # Run hyperparameter optimization
    best_params = optimize_hyperparameters()
    
    # Print summary
    print_optimization_summary(best_params)
    
    # Validate best model
    validate_best_model(best_params)
