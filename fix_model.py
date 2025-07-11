#!/usr/bin/env python3
"""
Script to fix pickle loading issues by re-saving the model
Run this once to fix the pickle module reference problem
"""

import torch
import pickle
import sys
import os

# Import ALL classes from model to fix pickle references
from model import (GPTLanguageModel, Block, Head, MultiHeadAttention, 
                   FeedForward, vocab_size, device)

# Create a custom unpickler that maps __main__ references to model module
class FixedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Redirect __main__ references to the model module
        if module == '__main__':
            module = 'model'
        return super().find_class(module, name)

def fix_model_pickle():
    """Load and re-save the model to fix pickle module references"""
    try:
        print("Fixing pickle module references...")
        print("Loading original model with fixed unpickler...")
        
        # Use custom unpickler to handle __main__ references
        with open('model-01.pkl', 'rb') as f:
            unpickler = FixedUnpickler(f)
            model = unpickler.load()
            
        print("Model loaded successfully!")
        print(f"Model type: {type(model)}")
        
        # Re-save the model with proper module references
        print("Re-saving model with correct module references...")
        with open('model-01-fixed.pkl', 'wb') as f:
            pickle.dump(model, f)
            
        print("Model saved as 'model-01-fixed.pkl'")
        
        # Test loading the fixed model
        print("Testing the fixed model...")
        with open('model-01-fixed.pkl', 'rb') as f:
            test_model = pickle.load(f)
        print("Fixed model loads correctly!")
        
        # Optionally, replace the original file
        response = input("\nReplace original model-01.pkl with fixed version? (y/n): ")
        if response.lower() == 'y':
            import shutil
            # Backup original
            shutil.copy('model-01.pkl', 'model-01-backup.pkl')
            # Replace with fixed version
            shutil.copy('model-01-fixed.pkl', 'model-01.pkl')
            print("Original model backed up as 'model-01-backup.pkl'")
            print("Fixed model is now 'model-01.pkl'")
            
            # Clean up
            os.remove('model-01-fixed.pkl')
            print("Cleaned up temporary files")
        else:
            print("To use the fixed model, update your code to load 'model-01-fixed.pkl'")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you're in the same directory as model.py and model-01.pkl")
        print("2. Check that vocab.txt exists (needed to calculate vocab_size)")
        print("3. Verify that model-01.pkl was created by your training script")
        
        # Additional debugging info
        print(f"\nCurrent directory: {os.getcwd()}")
        print(f"Files present:")
        for file in ['model.py', 'model-01.pkl', 'vocab.txt']:
            exists = "YES" if os.path.exists(file) else "NO"
            print(f"   {exists} - {file}")

if __name__ == "__main__":
    fix_model_pickle()