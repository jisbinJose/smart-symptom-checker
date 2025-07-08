# Create dataset_loader.py
from datasets import load_dataset
import streamlit as st

@st.cache_data
def load_eye_disease_dataset():
    """Load the falah/eye-disease-dataset"""
    try:
        dataset = load_dataset("falah/eye-disease-dataset")
        return dataset
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return None

def explore_dataset(dataset):
    """Explore dataset structure and samples"""
    if dataset:
        st.write(f"Dataset keys: {dataset.keys()}")
        st.write(f"Training samples: {len(dataset['train'])}")
        
        # Show sample
        sample = dataset['train'][0]
        st.write(f"Sample keys: {sample.keys()}")
        return sample