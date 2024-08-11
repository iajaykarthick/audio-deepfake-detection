# Models Module

This module contains a collection of Python scripts dedicated to building and evaluating machine learning models that can differentiate between real and fake audio recordings. It includes models for both binary classification, which discerns between real and fake audio, and multiclass classification, which identifies various types of fake audio alongside genuine audio.

## Overview

The scripts in the `models` module are tailored for specific machine learning tasks, using both traditional methods and neural networks:

- `baseline.py`: Builds and evaluates binary classification models for determining whether an audio is real or fake.
- `baseline_multiclass.py`: Develops models for multiclass classification to distinguish among multiple types of fake audio and real audio.
- `binary_classification_nn.py`: Constructs and tests neural network models specifically for binary classification of audio data.
- `multiclass_classification_nn.py`: Implements neural network models for comprehensive multiclass classification.

## Purpose

These scripts are specifically designed for tasks in audio verification, crucial for applications like security, media integrity, and content verification. The module supports the creation, training, and evaluation of models that are vital for detecting deepfake audio.

## Usage

To use these scripts, specify the input file and the target column when running them. Ensure your data is properly formatted and all dependencies are configured:

```bash
python baseline.py <input_file> <target_column> # For binary classification
python baseline_multiclass.py <input_file> <target_column> # For multiclass classification
python binary_classification_nn.py <input_file> <target_column> # For neural network-based binary classification
python multiclass_classification_nn.py <input_file> <target_column> # For neural network-based multiclass classification
```