# Plant Disease Classification

This project implements a machine learning model for classifying plant diseases using image analysis. The model is trained on the PlantVillage dataset and can identify various plant diseases across different plant species.

## Features

- Image-based plant disease classification
- Support for multiple plant species and diseases
- HOG (Histogram of Oriented Gradients) feature extraction
- Machine learning pipeline using scikit-learn
- Interactive Jupyter notebook for analysis and visualization

## Requirements

The project requires the following Python packages:
```
numpy>=1.19.2
opencv-python>=4.5.1
scikit-learn>=0.24.1
pillow>=8.2.0
pandas>=1.2.4
joblib>=1.0.1
matplotlib>=3.4.2
seaborn>=0.11.1
```

## Project Structure

- `plant_disease_classification.ipynb`: Main Jupyter notebook containing the model training and analysis
- `predict_disease.py`: Script for making predictions on new images
- `train_model.py`: Script for training the model
- `requirements.txt`: List of Python dependencies
- `plant_disease_model.joblib`: Trained model file

## Setup and Installation

1. Clone this repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the PlantVillage dataset and place it in the project directory
4. Run the Jupyter notebook or use the prediction script

## Usage

### Using the Jupyter Notebook

Open and run `plant_disease_classification.ipynb` to:
- Load and preprocess the dataset
- Extract HOG features from images
- Train the classification model
- Evaluate model performance
- Visualize results

### Making Predictions

Use `predict_disease.py` to classify new plant images:
```python
python predict_disease.py path/to/image.jpg
```

## Model Details

The classification pipeline includes:
- Image preprocessing and resizing
- HOG feature extraction
- PCA dimensionality reduction
- Logistic Regression classifier

## Results

The model achieves good accuracy in classifying plant diseases, with detailed performance metrics available in the notebook.

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
