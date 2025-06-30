# Sentiment Analysis Tool Using Natural Language Processing (NLP)

This project focuses on classifying user reviews as either **positive** or **negative** using Natural Language Processing (NLP) technique and machine learning algorithms. The model is trained on a dataset of user reviews, with the data split into **80% for training** and **20% for testing**. Multiple classification models have been evaluated to identify the best-performing one.

## 📁 Project Structure

- `notebooks/project.ipynb`  
  The main Colab notebook containing the complete workflow — data preprocessing, model training, evaluation, and visualization.
- `data/`  
  Contains the user reviews dataset used for training and testing. Text data is preprocessed and transformed into features using the Bag of Words model.
- `results/`  
  Includes model evaluation outputs such as accuracy scores, confusion matrices, and visualization plots (e.g., word clouds, performance graphs).

## 🔍 Features

- Utilizes the **Bag of Words** technique to convert textual data into numerical features.
- Implements a **Naive Bayes Classifier** for sentiment prediction (positive or negative).
- Includes **visualizations** comparing predicted vs actual sentiment labels for performance evaluation.

## 🚀 Technologies Used

- **Python** – Core programming language used for development.
- **Scikit-learn** – Machine learning library for model training and evaluation.
- **Pandas** – Data manipulation and analysis.
- **Matplotlib** – Data visualization and plotting.
- **re (Regular Expressions)** – Text cleaning and pattern matching.
- **NLTK (Natural Language Toolkit)** – Tokenization, stopword removal, and other NLP preprocessing.
- **Google Colab** – Cloud-based Jupyter environment for running and sharing notebooks.

## 📊 Example Results

Below is a sample output showing the accuracy of the Naive Bayes Classifier in classifying sentiment:

- **Accuracy**: 73%
- **Confusion Matrix**:
  - True Positives: 55
  - True Negatives: 91
  - False Positives: 42
  - False Negatives: 12

![Sentiment Prediction Accuracy](results/accuracy_plot.png)

The model shows strong performance in distinguishing between positive and negative reviews. Additional evaluation metrics like precision, recall, and F1-score can be found in the results folder.

## 🛠️ Setup Instructions

To run this project locally:

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/sentiment-analysis-nlp.git
cd sentiment-analysis-nlp
```
2. **Open the notebook in Google Colab**
3. **Upload the dataset** - Make sure the correct path is mentioned in the code (Path in Colab)
   `dataset = pd.read_csv('File_Path.tsv', delimiter='\t', quoting=3)`
