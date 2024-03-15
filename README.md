# Email Classifier

This project demonstrates a simple email classifier using the Naive Bayes algorithm.

## Overview

The classifier is trained on a sample dataset of emails categorized into different classes such as "Personal", "Social", and "Others". It preprocesses the text data by removing stopwords and tokenizing the text. Then, it trains a Multinomial Naive Bayes classifier using TF-IDF features.

## Usage

1. **Training the Classifier**: The classifier is trained on the provided sample data. It preprocesses the text, splits the data into training and testing sets, and then trains the classifier.

2. **Evaluation**: The classifier's accuracy is evaluated on the testing set to measure its performance.

3. **Prediction**: You can test the classifier with new emails. It predicts the category of each email based on its content.

## Dependencies

- nltk
- scikit-learn

You can install the dependencies using pip:


## Usage

To run the code, execute the provided Python script `email_classifier.py`.

```bash
python email_classifier.py

## Sample Output.


Accuracy: 0.75

Email: Hey, how have you been? - Predicted category: Personal
Email: Join me for a party tonight! - Predicted category: Social
Email: Limited time offer: 50% discount! - Predicted category: Others
Email: Reminder: Doctor's appointment tomorrow. - Predicted category: Others
Email: Congratulations on your promotion! - Predicted category: Personal

## License
This project is licensed under the MIT License - see the LICENSE file for details.
