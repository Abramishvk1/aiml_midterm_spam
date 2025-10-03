# AIML Midterm Exam – Spam Email Detection

## Files uploaded
- `kh_abramishvili2024_847293.csv` - dataset (uploaded).

## Data description
Columns in the dataset (2500 rows):
['words', 'links', 'capital_words', 'spam_word_count', 'is_spam']

## Steps performed
1. Loaded the dataset and used the features `words`, `links`, `capital_words`, `spam_word_count`.
2. Split the data into **70% training** and **30% testing** using stratified split.
3. Trained a **Logistic Regression** classifier (scikit-learn `LogisticRegression`) on the training set.
4. Evaluated the model on the test set and produced confusion matrix and accuracy.
5. Implemented a `spam_classifier.py` script that extracts features from raw email text and classifies it using the trained model.

## Model details
- Training samples: 1750
- Test samples: 750
- Coefficients found by the model:
{'words': 0.007089664498887002, 'links': 0.8237172464347292, 'capital_words': 0.4431307492640287, 'spam_word_count': 0.7513066585155839}
- Intercept: -9.196985

## Performance on test set
- Accuracy: 0.9693

Confusion matrix (rows = true, columns = predicted):
[[366, 11], [12, 361]]

## How to reproduce
1. Clone repository and ensure Python packages installed:
```
pip install pandas scikit-learn matplotlib seaborn
```
2. Run training and evaluation (already done):
```
python spam_classifier.py "Your email text here"
```
or provide a path to a text file containing the email.

## Sample emails and classification
- **Example spam email (constructed):**
"""CONGRATULATIONS! You are a WINNER. Click here http://free-prize.example.com to claim your FREE PRIZE now. LIMITED offer!"""
This contains many spam words, a link, and uppercase words so model predicts SPAM.

- **Example legitimate email (constructed):**
"""Hi John, I hope you are well. Let's schedule a meeting next week to discuss the project timeline. Best regards, Khvicha"""
Contains normal sentence structure and no spam trigger words or links.

## Visualizations
- `hist_words_by_class.png` - distribution of 'words' feature by class.
- `scatter_links_spamwords.png` - scatter plot of links vs spam_word_count colored by class.
- `confusion_matrix.png` - confusion matrix image.

