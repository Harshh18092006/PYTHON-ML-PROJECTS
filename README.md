# 📧 Email Classification Using Naive Bayes (Python)

A machine learning project to classify email messages into categories like Spam, Personal, Academic, etc., using the Naive Bayes algorithm. Implemented in Python using Visual Studio Code (VS Code).

---

## 🧠 Project Objective

Automatically classify emails into categories using Natural Language Processing (NLP) and Machine Learning. This helps in organizing emails more efficiently and detecting spam.

---

## 🛠 Technologies Used

- Python 3.x
- VS Code (IDE)
- Libraries:
  - pandas
  - matplotlib
  - seaborn
  - scikit-learn

---

## 📂 Files

- email_classifier.py: Main Python script containing the full workflow
- balanced_email_dataset.csv: Dataset used for training and testing

---

## 📊 Features

- Load and clean email dataset
- Visualize distribution of email categories
- Convert text to numeric features using CountVectorizer
- Train a Naive Bayes classifier (MultinomialNB)
- Evaluate model accuracy and generate a confusion matrix
- Predict category of new sample messages

---

## 📁 Dataset Info

Filename: balanced_email_dataset.csv  
Columns:
- Email_Tex: The content of the email
- Label: Category of the email (e.g., Spam, Personal, Clubs, Academic)

### ✅ Sample Rows

csv
Email_Tex,Label
"Final reminder: Your account will be suspended.",Spam
"Join us for the annual college fest!",Clubs
"Don't forget your assignment deadline is tomorrow.",Academic
"Hey! Want to catch up this weekend?",Personal

### 📦 Requirements
pip install pandas matplotlib seaborn scikit-learn
