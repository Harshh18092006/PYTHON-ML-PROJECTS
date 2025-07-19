# Step 1: Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Step 2: Load the dataset
df = pd.read_csv('balanced_email_dataset.csv')

# ✅ Check and fix the column name
print("Available columns:", df.columns)

# ✅ Use correct column name
# Assume it is 'Email Text' (not 'Email_Text')
df.dropna(subset=['Email_Tex'], inplace=True)

# Step 3: Bar chart of label distribution
plt.figure(figsize=(8,5))
sns.countplot(data=df, x='Label', palette='Set2')
plt.title("Distribution of Email Categories")
plt.xlabel("Label")
plt.ylabel("Count")
plt.grid(axis='y', linestyle='dotted')
plt.show()

# Step 4: Prepare features and labels
X = df['Email_Tex']
y = df['Label']

# Step 5: Vectorize the email text
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Step 6: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Step 7: Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 8: Predict
y_pred = model.predict(X_test)

# Step 9: Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

# Step 10: Confusion Matrix
labels = sorted(list(set(y_test) | set(y_pred)))  # Ensure all labels are included
conf_mat = confusion_matrix(y_test, y_pred, labels=labels)

plt.figure(figsize=(8,6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title("Confusion Matrix")
plt.show()

# Step 11: Predict new email
sample_email = ["lets meet today"]
sample_vectorized = vectorizer.transform(sample_email)
print("\nPrediction for sample email:", model.predict(sample_vectorized)[0])