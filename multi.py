


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

# # File paths for positive and negative reviews
# pos_file = 'C:\Users\manis\OneDrive\Desktop\s\CS683-main\Assignment\rt-polaritydata\rt-polaritydata\rt-polarity.neg'
# neg_file = 'C:\Users\manis\OneDrive\Desktop\s\CS683-main\Assignment\rt-polaritydata\rt-polaritydata\rt-polarity.pos'

pos_file = r'rt-polaritydata\rt-polaritydata\rt-polarity.pos'
neg_file = r'rt-polaritydata\rt-polaritydata\rt-polarity.neg'


# Reading reviews from files
with open(neg_file, 'r', encoding='latin-1') as neg_f:
    neg_reviews = neg_f.readlines()

with open(pos_file, 'r', encoding='latin-1') as pos_f:
    pos_reviews = pos_f.readlines()

# Creating dataframes for positive and negative reviews
neg_df = pd.DataFrame(neg_reviews, columns=["Review"])
neg_df['class'] = 0
pos_df = pd.DataFrame(pos_reviews, columns=["Review"])
pos_df['class'] = 1

# Splitting data into training, validation, and test sets
train_data = pd.concat([neg_df.iloc[:4000], pos_df.iloc[:4000]])
val_data = pd.concat([neg_df.iloc[4000:4500], pos_df.iloc[4000:4500]])
test_data = pd.concat([neg_df.iloc[4500:], pos_df.iloc[4500:]])

X_train, y_train = train_data['Review'], train_data['class']
X_val, y_val = val_data['Review'], val_data['class']
X_test, y_test = test_data['Review'], test_data['class']

# Applying TF-IDF vectorization
tfidf = TfidfVectorizer(max_features=5000)
X_train = tfidf.fit_transform(X_train)
X_val = tfidf.transform(X_val)
X_test = tfidf.transform(X_test)

# Training a RandomForest classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

# Predicting on validation set
val_pred = rf_clf.predict(X_val)
val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(y_val, val_pred, average='binary')

# Converting metrics to percentage
val_precision = val_precision * 100
val_recall = val_recall * 100
val_f1 = val_f1 * 100

# Predicting on test set
y_pred = rf_clf.predict(X_test)

# Calculating confusion matrix and metrics
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
precision = precision * 100
recall = recall * 100
f1 = f1 * 100

# Printing final performance metrics
print("\nFinal Performance Metrics:")
print(f"True Positives (TP): {tp}")
print(f"True Negatives (TN): {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"Precision: {precision:.3f}%")
print(f"Recall: {recall:.3f}%")
print(f"F1-score: {f1:.3f}%")

# Plotting the confusion matrix heatmap
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix Heatmap')
plt.show()
plt.savefig('confusion_matrix_heatmap.png', bbox_inches='tight', dpi=300)  
