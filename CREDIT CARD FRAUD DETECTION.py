import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

# Load the dataset
credit_card_file_path = "C:/Users/hites/Downloads/credit card fraud data set/creditcard.csv"
credit_card_df = pd.read_csv(credit_card_file_path)

# Prepare the data
X = credit_card_df.drop(columns=['Class'])
y = credit_card_df['Class']

# spliting the data for traing and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# i used random forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Makeing the  predictions
y_pred = model.predict(X_test)

# model evaluation
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraudulent'])
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)
print('Confusion Matrix:')
print(conf_matrix)

# Save the model
joblib.dump(model, 'credit_card_fraud_model.pkl')

# Save the features used by the model
joblib.dump(X.columns, 'model_features.pkl')
