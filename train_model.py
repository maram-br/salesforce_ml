import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Load the updated CSV
df = pd.read_csv("opportunities.csv", sep=';')

# === 1. Create Target Column ===
df['IsWon'] = df['Stage'].apply(lambda x: 1 if x == 'Closed Won' else 0)

# === 2. Encode categorical features ===
# Rename columns for easier access
df.rename(columns={
    "Account Name: Account Name": "Account Name",
    "Opportunity Owner: Full Name": "Opportunity Owner",
    "DRX Carline: DRX Carline Name": "Carline Name"
}, inplace=True)

# Convert all categorical features into numerical codes
categorical_cols = [
    "Account Name",
    "Opportunity Owner",
    "Carline Name",
    "Carline Segment",
    "E-Mobility Category",
    "Project Category"
]

for col in categorical_cols:
    df[col] = df[col].astype('category').cat.codes

# === 3. Define features and target ===
X = df[categorical_cols]
y = df["IsWon"]

# === 4. Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 5. Train Model ===
model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# === 6. Evaluate ===
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("Prediction values:", y_pred)
print("Target distribution:", df['IsWon'].value_counts(normalize=True))

# === 7. Save model ===
joblib.dump(model, "random_forest_model.pkl")
