import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load the CSV
df = pd.read_csv("opportunities.csv", sep=';')

# === 1. Create Target Column ===
df['IsWon'] = df['Stage'].apply(lambda x: 1 if x == 'Closed Won' else 0)

# === 2. Parse Dates ===
df['Created Date'] = pd.to_datetime(df['Created Date'], dayfirst=True, errors='coerce')
df['Close Date'] = pd.to_datetime(df['Close Date'], dayfirst=True, errors='coerce')


# === 3. Feature Engineering ===
df['DaysToClose'] = (df['Close Date'] - df['Created Date']).dt.days

# === 4. Encode categorical features ===
df['Owner Role'] = df['Owner Role'].astype('category').cat.codes
df['Opportunity Owner'] = df['Opportunity Owner'].astype('category').cat.codes
df['Fiscal Period'] = df['Fiscal Period'].astype('category').cat.codes

# === 5. Drop rows with missing required values (if any) ===
df.dropna(subset=["Probability (%)", "Age", "DaysToClose"], inplace=True)

# Clean numeric fields that might use commas
for col in ["Probability (%)", "Age", "DaysToClose"]:
    df[col] = df[col].astype(str).str.replace(",", ".").astype(float)

# === 6. Define features and target ===
X = df[[
    "Owner Role",
    "Opportunity Owner",
    "Fiscal Period",
    "Probability (%)",
    "Age",
    "DaysToClose"
]]
y = df["IsWon"]

# === 7. Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 8. Train Model ===
model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# === 9. Evaluate ===
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(y_pred)  # prints array like [0 1 1 0 0 1 ...]
print(df['IsWon'].value_counts(normalize=True))
"""
new_data = pd.DataFrame({
    "Owner Role": [1],  # example encoded value
    "Opportunity Owner": [5],
    "Fiscal Period": [1],
    "Probability (%)": [0],
    "Age": [0],
    "DaysToClose": [1]
})

new_prediction = model.predict(new_data)
print(new_prediction)  # Outputs: [1] or [0]
probs = model.predict_proba(new_data)
print("Probabilities:", probs)

"""
joblib.dump(model, "random_forest_model.pkl")