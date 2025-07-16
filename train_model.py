import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib

# Load the updated CSV
df = pd.read_csv("opportunities.csv", sep=';')

# === 1. Create Target Column ===
df['IsWon'] = df['Stage'].apply(lambda x: 1 if x == 'Closed Won' else 0)

# === 2. Define categorical features (keep raw strings) ===
categorical_cols = [
    "Account Name: Account Name",
    "Opportunity Owner: Full Name",
    "DRX Carline: DRX Carline Name",
    "Carline Segment",
    "E-Mobility Category",
    "Project Category"
]

# === 3. Define features and target ===
X = df[categorical_cols]
y = df["IsWon"]

# === 4. Split data ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 5. Build preprocessing pipeline ===
# OneHotEncoder handles unknown categories during prediction with handle_unknown='ignore'
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

# === 6. Create full pipeline ===
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42))
])

# === 7. Train pipeline ===
pipeline.fit(X_train, y_train)

# === 8. Evaluate ===
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))
print("Prediction values:", y_pred)
print("Target distribution:", df['IsWon'].value_counts(normalize=True))

# === 9. Save pipeline ===
joblib.dump(pipeline, "random_forest_pipeline.pkl")
