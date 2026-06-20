import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# 1. Load the correct dataset 
df = pd.read_csv('models/crop_recommendation_dataset.csv') 

# 2. Extract the exact features your app.py expects
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label'] # This should be the column containing the crop names

# 3. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the Random Forest
print("Training Random Forest...")
model = RandomForestClassifier(n_estimators=20, random_state=42)
model.fit(X_train, y_train)

with open('RandomForest.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Success! RandomForest.pkl has been generated.")