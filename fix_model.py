import joblib

model = joblib.load("model.pkl")

joblib.dump(model, "model.pkl")

print("model.pkl updated successfully")