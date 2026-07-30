import joblib
import sklearn

print("Scikit-learn Version:", sklearn.__version__)

model = joblib.load("model.pkl")

print("Model Loaded Successfully")
print(type(model))