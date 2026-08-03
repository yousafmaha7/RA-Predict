import joblib

model = joblib.load("model/RA_Predict.pkl")

print(type(model))

try:
    print(model.feature_names_in_)
except AttributeError:
    print("No feature names stored.")

print("Model loaded successfully.")
