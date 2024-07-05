import joblib
import pandas as pd

# Load the model
model = joblib.load('credit_card_fraud_model.pkl')

# load the features
features = joblib.load('model_features.pkl')

# i also tried takeing different dat value but because of index missing model doesn't work properly that's why instead of takeing any new value i put the value from the dat set
new_data = pd.DataFrame({
    'Time': [100000, 150000],
    'V1': [-1.359807, 1.191857],
    'V2': [-0.072781, 0.266151],
    'V3': [1.378155, 0.448154],
    'V4': [2.536347, 0.166480],
    'V5': [0.462388, 0.060018],
    'V6': [-0.338321, -0.082361],
    'V7': [0.239599, -0.078803],
    'V8': [0.098698, 0.085102],
    'V9': [0.363787, -0.255425],
    'V10': [0.090794, -0.166974],
    'V11': [-0.551600, 1.612726],
    'V12': [-0.617801, 1.065235],
    'V13': [-0.991390, 0.489095],
    'V14': [-0.311169, -0.143772],
    'V15': [1.468177, 0.635558],
    'V16': [-0.470400, 0.463917],
    'V17': [0.207971, -0.114805],
    'V18': [0.025791, -0.183361],
    'V19': [0.403993, -0.145783],
    'V20': [0.251412, -0.069083],
    'V21': [-0.018307, -0.225775],
    'V22': [0.277838, -0.638672],
    'V23': [-0.110474, 0.101288],
    'V24': [0.066928, -0.339846],
    'V25': [0.128539, 0.167170],
    'V26': [-0.189115, 0.125895],
    'V27': [0.133558, -0.008983],
    'V28': [-0.021053, 0.014724],
    'Amount': [149.62, 2.69]
})
new_data = new_data[features]

# Make predictions
predictions = model.predict(new_data)

print(f'Predictions: {predictions}')  # if output value is  :fraudulent (1) or legitimate (0)
