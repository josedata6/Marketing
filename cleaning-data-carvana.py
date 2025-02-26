# cleaning data carvana

# ###### hadling missing values in variablewheeltypeID
# import pandas as pd

# # Load the dataset
# df = pd.read_csv("training.csv")  # Replace with actual file path

# # Check missing values in WheelTypeID
# print("Missing values in WheelTypeID:", df['WheelTypeID'].isnull().sum())

# # Strategy 1: Fill missing values with mode if <5% missing
# if df['WheelTypeID'].isnull().mean() < 0.05:
#     most_frequent = df['WheelTypeID'].mode()[0]
#     df['WheelTypeID'].fillna(most_frequent, inplace=True)
# else:
#     # Strategy 2: Assign a new category (-1) if >5% missing
#     df['WheelTypeID'].fillna(-1, inplace=True)

# # Optional: Create a missing indicator
# df['WheelTypeID_missing'] = df['WheelTypeID'].apply(lambda x: 1 if x == -1 else 0)

# # Verify missing values are handled
# print("Remaining missing values:", df['WheelTypeID'].isnull().sum())

# # Save cleaned dataset
# df.to_csv("cleaned_training.csv", index=False)



############################
########## Logistic Regression Model

# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report, accuracy_score

# # Load data
# df = pd.read_csv("training-3.csv")

# # Selecting relevant variables
# features = [
#     "VehYear", "VehicleAge", "VehOdo", "MMRAcquisitionAuctionAveragePrice",
#     "MMRAcquisitionAuctionCleanPrice", "MMRAcquisitionRetailAveragePrice",
#     "MMRCurrentAuctionAveragePrice", "MMRCurrentRetailAveragePrice",
#     "WarrantyCost", "IsOnlineSale"
# ]

# # Handling categorical variables (encoding)
# df = pd.get_dummies(df, columns=["Transmission", "WheelType", "Nationality", "Size", "TopThreeAmericanName"], drop_first=True)

# # Define X and y
# X = df[features]
# y = df["IsBadBuy"]

# # Splitting data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Standardizing numeric features
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# print(X_train.isnull().sum())
# print(y_train.isnull().sum())

# # Logistic Regression Model
# model = LogisticRegression(max_iter=1000)
# model.fit(X_train, y_train)

# # Predictions
# y_pred = model.predict(X_test)

# # Model Evaluation
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("Classification Report:\n", classification_report(y_test, y_pred))

# # Coefficients & p-values
# coefficients = pd.DataFrame({"Feature": X.columns, "Coefficient": model.coef_[0]})
# print(coefficients)


######################################

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load data
df = pd.read_csv("training-3.csv")

# Selecting relevant variables
features = [
    "VehYear", "VehicleAge", "VehOdo", "MMRAcquisitionAuctionAveragePrice",
    "MMRAcquisitionAuctionCleanPrice", "MMRAcquisitionRetailAveragePrice",
    "MMRCurrentAuctionAveragePrice", "MMRCurrentRetailAveragePrice",
    "WarrantyCost", "IsOnlineSale"
]

# Handling categorical variables (encoding)
df = pd.get_dummies(df, columns=["Transmission", "WheelType", "Nationality", "Size", "TopThreeAmericanName"], drop_first=True)

# Define X and y
X = df[features]
y = df["IsBadBuy"]

# Check for missing values
X.fillna(X.median(), inplace=True)
y.fillna(0, inplace=True)

# Convert y to integer type
y = y.astype(int)

# Splitting data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing numeric features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape y_train if needed
y_train = y_train.ravel()

# Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

######### over sample
from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

model.fit(X_train_balanced, y_train_balanced)

#### penalize misclassification of minority class (Bad Buy) by setting class weights:
model = LogisticRegression(class_weight="balanced", max_iter=1000)
model.fit(X_train, y_train)

####Random Forest
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)