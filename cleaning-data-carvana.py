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

# # Check for missing values
# X.fillna(X.median(), inplace=True)
# y.fillna(0, inplace=True)

# # Convert y to integer type
# y = y.astype(int)

# # Splitting data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Standardizing numeric features
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # Reshape y_train if needed
# y_train = y_train.ravel()

# # Logistic Regression Model
# model = LogisticRegression(max_iter=1000)
# model.fit(X_train, y_train)

# # Predictions
# y_pred = model.predict(X_test)

# # Model Evaluation
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("Classification Report:\n", classification_report(y_test, y_pred))

# ######### over sample
# from imblearn.over_sampling import SMOTE

# smote = SMOTE(sampling_strategy='auto', random_state=42)
# X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# model.fit(X_train_balanced, y_train_balanced)

# #### penalize misclassification of minority class (Bad Buy) by setting class weights:
# model = LogisticRegression(class_weight="balanced", max_iter=1000)
# model.fit(X_train, y_train)

# ####Random Forest
# from sklearn.ensemble import RandomForestClassifier

# rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
# rf_model.fit(X_train, y_train)

# y_pred = rf_model.predict(X_test)


###############################
########## testing less variables

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
#     "VehicleAge", "VehOdo",
#     "MMRCurrentAuctionAveragePrice", "MMRCurrentRetailAveragePrice",
#     "WarrantyCost"
# ]

# # Handling categorical variables (encoding)
# # df = pd.get_dummies(df, columns=["Transmission", "WheelType", "Nationality", "Size", "TopThreeAmericanName"], drop_first=True)

# # Define X and y
# X = df[features]
# y = df["IsBadBuy"]

# # Check for missing values
# X.fillna(X.median(), inplace=True)
# y.fillna(0, inplace=True)

# # Convert y to integer type
# y = y.astype(int)

# # Splitting data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Standardizing numeric features
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # Reshape y_train if needed
# y_train = y_train.ravel()

# # Logistic Regression Model
# model = LogisticRegression(max_iter=1000)
# model.fit(X_train, y_train)

# # Predictions
# y_pred = model.predict(X_test)

# # Model Evaluation
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("Classification Report:\n", classification_report(y_test, y_pred))

# ######### over sample
# from imblearn.over_sampling import SMOTE

# smote = SMOTE(sampling_strategy='auto', random_state=42)
# X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# model.fit(X_train_balanced, y_train_balanced)

# #### penalize misclassification of minority class (Bad Buy) by setting class weights:
# model = LogisticRegression(class_weight="balanced", max_iter=1000)
# model.fit(X_train, y_train)

# ####Random Forest
# from sklearn.ensemble import RandomForestClassifier

# rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
# rf_model.fit(X_train, y_train)

# y_pred = rf_model.predict(X_test)

################## variables selected with foward and back fill

# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.feature_selection import SequentialFeatureSelector

# # Load the CSV data
# df = pd.read_csv("training-3.csv")

# # Select features and target variable
# features = ["VehicleAge", "VehOdo", "MMRCurrentAuctionAveragePrice", "MMRCurrentRetailAveragePrice", "WarrantyCost"]
# target = "IsBadBuy"

# # Check for missing values and handle them (e.g., fill with mean)
# for feature in features:
#     if df[feature].isnull().any():
#         df[feature].fillna(df[feature].mean(), inplace=True)

# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# # Initialize the Logistic Regression model
# model = LogisticRegression(max_iter=1000)  # Increase max_iter for convergence

# # Forward Feature Selection
# forward_selector = SequentialFeatureSelector(model, n_features_to_select="auto", direction="forward", cv=5)
# forward_selector.fit(X_train, y_train)
# selected_forward = X_train.columns[forward_selector.get_support()]
# print(f"Selected Features (Forward): {list(selected_forward)}")

# # Backward Feature Selection
# backward_selector = SequentialFeatureSelector(model, n_features_to_select="auto", direction="backward", cv=5)
# backward_selector.fit(X_train, y_train)
# selected_backward = X_train.columns[backward_selector.get_support()]
# print(f"Selected Features (Backward): {list(selected_backward)}")

# # Train and Evaluate Models
# X_train_forward, X_test_forward = X_train[selected_forward], X_test[selected_forward]
# X_train_backward, X_test_backward = X_train[selected_backward], X_test[selected_backward]

# # Train forward selection model
# model.fit(X_train_forward, y_train)
# y_pred_forward = model.predict(X_test_forward)
# accuracy_forward = accuracy_score(y_test, y_pred_forward)
# print(f"Forward Selection Accuracy: {accuracy_forward:.4f}")
# print(classification_report(y_test, y_pred_forward))

# # Train backward selection model
# model.fit(X_train_backward, y_train)
# y_pred_backward = model.predict(X_test_backward)
# accuracy_backward = accuracy_score(y_test, y_pred_backward)
# print(f"Backward Selection Accuracy: {accuracy_backward:.4f}")
# print(classification_report(y_test, y_pred_backward))

####################################################################################
# ############ confusion matrix with chosen variables
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import confusion_matrix, classification_report

# # Load the dataset from CSV
# file_path = "Model3.csv"
# df = pd.read_csv(file_path)

# # Define features and target variable
# X = df.drop(columns=["IsBadBuy"])
# y = df["IsBadBuy"]

# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train a Random Forest model
# model = RandomForestClassifier(random_state=42)
# model.fit(X_train, y_train)

# # Make predictions
# y_pred = model.predict(X_test)

# # Compute confusion matrix
# cm = confusion_matrix(y_test, y_pred)

# # Plot confusion matrix
# plt.figure(figsize=(5,4))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Bad Buy", "Bad Buy"], yticklabels=["Not Bad Buy", "Bad Buy"])
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title("Confusion Matrix")
# plt.show()

# # Print classification report
# print(classification_report(y_test, y_pred))

####################################################################################
# ############ confusion matrix with all variables

# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import confusion_matrix
# from sklearn.impute import SimpleImputer

# # Load the dataset
# file_path = "training.csv"
# df = pd.read_csv(file_path)

# # Handle the PurchDate issue (drop it since it's non-numeric)
# if "PurchDate" in df.columns:
#     df = df.drop(columns=["PurchDate"])  

# # Encode categorical variables
# label_encoders = {}
# for col in df.select_dtypes(include=["object"]).columns:
#     le = LabelEncoder()
#     df[col] = le.fit_transform(df[col].astype(str))  # Convert to numerical
#     label_encoders[col] = le  

# # Define features (X) and target (y)
# X = df.drop(columns=["IsBadBuy"])  
# y = df["IsBadBuy"]

# # Handle missing values in X (features)
# imputer = SimpleImputer(strategy="median")  # Fill missing values with median
# X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)  

# # Drop rows where the target variable (y) is NaN
# X = X[y.notna()]
# y = y.dropna()

# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train Logistic Regression model
# model = LogisticRegression(max_iter=1000)
# model.fit(X_train, y_train)

# # Generate predictions
# y_pred = model.predict(X_test)

# # Compute confusion matrix
# cm = confusion_matrix(y_test, y_pred)

# # Plot the confusion matrix
# plt.figure(figsize=(6, 5))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Good Buy", "Bad Buy"], yticklabels=["Good Buy", "Bad Buy"])
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.title("Confusion Matrix")
# plt.show()


############################################################################################
##############################logistic correlation matrix with choosen variables in model3

# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Load the dataset from CSV
# file_path = "Model3.csv"
# df = pd.read_csv(file_path)

# # Drop the target variable ("IsBadBuy") since it's categorical (binary)
# X = df.drop(columns=["IsBadBuy"])

# # Compute the correlation matrix
# corr_matrix = X.corr(method="pearson")

# # Plot the heatmap
# plt.figure(figsize=(10, 8))
# sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
# plt.title("Correlation Matrix for Logistic Regression")
# plt.show()

############################################################################################
##############################logistic correlation matrix with all variables in training.csv

# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Load the dataset from CSV
# file_path = "training.csv"
# df = pd.read_csv(file_path)

# # Drop non-numeric columns
# df = df.select_dtypes(include=["number"])  # Keeps only numerical columns

# # Compute the correlation matrix
# corr_matrix = df.corr(method="pearson")

# # Plot the heatmap
# plt.figure(figsize=(10, 8))
# sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
# plt.title("Correlation Matrix for Logistic Regression")
# plt.show()

##########################
########## testing other confusion matrix

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('Model3.csv')

# Prepare the data
X = data.drop('IsBadBuy', axis=1)  # Features
y = data['IsBadBuy']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Create confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()