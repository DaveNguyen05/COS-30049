import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, silhouette_score, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the dataset
file_path = 'Property Sales of Melbourne City.csv'
data = pd.read_csv(file_path)

# Selecting key features
key_features = ['Suburb', 'Rooms', 'Type', 'Price', 'Distance', 
                'Bathroom', 'Car', 'Landsize', 'BuildingArea', 
                'YearBuilt', 'Lattitude', 'Longtitude', 'Regionname']

# Dropping unnecessary columns
cleaned_data = data[key_features]

# Handling missing values: filling missing numerical values with median
cleaned_data.loc[:, 'Landsize'] = cleaned_data['Landsize'].fillna(cleaned_data['Landsize'].median())
cleaned_data.loc[:, 'BuildingArea'] = cleaned_data['BuildingArea'].fillna(cleaned_data['BuildingArea'].median())
cleaned_data.loc[:, 'YearBuilt'] = cleaned_data['YearBuilt'].fillna(cleaned_data['YearBuilt'].median())
cleaned_data.loc[:, 'Bathroom'] = cleaned_data['Bathroom'].fillna(cleaned_data['Bathroom'].median())
cleaned_data.loc[:, 'Car'] = cleaned_data['Car'].fillna(cleaned_data['Car'].median())
cleaned_data.loc[:, 'Lattitude'] = cleaned_data['Lattitude'].fillna(cleaned_data['Lattitude'].median())
cleaned_data.loc[:, 'Longtitude'] = cleaned_data['Longtitude'].fillna(cleaned_data['Longtitude'].median())

# Handling missing categorical values by filling with 'Unknown'
cleaned_data['Suburb'].fillna('Unknown', inplace=True)
cleaned_data['Type'].fillna('Unknown', inplace=True)
cleaned_data['Regionname'].fillna('Unknown', inplace=True)

# One-Hot Encoding for categorical columns
encoder = OneHotEncoder(sparse_output=False)
encoded_features = encoder.fit_transform(cleaned_data[['Suburb', 'Type', 'Regionname']])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['Suburb', 'Type', 'Regionname']))

# Combine the encoded features with the rest of the dataset
cleaned_data = pd.concat([cleaned_data.drop(columns=['Suburb', 'Type', 'Regionname']), encoded_df], axis=1)

# Drop any remaining NaN values
cleaned_data.dropna(inplace=True)

# Normalizing numerical data
scaler = MinMaxScaler()
numerical_columns = ['Rooms', 'Price', 'Distance', 'Bathroom', 'Car', 
                     'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude']
cleaned_data[numerical_columns] = scaler.fit_transform(cleaned_data[numerical_columns])

# Log-transform the target variable (Price)
cleaned_data['Price'] = np.log1p(cleaned_data['Price'])

# Correlation matrix visualization with numerical columns only
plt.figure(figsize=(10, 8))  # Adjust size as needed
numerical_columns = ['Rooms', 'Distance', 'Bathroom', 'Car', 
                     'Landsize', 'BuildingArea', 'YearBuilt', 
                     'Lattitude', 'Longtitude', 'Price']
corr_matrix = cleaned_data[numerical_columns].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix for Numerical Features', fontsize=14)
plt.tight_layout()
plt.show()

# Boxplot for Region vs Price
plt.figure(figsize=(12, 8))
# Assuming 'Regionname' was one-hot encoded earlier, but for simplicity, use the original column if available
data['Price'] = np.log1p(data['Price'])  # Re-log the target variable
sns.boxplot(x='Regionname', y='Price', data=data)
plt.xticks(rotation=90)  # Rotate region names for better readability
plt.title('Boxplot: Region vs Log-Transformed Price', fontsize=14)
plt.xlabel('Region', fontsize=12)
plt.ylabel('Log-Transformed Price', fontsize=12)
plt.tight_layout()
plt.show()

# Splitting the dataset into features (X) and target (y)
X = cleaned_data.drop(columns=['Price'])
y = cleaned_data['Price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model 1: Random Forest Regressor
forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
forest_model.fit(X_train, y_train)

# Save the trained model to a file
joblib.dump(forest_model, "random_forest_model.pkl")

y_pred_forest = forest_model.predict(X_test)
mse_forest = mean_squared_error(y_test, y_pred_forest)
r2_forest = r2_score(y_test, y_pred_forest)
print(f'Random Forest Mean Squared Error: {mse_forest:.4f}')
print(f'Random Forest R2 Score: {r2_forest:.4f}')

# Model 2: Gradient Boosting Regressor
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)

# Save the trained model to a file
joblib.dump(gb_model, "gradient_boosting_model.pkl")

y_pred_gb = gb_model.predict(X_test)
mse_gb = mean_squared_error(y_test, y_pred_gb)
r2_gb = r2_score(y_test, y_pred_gb)
print(f'Gradient Boosting Mean Squared Error: {mse_gb:.4f}')
print(f'Gradient Boosting R2 Score: {r2_gb:.4f}')

# Model 3: K-Means Clustering
# Clustering the features (excluding Price)
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X)
clusters = kmeans.predict(X)

# Evaluate clustering with silhouette score
silhouette_avg = silhouette_score(X, clusters)
print(f'K-Means Silhouette Score: {silhouette_avg:.4f}')

# Visualizing Feature Importances for Random Forest
importances = forest_model.feature_importances_
indices = np.argsort(importances)[-10:]  # Top 10 features

plt.figure(figsize=(10, 6))
plt.barh(range(len(indices)), importances[indices], align='center', color='blue')
plt.yticks(range(len(indices)), [X.columns[i] for i in indices], fontsize=12)
plt.xlabel('Relative Importance', fontsize=12)
plt.title('Top 10 Feature Importances in Random Forest', fontsize=14)
plt.tight_layout()  # Ensures labels are not cut off
plt.show()

# Residual Plot for Random Forest
plt.figure(figsize=(10, 6))
sns.residplot(x=y_test, y=y_pred_forest, lowess=True, color='g')
plt.title('Residuals Plot for Random Forest (Checking Model Accuracy)', fontsize=14)
plt.xlabel('Actual Prices (Log Transformed)', fontsize=12)
plt.ylabel('Residuals', fontsize=12)
plt.tight_layout()
plt.show()

# Residual Plot for Gradient Boosting
plt.figure(figsize=(10, 6))
sns.residplot(x=y_test, y=y_pred_gb, lowess=True, color='b')
plt.title('Residuals Plot for Gradient Boosting (Checking Model Accuracy)', fontsize=14)
plt.xlabel('Actual Prices (Log Transformed)', fontsize=12)
plt.ylabel('Residuals', fontsize=12)
plt.tight_layout()
plt.show()

# Predicted vs Actual Prices (Random Forest)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_forest, color='red', alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='black', linestyle='--')
plt.title('Predicted vs Actual Prices (Random Forest)', fontsize=14)
plt.xlabel('Actual Prices (Log Transformed)', fontsize=12)
plt.ylabel('Predicted Prices (Log Transformed)', fontsize=12)
plt.tight_layout()
plt.show()

# Predicted vs Actual Prices (Gradient Boosting)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_gb, color='blue', alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='black', linestyle='--')
plt.title('Predicted vs Actual Prices (Gradient Boosting)', fontsize=14)
plt.xlabel('Actual Prices (Log Transformed)', fontsize=12)
plt.ylabel('Predicted Prices (Log Transformed)', fontsize=12)
plt.tight_layout()
plt.show()
