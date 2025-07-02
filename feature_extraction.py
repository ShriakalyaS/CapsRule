from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler

# Define the number of top features to select (you can change this value)
num_features_to_select = 30  

# Compute Mutual Information scores
X_data = data_combined.drop(columns=['Label'])
  # Drop the label column for feature selection
y_data = data_combined['Label']  # Extract the labels


X_data = X_data.drop(columns=['SimillarHTTP'])
print(X_data.dtypes)
import numpy as np

print("Infinite values in X_data:", np.isinf(X_data).sum().sum())
# Replace infinite values with NaN
X_data.replace([np.inf, -np.inf], np.nan, inplace=True)

# Fill NaN values with the column's max (excluding NaN itself)
X_data.fillna(X_data.max(), inplace=True)

# Verify again
print("Infinite values after replacement:", np.isinf(X_data).sum().sum())

# Convert categorical labels to numerical if needed
y_data, _ = pd.factorize(y_data)

# Normalize data before MI computation (optional, but can improve MI estimation)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_data)

# Compute MI scores
mi_scores = mutual_info_classif(X_scaled, y_data)

# Create a DataFrame to store MI scores with feature names
mi_df = pd.DataFrame({'Feature': X_data.columns, 'MI_Score': mi_scores})

# Sort features based on MI scores
mi_df = mi_df.sort_values(by='MI_Score', ascending=False)

# Select top features
selected_features = mi_df.iloc[:num_features_to_select]['Feature'].tolist()

print("Selected Features:")
print(selected_features)
X_selected = X_data[selected_features]
