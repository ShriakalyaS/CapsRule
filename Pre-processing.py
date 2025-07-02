'''
Pre-processing
'''
train_data, test_data, train_labels, test_labels = train_test_split(
    X_selected, onehot_labels, test_size=0.3, stratify=None)
train_data, val_data, train_labels, val_labels = train_test_split(
    train_data, train_labels, test_size=0.1, stratify=None)

# convert the one-hot labels back to float
tclass_labels = np.argmax(train_labels, axis=1)
vclass_labels = np.argmax(val_labels, axis=1)
teclass_labels = np.argmax(test_labels, axis=1)

tmp_train = np.hstack((train_data, tclass_labels.reshape((len(train_labels), 1))))
df_train = pd.DataFrame(tmp_train)

tmp_val = np.hstack((val_data, vclass_labels.reshape((len(val_labels), 1))))
df_val = pd.DataFrame(tmp_val)

tmp_test = np.hstack((test_data, teclass_labels.reshape((len(test_labels), 1))))
df_test = pd.DataFrame(tmp_test)

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_data_resampled, y_data_resampled = smote.fit_resample(X_data, y_data)
data_combined_resampled = pd.DataFrame(X_data_resampled, columns=X_data.columns)
data_combined_resampled['Label'] = y_data_resampled
data = data_combined_resampled.to_numpy()
labels = np.array(data[:, -1])
