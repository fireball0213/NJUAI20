import numpy as np

X = np.array([[1, 2, 3], [4, 0, 2], [1, 2, 9], [3, 5, 2]])

X_mean_centered = X - np.mean(X, axis=0)
X_cov = np.dot(X_mean_centered.T, X_mean_centered) / X_mean_centered.shape[0]
eig_values_x, eig_vectors_x = np.linalg.eig(X_cov)
idx = np.argsort(-eig_values_x)
largest_eig_value = eig_values_x[idx[0]]
largest_eig_vector = eig_vectors_x[:, idx[0]]
second_largest_eig_value = eig_values_x[idx[1]]
second_largest_eig_vector = eig_vectors_x[:, idx[1]]

print("second largest eigen value of X:", second_largest_eig_value)
print("corresponding eigen vector", second_largest_eig_vector)
largest_eig_vector = largest_eig_vector.reshape(1, 3)
y = X - (largest_eig_vector @ X_mean_centered.T).T @ largest_eig_vector
y = y - y.mean(axis=0)
y_cov = np.dot(y.T, y) / y.shape[0]
eig_values_y, eig_vectors_y = np.linalg.eig(y_cov)
largest_eig_value_y = eig_values_y[0]
largest_eig_vector_y = -eig_vectors_y[:, 0]

print("largest eigen value of Y:", largest_eig_value_y)
print("corresponding eigen vector", largest_eig_vector_y)
