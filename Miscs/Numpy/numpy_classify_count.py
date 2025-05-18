import numpy as np

#result = np.array([ True if model == target else False for model, target in zip(predict,y_combined) ])
#success_count = np.where(result == True, 1, 0).sum()
#total_count = result.shape[0]

labels = np.array([0, 1, 2, 1, 0, 2, 2, 1])
unique, counts = np.unique(labels, return_counts=True)

# Combine into dict (optional)
class_counts = dict(zip(unique, counts))
print(class_counts)

