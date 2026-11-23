# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and load the Iris dataset.
2. Separate the input features (X) and output labels (Y).
3. Train the SGD Classifier model using the fit() function.
4. 1. Predict the Iris flower species using the predict() function and display the result.

## Program:
```

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

iris = load_iris()
X = iris.data
Y = iris.target

model = make_pipeline(StandardScaler(), SGDClassifier(random_state=42, max_iter=1000))
model.fit(X, Y)

sample = np.array([[5.1, 3.5, 1.4, 0.2]])
prediction = model.predict(sample)

print("Predicted Iris Species:", iris.target_names[prediction[0]])

```

## Output:
<img width="275" height="47" alt="image" src="https://github.com/user-attachments/assets/d8938acb-2bb9-4d3a-8902-a7c01205d120" />


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
