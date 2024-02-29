"""
Write your unit tests here. Some tests to include are listed below.
This is not an exhaustive list.

- check that prediction is working correctly
- check that your loss function is being calculated correctly
- check that your gradient is being calculated correctly
- check that your weights update during training
"""

# Imports
import pytest
# (you will probably need to import more things here)
from regression import (logreg, utils)
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import log_loss
#from sklearn.linear_model import LogisticRegression

def test_prediction():
	log_model = logreg.LogisticRegressor(num_feats=3, learning_rate=0.15, tol=0.01, max_iter=50, batch_size=10)
	log_model.W = [0,0,0,0]
	test_X = [1,1,1,1]
	prediction = log_model.make_prediction(test_X)
	test_truth = .5 # sigmoid of 0

	assert prediction == test_truth

def test_loss_function():
	log_model = logreg.LogisticRegressor(num_feats=3, learning_rate=0.15, tol=0.01, max_iter=50, batch_size=10)

	test_pred = [.1, .1, .9]
	test_truth = [0, 0, 1]
	loss = log_model.loss_function(test_truth, test_pred)
	sklearn_loss = log_loss(test_truth, test_pred, normalize = True)
	assert np.isclose(loss, sklearn_loss)

def test_gradient():
	log_model = logreg.LogisticRegressor(num_feats=3, learning_rate=0.15, tol=0.01, max_iter=50, batch_size=10)
	log_model.W = [0,0,0,0]
	test_X = [1,1,1,1]
	test_truth = 1
	prediction = log_model.make_prediction(test_X)
	grad = np.dot((test_truth - prediction), test_X)/1
	print(grad)
	assert np.all(grad == [.5,.5,.5,.5])
	
	
def test_training():
	# load data
	X_train, X_val, y_train, y_val = utils.loadDataset(
        features=[
            'Penicillin V Potassium 500 MG',
            'Computed tomography of chest and abdomen',
            'Plain chest X-ray (procedure)',
            'Low Density Lipoprotein Cholesterol',
            'Creatinine',
            'AGE_DIAGNOSIS'
        ],
        split_percent=0.8,
        split_seed=42
    )
	# Scale the data, since values vary across feature. Note that we
    # fit on the training data and use the same scaler for X_val.
	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_val = sc.transform(X_val)

    # save W matrix before and after training
	log_model = logreg.LogisticRegressor(num_feats=6, learning_rate=0.15, tol=0.01, max_iter=50, batch_size=10)
	W_before = log_model.W.copy()
	log_model.train_model(X_train, y_train, X_val, y_val)
	W_after = log_model.W.copy()
	
	# check that before and after weights are different
	assert not np.allclose(W_before, W_after)