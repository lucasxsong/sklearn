from sklearn.linear_model import Ridge, RidgeCV

### 1. load data set...

### 2. standardize data (rescale to zero-mean and unit-variance)

### 3. choose alpha and fit model 

# in this case, alpha is the regularization strength, and reduces the variance of the estimate. 
# ridgeCV = ridge regession with [C]andidate alpha [V]alues
regr_cv = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0], normalize=True)

# decide what the best value for alpha is
model_cv = regr_cv.fit(X=trainX, y=trainYclass)
print("optimal alpha:", model_cv.alpha_)

### 4. score model on test data 

# for scoring: returns the `coefficient of determination`
# CoD: R^2, the proportion of variance in the dependent variable that is predictable from thei ndependent variable.
#  e.g. a score of .46 means that 46% of the variability of the dependent variable has been accounted for
print('ridge score:', model_cv.score(X=testX, y=testYclass))