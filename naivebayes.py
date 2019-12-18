gnb = GaussianNB()
bnb = BernoulliNB()
mnb = MultinomialNB()


# Gaussian: likelihood of the features at assumed to be gaussian, or bell curve distribution
y_pred1 = gnb.fit(trainX, trainY).predict(testX)
print("gaussian -> number of mislabeled points out of a total %d points: %d" % (testX.shape[0],(testY != y_pred1).sum()))

# Bernoulli: implemenets naive bayes for data that is distributed with multivariate bernoulli distribution (?)
# in laymens term, although there may be multiple features, each one is assumed to be a binary valued variable
y_pred2 = bnb.fit(trainX, trainY).predict(testX)
print("bernoulli -> number of mislabeled points out of a total %d points: %d" % (testX.shape[0],(testY != y_pred2).sum()))

# Multinomial: for multinomial distributed data, is is used in text classification where data is d
# represented in word vector counts
# smoothens by maximum likelihood (relative frequency counting)
y_pred3 = mnb.fit(trainX, trainY).predict(testX)
print("multinomial -> number of mislabeled points out of a total %d points: %d" % (testX.shape[0],(testY != y_pred3).sum()))

y_pred4 = gnb.fit(trainX, trainY).predict(trainX)
print("train error -> number of mislabeled points out of a total %d points: %d" % (trainX.shape[0],(trainY != y_pred4).sum()))


## in this case, bernoulli performs the best with an error rate of around 27%
