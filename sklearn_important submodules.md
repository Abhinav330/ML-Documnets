
```
Exported with AIPRM https://www.aiprm.com by Atall Robotics on 4/30/2023, 11:30:21 PM
```

---


**User:**
give the list of all the sub packages present in sklearn liberary

---

**ChatGPT:**
Sure, here's a list of all the sub-packages available in the scikit-learn (sklearn) library:
<li>sklearn.cluster: Clustering algorithms</li><li>sklearn.compose: Tools for combining and transforming data</li><li>sklearn.covariance: Robust covariance estimation</li><li>sklearn.cross_decomposition: Cross decomposition algorithms</li><li>sklearn.datasets: Toy datasets and sample generators</li><li>sklearn.decomposition: Matrix factorization and decomposition</li><li>sklearn.discriminant_analysis: Linear and quadratic discriminant analysis</li><li>sklearn.ensemble: Ensemble methods for classification and regression</li><li>sklearn.exceptions: Custom exceptions and warnings</li><li>sklearn.experimental: Experimental code</li><li>sklearn.feature_extraction: Feature extraction and selection</li><li>sklearn.gaussian_process: Gaussian process regression and classification</li><li>sklearn.inspection: Model inspection tools</li><li>sklearn.isotonic: Isotonic regression</li><li>sklearn.kernel_approximation: Kernel approximation</li><li>sklearn.kernel_ridge: Kernel ridge regression</li><li>sklearn.linear_model: Generalized linear models</li><li>sklearn.manifold: Manifold learning</li><li>sklearn.metrics: Metrics for model evaluation</li><li>sklearn.mixture: Gaussian mixture models</li><li>sklearn.model_selection: Model selection and evaluation</li><li>sklearn.multiclass: Tools for multi-class classification</li><li>sklearn.multioutput: Tools for multi-output regression</li><li>sklearn.naive_bayes: Naive Bayes algorithms</li><li>sklearn.neighbors: Nearest neighbors algorithms</li><li>sklearn.neural_network: Neural network models</li><li>sklearn.nonlinearities: Activation functions for neural networks</li><li>sklearn.novelties: Outlier detection</li><li>sklearn.partial_dependence: Partial dependence plots</li><li>sklearn.pipeline: Pipeline tools</li><li>sklearn.preprocessing: Data preprocessing and normalization</li><li>sklearn.random_projection: Random projections</li><li>sklearn.semi_supervised: Semi-supervised learning</li><li>sklearn.svm: Support vector machines</li><li>sklearn.tree: Decision trees and tree-based models</li><li>sklearn.utils: Utility functions</li>
I hope this helps!

---

**User:**
for each item in the list, give important functions. also explain them briefly.

---

**ChatGPT:**
Sure, here are some important functions for each sub-package in scikit-learn (sklearn) and a brief explanation of what they do:
<li>sklearn.cluster:</li>
<li>KMeans: Clustering algorithm that partitions data into k clusters using the k-means algorithm.</li><li>AgglomerativeClustering: Hierarchical clustering algorithm that recursively merges the pair of clusters that minimally increases a given linkage distance.</li><li>DBSCAN: Density-based clustering algorithm that groups together points that are close to each other and separates those that are far away.</li>
<li>sklearn.compose:</li>
<li>ColumnTransformer: Applies transformers to specific columns of an array or dataframe.</li><li>TransformedTargetRegressor: Applies a transformer to the target variable before fitting a regression model.</li>
<li>sklearn.covariance:</li>
<li>EllipticEnvelope: Robust estimator of the covariance matrix of an elliptical distribution.</li><li>MinCovDet: Robust estimator of the covariance matrix that gives less weight to outliers.</li><li>EmpiricalCovariance: Unbiased estimator of the covariance matrix based on a set of observations.</li>
<li>sklearn.cross_decomposition:</li>
<li>PLSRegression: Partial least squares regression that finds linear combinations of the predictor variables that explain the maximum covariance with the target variable.</li><li>CCA: Canonical correlation analysis that finds linear combinations of two sets of variables that are maximally correlated.</li>
<li>sklearn.datasets:</li>
<li>load_digits: Loads the handwritten digits dataset.</li><li>load_boston: Loads the Boston Housing dataset.</li><li>make_classification: Generates a random classification dataset with specified number of samples and features.</li>
<li>sklearn.decomposition:</li>
<li>PCA: Principal component analysis that finds the linear combinations of the original features that explain the maximum variance in the data.</li><li>NMF: Non-negative matrix factorization that factorizes a non-negative matrix into two non-negative matrices with a reduced rank.</li><li>FactorAnalysis: Factor analysis that models the observed variables as linear combinations of unobserved factors.</li>
<li>sklearn.discriminant_analysis:</li>
<li>LinearDiscriminantAnalysis: Linear discriminant analysis that finds the linear combination of features that maximizes the ratio of between-class variance to within-class variance.</li><li>QuadraticDiscriminantAnalysis: Quadratic discriminant analysis that models the distribution of each class using a quadratic function.</li>
<li>sklearn.ensemble:</li>
<li>RandomForestClassifier: Random forest classifier that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control overfitting.</li><li>GradientBoostingRegressor: Gradient boosting regressor that fits a series of regression trees on the residuals of the previous trees and uses a shrinkage parameter to prevent overfitting.</li><li>VotingClassifier: Voting classifier that combines the predictions of multiple classifiers using a majority voting rule.</li>
<li>sklearn.exceptions:</li>
<li>NotFittedError: Exception that is raised when a method that requires a fitted estimator is called before the estimator has been fit to data.</li><li>ConvergenceWarning: Warning that is raised when an iterative optimization algorithm fails to converge.</li>
<li>sklearn.experimental:</li>
<li>enable_hist_gradient_boosting: Enables the experimental histogram-based gradient boosting estimator.</li><li>enable_halving_search_cv: Enables the experimental hyperparameter tuning using successive halving and early stopping.</li>
<li>sklearn.feature_extraction:</li>
<li>CountVectorizer: Converts a collection of text documents into a matrix of token counts.</li><li>TfidfTransformer: Transforms a matrix of token counts into a matrix of tf-idf values.</li><li>DictVectorizer: Converts a collection of dictionaries into a matrix of token counts or tf-idf values.</li>
<li>sklearn.gaussian_process:</li>
<li>GaussianProcessRegressor: Gaussian process regression that models</li>

---

**User:**
Continue writing please

---

**ChatGPT:**
the target function as a Gaussian process and uses Bayesian inference to make predictions.
<li>GaussianProcessClassifier: Gaussian process classifier that models the decision function as a Gaussian process and uses Bayesian inference to classify new data points.</li>
<li>sklearn.inspection:</li>
<li>permutation_importance: Computes feature importance scores for a fitted estimator by permuting the values of each feature and measuring the resulting decrease in the score.</li><li>partial_dependence: Computes partial dependence plots for a fitted estimator and a subset of features.</li>
<li>sklearn.isotonic:</li>
<li>IsotonicRegression: Non-parametric regression that fits a non-decreasing function to the data using isotonic regression.</li>
<li>sklearn.kernel_approximation:</li>
<li>RBFSampler: Random kitchen sinks approximation that maps data into a higher-dimensional space using a random subset of the radial basis function (RBF) kernel.</li><li>Nystroem: Nystrom approximation that uses a subset of training points to approximate the kernel matrix and maps new data points into the feature space using the approximation.</li>
<li>sklearn.kernel_ridge:</li>
<li>KernelRidge: Kernel ridge regression that fits a linear combination of kernel functions to the data.</li>
<li>sklearn.linear_model:</li>
<li>LinearRegression: Ordinary least squares regression that minimizes the sum of squared errors between the predicted and actual values.</li><li>LogisticRegression: Logistic regression that models the probability of belonging to a certain class as a logistic function of the predictor variables.</li><li>Lasso: Lasso regression that penalizes the sum of absolute values of the regression coefficients to perform feature selection.</li>
<li>sklearn.manifold:</li>
<li>TSNE: t-Distributed Stochastic Neighbor Embedding that reduces the dimensionality of the data by modeling the pairwise similarities between the points using a t-distribution.</li><li>MDS: Multidimensional scaling that finds a low-dimensional representation of the data that preserves the pairwise distances between the points.</li>
<li>sklearn.metrics:</li>
<li>accuracy_score: Computes the accuracy score of a classification model by comparing the predicted and actual labels.</li><li>r2_score: Computes the R-squared score of a regression model by measuring the proportion of the variance in the data that is explained by the model.</li><li>precision_score: Computes the precision score of a classification model by measuring the proportion of true positives among the predicted positives.</li>
<li>sklearn.mixture:</li>
<li>GaussianMixture: Gaussian mixture model that models the data as a mixture of Gaussian distributions and uses the expectation-maximization algorithm to estimate the parameters.</li>
<li>sklearn.model_selection:</li>
<li>train_test_split: Splits the data into random train and test subsets.</li><li>GridSearchCV: Performs an exhaustive search over a specified parameter grid for a given estimator and returns the estimator with the best performance.</li><li>RandomizedSearchCV: Performs a randomized search over a specified parameter distribution for a given estimator and returns the estimator with the best performance.</li>
<li>sklearn.multioutput:</li>
<li>MultiOutputRegressor: Multi-output regression that fits a separate regressor for each output variable.</li>
<li>sklearn.naive_bayes:</li>
<li>GaussianNB: Gaussian naive Bayes that models the likelihood of the predictor variables as a Gaussian distribution for each class and uses Bayes' theorem to compute the posterior probability of each class.</li><li>MultinomialNB: Multinomial naive Bayes that models the likelihood of the predictor variables as a multinomial distribution for each class and uses Bayes' theorem to compute the posterior probability of each class.</li>
<li>sklearn.neighbors:</li>
<li>KNeighborsClassifier: k-nearest neighbors classifier that assigns the majority class among the k-nearest neighbors of each data point.</li><li>RadiusNeighborsClassifier: Radius-based neighbors classifier that assigns the majority class among</li>

---

**User:**
Continue writing please

---

**ChatGPT:**
the neighbors within a fixed radius of each data point.
<li>KNeighborsRegressor: k-nearest neighbors regression that predicts the average of the target values of the k-nearest neighbors of each data point.</li>
<li>sklearn.neural_network:</li>
<li>MLPClassifier: Multi-layer perceptron classifier that trains a neural network with one or more hidden layers to predict the class labels.</li><li>MLPRegressor: Multi-layer perceptron regression that trains a neural network with one or more hidden layers to predict the continuous target values.</li>
<li>sklearn.pipeline:</li>
<li>Pipeline: Chains multiple estimators into a single object to simplify the workflow of cross-validation, hyperparameter tuning, and model selection.</li>
<li>sklearn.preprocessing:</li>
<li>StandardScaler: Scales the data to have zero mean and unit variance.</li><li>MinMaxScaler: Scales the data to the range [0, 1].</li><li>LabelEncoder: Encodes categorical variables as integers.</li>
<li>sklearn.svm:</li>
<li>SVC: Support vector machine classifier that maximizes the margin between the classes by finding the hyperplane that separates the data points with the largest margin.</li><li>SVR: Support vector machine regression that finds the hyperplane that best fits the data by minimizing the sum of squared errors between the predicted and actual target values.</li>
<li>sklearn.tree:</li>
<li>DecisionTreeClassifier: Decision tree classifier that recursively splits the data into subsets based on the values of the predictor variables to maximize the information gain.</li><li>DecisionTreeRegressor: Decision tree regression that recursively splits the data into subsets based on the values of the predictor variables to minimize the sum of squared errors between the predicted and actual target values.</li>
<li>sklearn.utils:</li>
<li>shuffle: Shuffles the data randomly along the first axis (i.e., the samples) without changing the target values.</li><li>resample: Resamples the data randomly with replacement to balance the classes.</li>