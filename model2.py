def model(X, y):
    X = X.copy()
    y = y.copy()
    pipeline = Pipeline(steps=[['scaler', MinMaxScaler()],
                               ['classifier', LogisticRegression(random_state=11, max_iter=1000)]])
    
    param_grid = {'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    
    grid_search = GridSearchCV(estimator=pipeline,
                               param_grid=param_grid,
                               scoring='accuracy',
                               n_jobs=-1,
                               cv=3)
    
    grid_search.fit(X, y)
    
    return grid_search

#Dropping sepal width
X_train = X_train.drop(columns='sepal width (cm)').copy()
iris_model1 = model(X_train, y_train)
print(f'Best params: {iris_model1.best_params_}\nBest score: {iris_model1.best_score_}')