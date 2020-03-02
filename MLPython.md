# Machine Learning Snippets with Python

Los Snippets son trozos de código, o chuletas, en este caso sobre Machine Learning con Python.


## Dataset preparation
```python
# Las variables independientes X se tienen que dar en forma de dataframe de 2 dimensiones
X = df[['col1','col2','col3']] 

# La variable dependiente y (target) viene en forma de vector (serie)
y = df['col_target']
```

## Separación Train/Validation/Test

Si se tiene un dataset suficientemente grande conviene separar en train, validation y test (sino solo en train y test):
- con __Train__: se entrena el modelo
- con __Validation__: se le aplica el modelo que se ha entrenado y se ajustan los hiperparámetros para mejorar las métricas
- con __Test__: solo se dan resultados, no se pueden ajustar parámetros dependiendo de los resultados de Test


___Nota:___ 
Mínimo 1000 registros para el conjunto de Validation.


___Nota:___ 
Validation y Test tienen que seguir la misma distribución. 

Si aún teniendo la misma distribución, las métricas son muy distintas, puede ser por un sobreajuste al conjunto de Validation. Se podría arreglar:
- usando cross-validation
- usando más datos en el conjunto de Validation
- cambiando la métrica con la que se compara (porque puede ser muy inestable con pequeños cambios de las distribuciones)



```python
# Import the library
from sklearn.model_selection import train_test_split

# Create 2 groups each with inputs X and targets y
# test_size fija el porcentaje de datos que se guardarán como Test
# random_state fija la semilla aleatoria para trabajar siempre con los mismos datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state = 42)

# Fit only with training data
reg.fit(X_train,y_train)
```

## Cross Validation

```python
# Load the library
from sklearn.model_selection import cross_val_score

# We calculate the metric for several subsets
# Hay que indicarle el modelo (reg),
# el número de particiones (cv) que se quiere hacer al conjunto 
# y la métrica que se quiere usar para evaluar los resultados (scoring)
cross_val_score(reg,X,y,cv=5,scoring="neg_mean_squared_error")

```




# Regression

## Linear Regression

Parameters: none

```python
# Load the library
from sklearn.linear_model import LinearRegression

# Create an instance of the model
regL = LinearRegression()

# Fit the regressor
regL.fit(X_train,y_train)

# Do predictions
y_pred_regL = regL.predict(X_test)
other_pred_regL = regL.predict([[2540],[3500],[4000]])
```


## K Nearest Neighbors

KNN es más visual para clasificación pero también se puede usar en regresión.
Elige los puntos más cercanos al punto X (cuya y queremos predecir) y luego predice la y como la media de las y de esos puntos.

Main parameters: 
- _n_neighbors_: cantidad de elementos en el grupo (NO es el número de grupos como en K-Means)
- _weights_: función de peso para determinar cuales son los puntos más cercanos (se puede definir una nueva función)

      - uniform: distribución uniforme de los pesos
      - distance: los puntos más cercanos pesan más -> más sensible a outliers

```python
# Load the library
from sklearn.neighbors import KNeighborsRegressor

# Create an instance
regKN = KNeighborsRegressor(n_neighbors=2)

# Fit the data
regKN.fit(X_train,y_train)

# Do predictions
y_pred_regKN = regKN.predict(X_test)
```


## SVM Support Vector Machine

Main parameters:
- _kernel_: tipo de función que se le aplica a los datos para aumentar la dimensión y que sea linealmente separable
      - linear, poly, rbf, sigmoid,...
- _C_: parámetro de regularización (L2 Ridge), inversamente proporcionales
- _epsilon_: margen de seguridad donde no se le aplica penalización 

```python
# Load the library
from sklearn.svm import SVR

# Create an instance
regSVR = SVR(kernel="rbf",C=0.1)

# Fit the data
regSVR.fit(X_train,y_train)

# Do predictions
y_pred_regSVR = regSVR.predict(X_test)
```



## Decision Tree

Se elige el atributo (valor de una columna) que consigue dividir la muestra de la mejor manera.

Main parameters:
- _max_depth_: por defecto es None -> los nodos se expanden hasta que las hojas son puras o hasta que contienen min_samples_leaf elementos
- _min_samples_leaf_: mínimo número de elementos en un nodo

```python
# Load the library
from sklearn.tree import DecisionTreeRegressor

# Create an instance
regDTR = DecisionTreeRegressor(max_depth=3)

# Fit the data
regDTR.fit(X_train,y_train)
```


# Cross Validation and testing parameters

Cuando tenemos modelos más complejos dependiendo de muchos hiperparámetros necesitamos un método de validación cruzada que haga modelos combinándolos unos con otros:
- __GridSearchCV__: prueba los parámetros todos con todos
- __RandomSearchCV__: prueba aleatoriamente la combinación de unos parámetros con otros 


## Grid Search
```python
# Import libraries for GridSearch and model
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor

# Create an instance
# GridSearch hará la combinación de parámetros que estén dentro de param_grid
regKN_GS = GridSearchCV(KNeighborsRegressor(),
                        param_grid={"n_neighbors":np.arange(3,50)},  
                        cv = 5,
                        scoring = "neg_mean_absolute_error")
                        
# Fit will test all of the combinations
regKN_GS.fit(X_train,y_train)

# Print best parameters
print(regKN_GS.best_score_)
print(regKN_GS.best_params_)

# Take best estimator (best model)
regKN_best = regKN_GS.best_estimator_

# Do predictions
y_pred_regKN = regKN_best.predict(X_test)
```


## Randomized Grid Search
```python
# Import libraries
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor

# Create an instance
# Trabaja como GridSearch pero hay que definir el número de iteraciones aleatorias entre parámetros con n_iter
regDT_RS = RandomizedSearchCV(DecisionTreeRegressor(),
                              param_distributions={"max_depth":np.arange(2,8),
                                                   "min_samples_leaf":[10,30,50,100]},
                              cv=5, 
                              scoring="neg_mean_absolute_error",                 
                              n_iter=5)

# Fit will test all of the combinations
regDT_RS.fit(X_train,y_train)

# Print best parameters
print(regDT_RS.best_score_)
print(regDT_RS.best_params_)

# Take best estimator (best model)
regDT_best = regDT_RS.best_estimator_

# Do predictions
y_pred_regDT = regDT_best.predict(X_test)

```


## Random Forest

Main parameters:
- _max_depth_
- _min_samples_leaf_
- _n_estimators_: número de árboles en el forest

```python
# Import the library
from sklearn.ensemble import RandomForestRegressor

# Create an instance
# verbose es para que imprima mensajes de lo que va haciendo (lo vuelve más lento)
regRF_GS = GridSearchCV(RandomForestRegressor(),n_jobs=-1,
                        param_grid = {"min_samples_leaf":[1,2,3],
                                       "max_depth":np.arange(3,20),
                                       "n_estimators":[500]},
                  cv=5,
                  scoring="neg_mean_absolute_error",
                  verbose=9)
      
# Fit will test all of the combinations
regRF_GS.fit(X_train,y_train)

# Print best parameters
print(regRF_GS.best_params_)
print(regRF_GS.best_score_)

# Take best estimator (best model)
regRF_best = regRF_GS.best_estimator_

# Do predictions
y_pred_regRF = regRF_best.predict(X_test)
```



# XGBooster

Primero hay que instalarlo: #conda install -c anaconda py-xgboost

Main parameters: 
- _max_depth_
- _eta_ (learning rate): después de cada boosting se pueden obtener los pesos de los features, el eta los modifica para que no haya overfitting

```python
# Load the library
from sklearn.model_selection import GridSearchCV
from xgboost.sklearn import XGBRegressor

# Create an instance
xgb1 = XGBRegressor()
parameters = {'nthread':[4], 
              'learning_rate': [0.01, 0.05, 0.1], 
              'max_depth':np.arange(3,10),
              'silent': [1],
              'subsample': [0.7],
              'colsample_bytree':np.arange(0.3,1),
              'n_estimators': [100]}

regXGBGS = GridSearchCV(xgb1,
                        parameters,
                        cv = 5,
                        n_jobs = -1,
                        scoring="neg_mean_absolute_error",
                        verbose=9)

# Fit the data
regXGBGS.fit(X_train,y_train)

# Print best parameters
print(regXGBGS.best_params_)
print(regXGBGS.best_score_)

# Take best estimator (best model)
regXGBGS_best = regXGBGS.best_estimator_

# Do predictions
y_regXGBGS_pred = regXGBGS_best.predict(X_test)
```



# Metrics
## Regression

### MAE
```python
# Load the scorer
from sklearn.metrics import mean_absolute_error

# Use against predictions
mean_absolute_error(reg.predict(X_test),y_test)
```

### MAPE
```python
np.mean(np.abs(reg.predict(X_test)-y_test)/y_test)
```

### RMSE
```python
# Load the scorer
from sklearn.metrics import mean_squared_error

# Use against predictions (we must calculate the square root of the MSE)
np.sqrt(mean_squared_error(reg.predict(X_test),y_test))
```

### Correlation
```python
# Direct Calculation
np.corrcoef(reg.predict(X_test),y_test)[0][1]

# Custom Scorer: si lo queremos meter como métrica en un CV donde no lo hay
from sklearn.metrics import make_scorer
def corr(pred,y_test):
return np.corrcoef(pred,y_test)[0][1]

# Put the scorer in cross_val_score
cross_val_score(reg,X,y,cv=5,scoring=make_scorer(corr))
```

### Bias
```python
# Direct Calculation
np.mean(reg.predict(X_test)-y_test)

# Custom Scorer
from sklearn.metrics import make_scorer
def bias(pred,y_test):
return np.mean(pred-y_test)

# Put the scorer in cross_val_score
cross_val_score(reg,X,y,cv=5,scoring=make_scorer(bias))
```




# Classification
## Logistic Regression

Cuando la regresión lineal funciona mal se suele pasar a la logística poniendo 0 y 1 según una cota que se decida.

```python
# Load the library
from sklearn.linear_model import LogisticRegression

# Create an instance of the classifier
clLR=LogisticRegression()

# Fit the data
clLR.fit(X_train,y_train)

# Se puede hacer una validación cruzada (la regresión no tiene parámetros para hacer un GridSearch)
from sklearn.model_selection import cross_val_score
cross_val_score(clLR,X,y,cv=5,scoring = 'accuracy').mean()
```



## K Nearest Neighbors

Se eligen los grupos dependiendo de la cantidad de vecinos más cercanos (k)

Main parameters: 
- n_neighbors: cantidad de elementos en el grupo (NO es el número de grupos como en K-Means)
- weights: función de peso para determinar cuales son los puntos más cercanos (se puede definir una nueva función)

      - uniform: distribución uniforme de los pesos
      - distance: los puntos más cercanos pesan más -> más sensible a outliers


### Sin GridSearchCV
```python
# Load the library
from sklearn.neighbors import KNeighborsClassifier

# Create an instance
clKN = KNeighborsClassifier(n_neighbors=2)

# Fit the data
clKN.fit(X_train,y_train)
```

### Con GridSearchCV
```python
# Import Library
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

# Create instance
clKN_GS = GridSearchCV(KNeighborsClassifier(),
                        param_grid = {"n_neighbors":np.arange(3,50)},
                        cv=5,
                        scoring="accuracy",
                        verbose=9)
                                       
# Fit will test all of the combinations
clKN_GS .fit(X_train,y_train)

# Print best parameters
print(clKN_GS.best_params_)
print(clKN_GS.best_score_)

# Take best estimator (best model)
clKN_best = clKN_GS.best_estimator_

# Do predictions
y_pred_clKN = clKN_best.predict(X_test)
```



## SVM Support Vector Machine

Clasifica elementos que sean linealmente separables (busca una frontera entre ellos). 

Por eso cuando los datos no se reparten de manera lineal hay que aplicarle funciones que les aumenten dimensiones (porque aumenta la probabilidad de que el conjunto sea linealmente separable).

Crea un margen de seguridad buscando la linea que quede más lejos de los puntos.

Main parameters:
- C: Sum of Error Margins
- kernel:
      - linear: line of separation
      - rbf: circle of separation
            * Additional paramater -> gamma: Inverse of the radius
      - poly: curved line of separation
            * Additional paramater -> degree: Degree of the polynome


### Sin GridSearchCV
```python
# Load the library
from sklearn.svm import SVC
# Create an instance of the classifier
clf = SVC(kernel="linear",C=10)
# Fit the data
clf.fit(X,y)
```

### Con GridSearchCV
```python
# Import Library
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# Create instance
clSVC_GS = GridSearchCV(SVC(kernel="poly"),
                        param_grid = {"C":np.arange(10,100),"degree":np.arange(1,5)},
                        cv=5,
                        scoring="accuracy")
                             
# Fit will test all of the combinations
clSVC_GS.fit(X_train,y_train)

# Print best parameters
print(clSVC_GS.best_params_)
print(clSVC_GS.best_score_)

# Take best estimator (best model)
clSVC_best = clSVC_GS.best_estimator_

# Do predictions
y_pred_clSVC = clSVC_best.predict(X_test)
```



## Decision Tree

Main parameters:
- max_depth: por defecto es None -> los nodos se expanden hasta que las hojas son puras o hasta que contienen min_samples_leaf elementos
- min_samples_leaf: mínimo número de elementos en un nodo


### Sin GridSearchCV
```python
# Import library
from sklearn.tree import DecisionTreeClassifier
# Create instance
clf = DecisionTreeClassifier(min_samples_leaf=20,max_depth=3)
# Fit the data
clf.fit(X,y)
```


### Con GridSearchCV
```python
# Import Library
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

# Create instance
clDT_GS = GridSearchCV(DecisionTreeClassifier(),
                        param_grid = {"min_samples_leaf":np.arange(3,50),
                                      "max_depth":np.arange(1,4)},
                        cv=5,
                        scoring="accuracy",
                        verbose=9)
                               
# Fit will test all of the combinations
clDT_GS.fit(X_train,y_train)

# Print best parameters
print(clDT_GS.best_params_)
print(clDT_GS.best_score_)

# Take best estimator (best model)
clDT_best = clDT_GS.best_estimator_

# Do predictions
y_pred_clDT = clDT_best.predict(X_test)
```




## Random Forest

Main parameters:
- max_depth
- min_samples_leaf
- n_estimators: número de árboles en el forest

```python
# Import Library
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Create instance
clRF_GS = GridSearchCV(RandomForestClassifier(n_jobs=-1),
                        param_grid = {"min_samples_leaf":[10,20,30,40,50],
                                     "max_depth":np.arange(1,4),
                                     "n_estimators":[50]},
                        cv=5,
                        scoring="accuracy",
                        verbose=9)
                     
                     
# Fit will test all of the combinations
clfRF_GS.fit(X_train,y_train)

# Print best parameters
print(clfRF_GS.best_params_)
print(clfRF_GS.best_score_)

# Take best estimator (best model)
clfRF_best = clfRF_GS.best_estimator_

# Do predictions
y_pred_clRF = clfRF_best.predict(X_test)
```



## Gradient Boosting Tree
```python
# Import Library
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

# Create instance
clGB_GS = GridSearchCV(GradientBoostingClassifier(n_estimators=100),
                        param_grid={"max_depth":np.arange(2,10),
                                    "learning_rate":np.arange(1,10)/10},
                        cv=5,
                        scoring="neg_mean_absolute_error")
                        verbose=9)   
                              
# Fit will test all of the combinations
clGB_GS.fit(X_train,y_train)

# Print best parameters
print(clGB_GS.best_params_)
print(clGB_GS.best_score_)

# Take best estimator (best model)
clGB_best = clGB_GS.best_estimator_

# Do predictions
y_pred_clGB = clGB_best.predict(X_test)
```


# Metrics
## Classification

!(https://www.researchgate.net/publication/328148379/figure/fig1/AS:679514740895744@1539020347601/Model-performance-metrics-Visual-representation-of-the-classification-model-metrics.png)


### Accuracy

Es la proporción de todo lo que he acertado frente a la población (TP+TN)/(TP+FN+FP+TN)
```python
# Import metrics
from sklearn.metrics import accuracy_score
accuracy_score(y_test,cl.predict(X_test))

# If using Cross Validation
cross_val_score(clf,X,y,scoring="accuracy")
```


### Precision and Recall

Precision: proporción de los valores predichos que ha sido acertada (TP)/(TP+FP) _(mis 1 predicted)_

Recall: proporción de los valores correctos (1) que ha sido predicha correctamente (TP)/(TP+FN) _(los 1 true)_
```python
# Import metrics
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix, classification_report
precision_score(y_test,cl.predict(X_test))
classification_report(y_test,cl.predict(X_test))

# If using Cross Validation
cross_val_score(cl,X,y,scoring="precision")
cross_val_score(cl,X,y,scoring="recall")
```


### ROC curve
```python
# Import metrics
from sklearn.metrics import roc_curve

# We chose the target
target_pos = 1  # Or 0 for the other class
fp,tp,_ = roc_curve(y_test,pred[:,target_pos])
plt.plot(fp,tp)
```


#### AUC
```python
# Import metrics
from sklearn.metrics import roc_curve, auc
fp,tp,_ = roc_curve(y_test,pred[:,1])
auc(fp,tp)

# If using Cross Validation
cross_val_score(cl,X,y,scoring="roc_auc")
```


