from from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
import joblib
# load dataset from sklearn
housing = datasets.fetch_california_housing()



# split data to features and targets
x = housing.data
y = housing.target


train_test_split(x, y, random_state=432)

x_train, x_test, y_train, y_test = train_test_split(
    x,  y, test_size=0.2,
    random_state=432 )

print("old numebr of features:", len(x_train[0]))

poly = PolynomialFeatures()
x_train = poly.fit_transform(x_train)
x_test = poly.fit_transform(x_test)

print("new numebr of features:", len(x_train[0]))

model = HistGradientBoostingRegressor(
    max_iter=400, # best max iter
    learning_rate=0.05 # best learning rate
)
model.fit(x_train, y_train)

# save model
joblib.dump(model, "my_model.joblib")

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print("existing model score:", r2)

# load saved model
saved_model = joblib.load("my_model.joblib")
# evaluate saved model
y_pred = saved_model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print("saved model score:", r2)
 import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
import joblib
# load dataset from sklearn
housing = datasets.fetch_california_housing()



# split data to features and targets
x = housing.data
y = housing.target


train_test_split(x, y, random_state=432)

x_train, x_test, y_train, y_test = train_test_split(
    x,  y, test_size=0.2,
    random_state=432 )

print("old numebr of features:", len(x_train[0]))

poly = PolynomialFeatures()
x_train = poly.fit_transform(x_train)
x_test = poly.fit_transform(x_test)

print("new numebr of features:", len(x_train[0]))

model = HistGradientBoostingRegressor(
    max_iter=400, # best max iter
    learning_rate=0.05 # best learning rate
)
model.fit(x_train, y_train)

# save model
joblib.dump(model, "my_model.joblib")

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print("existing model score:", r2)

# load saved model
saved_model = joblib.load("my_model.joblib")
# evaluate saved model
y_pred = saved_model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print("saved model score:", r2)
