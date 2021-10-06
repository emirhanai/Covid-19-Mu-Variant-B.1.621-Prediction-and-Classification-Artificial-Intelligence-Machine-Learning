#libraries of the loading
from sklearn.ensemble import *
from sklearn.model_selection import *
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import *

#datasets loading...
mu_variant_data = pd.read_csv('mu-variant-data.csv')

#change function..
def number_gender_change(x):
    match x:
        case 'Male':
            return 0
        case 'Female':
            return 1
        case _:
            return 2

#change of the str to int.
mu_variant_data['Gender_Change'] = mu_variant_data['Gender'].apply(number_gender_change)

#creating of fetures
coverage_X = mu_variant_data.loc[:,['S:N501Y','S:E484K','S:R346K','S:P681H','Coverage']].values

#creating of class,target or label :)
gender_y = mu_variant_data.loc[:,['Gender_Change']].values

#fetaures print
#print(gender_y)

#try :)

#                                 *******CLASSIFICATION IS THE MU VARIANT*******
# we create the Train and Test datasets.. (parameters of the suitable)
coverage_X_train, coverage_X_test, gender_y_train, gender_y_test = train_test_split(
    coverage_X,
    gender_y,
    test_size=0.22,
    stratify=None,
    shuffle=True,
    random_state=64
)

# we create of the our model 8Classifier)
model_mu_variant = BaggingClassifier(
    base_estimator=None,
    random_state=46,
    n_estimators=2,
    bootstrap=True,
    max_features=1.0,
    n_jobs=-1,
    warm_start=False
)

# model is fitting with Train datasets.
model_mu_variant.fit(coverage_X_train, gender_y_train)

# Creating the Prediction equation
prediction = model_mu_variant.predict(coverage_X_test)

# Creating of Accuracy score, F1 Score, Recall Score, Precision Score and Confusion Matrix :)
print(f"Accuracy score: {accuracy_score(gender_y_test, prediction)}")
print(f"F1 score: {f1_score(gender_y_test, prediction)}")
print(f"Recall score: {recall_score(gender_y_test, prediction)}")
print(f"Precision score: {precision_score(gender_y_test, prediction)}")
# print(f"Confusion Matrix: {confusion_matrix(gender_y_test, prediction)}")

    #                                 *******PREDICTION IS THE MU VARIANT*******
X_train, X_test, y_train, y_test = train_test_split(
    coverage_X,
    gender_y,
    test_size=0.05,
    stratify=None,
    shuffle=True,
    random_state=265
)

# creating of our predicting model
model_mu_variant_predict = BaggingRegressor(
    base_estimator=None,
    random_state=149,
    n_estimators=1,
    bootstrap=True,
    max_features=1.0,
    n_jobs=-1,
    warm_start=True, 
    bootstrap_features=True
)

# our model is fitting..
model_mu_variant_predict.fit(X_train, y_train)

# our model is predicting..
a = model_mu_variant_predict.predict(X_test)

predi = model_mu_variant_predict.predict([[1, 1, 0, 1, 2]])

for i in predi:
    match i:
        case i == [1.]:
            print("Congrats! Prediction Gender is the Female!")
        case i == [0.]:
            print("Congrats! Prediction Gender is the Male!")
        case _:
            print("Sorry! Prediction Gender is the Unknown!")

# our prediction software is the accuracy score :)
print(f"Prediction Model Accuracy Score: {r2_score(y_test, a)} ")


feature_names = mu_variant_data.loc[:,['S:N501Y','S:E484K','S:R346K','S:P681H','Coverage']].columns

from sklearn.tree import *

def save_decision_trees_as_dot(model_mu_variant, iteration, feature_name):
    #file_name = open("emirhan_mu_variant_classification" + str(iteration) + ".dot", 'w')
    dot_data = export_graphviz(
        model_mu_variant,
        #out_file=file_name,
        feature_names=feature_name,
        class_names=['Male','Female','Unknown'],
        rounded=True,
        proportion=False,
        precision=2,
        filled=True
    )
    #file_name.close()
    print("Classification {} saved as dot file".format(iteration + 1))


# Save of the .dot loop..
#for i in range(len(model_mu_variant.estimators_)):
    #save_decision_trees_as_dot(model_mu_variant.estimators_[i], i, feature_names)
    #print(i)


plt.scatter(mu_variant_data['Coverage'],mu_variant_data['Gender'])
plt.legend(["Coverage\nGender"])
plt.xlabel("Coverage")
plt.ylabel("Gender")
plt.title("Covid-19 Mu Variant [B.1.621] -- Gender/Coverage")
plt.show()
