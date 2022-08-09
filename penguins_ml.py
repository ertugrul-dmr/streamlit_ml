import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

penguin_df = pd.read_csv('penguins.csv')
penguin_df.dropna(inplace=True)
output = penguin_df['species']
features = penguin_df[['island', 'bill_length_mm', 'bill_depth_mm',
                       'flipper_length_mm', 'body_mass_g',
                       'sex']]
features = pd.get_dummies(features)
output, uniques = pd.factorize(output)


x_train, x_test, y_train, y_test = train_test_split(
    features, output, test_size=.8)

rfc = RandomForestClassifier(random_state=15)
rfc.fit(x_train, y_train)
y_pred = rfc.predict(x_test)
score = accuracy_score(y_pred, y_test)
print('Our accuracy score for this model is {}'.format(score))

with open("random_forest_penguin.pickle", "wb") as f:
    pickle.dump(rfc, f)

with open("output_penguin.pickle", "wb") as f:
    pickle.dump(uniques, f)

fig, ax = plt.subplots()

importances = pd.DataFrame(
    {'Importance': rfc.feature_importances_, 'feature': features.columns})

ax = sns.barplot(data=importances.sort_values(by='Importance', ascending=False),
                 x='Importance', y='feature', orient='h')
plt.title('Which features are the most important for species prediction?')
plt.ylabel('Importance')
plt.xlabel('Feature')
plt.tight_layout()
fig.savefig('feature_importance.png')
