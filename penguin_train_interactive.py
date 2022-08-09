import streamlit as st
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.title('Penguin Classifier')

st.write("This app uses 6 inputs to predict the species of penguin using a model built on the Palmer's Penguin's dataset. Use the form below to get started!")


penguin_file = st.file_uploader('Upload your own penguin data')

if penguin_file is None:
    st.write('Using pretrained weights')
    with open('random_forest_penguin.pickle', 'rb') as f:
        rfc = pickle.load(f)

    with open('output_penguin.pickle', 'rb') as f:
        unique_penguin_mapping = pickle.load(f)
    st.image('feature_importance.png')

else:
    st.write('Training model from given dataset')
    penguin_df = pd.read_csv(penguin_file)
    penguin_df = penguin_df.dropna()
    output = penguin_df['species']
    features = penguin_df[['island', 'bill_length_mm',
                           'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]
    features = pd.get_dummies(features)
    output, unique_penguin_mapping = pd.factorize(output)
    x_train, x_test, y_train, y_test = train_test_split(
        features, output, test_size=.8)
    rfc = RandomForestClassifier(random_state=15)
    rfc.fit(x_train, y_train)
    y_pred = rfc.predict(x_test)
    score = round(accuracy_score(y_pred, y_test), 2)
    st.write('We trained a Random Forest model on these data, it has a score of {}! Use the inputs below to try out the model.'.format(score))
with st.form('user_inputs'):
    island = st.selectbox(label='Penguin Island', options=[
        'Biscoe', 'Dream', 'Torgerson'])
    island_enc = [0 if i != island else 1 for i in [
        'Biscoe', 'Dream', 'Torgerson']]

    sex = st.selectbox('Sex', options=['Female', 'Male'])
    sex_enc = [0 if i != sex else 1 for i in ['Female', 'Male']]

    bill_length = st.number_input('Bill Length (mm)', min_value=0)
    bill_depth = st.number_input('Bill Depth (mm)', min_value=0)
    flipper_length = st.number_input('Flipper Length (mm)', min_value=0)
    body_mass = st.number_input('Body Mass (g)', min_value=0)
    st.form_submit_button()


new_prediction = rfc.predict([[bill_length, bill_depth, flipper_length, body_mass, island_enc[0], island_enc[1], island_enc[2],
                               sex_enc[0], sex_enc[1]]])

prediction_species = unique_penguin_mapping[new_prediction][0]

st.write('We predict your penguin is of the {} species'.format(prediction_species))
