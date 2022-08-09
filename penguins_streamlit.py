import streamlit as st
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


with open('random_forest_penguin.pickle', 'rb') as f:
    rfc = pickle.load(f)

with open('output_penguin.pickle', 'rb') as f:
    unique_penguin_mapping = pickle.load(f)

st.title('Penguin Classifier')

st.write("This app uses 6 inputs to predict the species of penguin using a model built on the Palmer's Penguin's dataset. Use the form below to get started!")


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
    submitted = st.form_submit_button()
    if submitted:
        pass
    else:
        st.stop()

new_prediction = rfc.predict([[bill_length, bill_depth, flipper_length, body_mass, island_enc[0], island_enc[1], island_enc[2],
                               sex_enc[0], sex_enc[1]]])

prediction_species = unique_penguin_mapping[new_prediction][0]

st.subheader("Predicting Your Penguin's Species:")

st.write('We predict your penguin is of the {} species'.format(prediction_species))


st.write('We used a machine learning (Random Forest) model to predict the species, the features used in this prediction are ranked by relative importance below.')
st.image('feature_importance.png')


st.write('Below are the histograms for each continuous variable separated by penguin species. The vertical line represents your the inputted value.')


@st.cache
def load_csv():
    df = pd.read_csv('penguins.csv')
    df = df.dropna()
    return df


penguin_df = load_csv()

fig, ax = plt.subplots()
ax = sns.displot(x=penguin_df['bill_length_mm'],
                 hue=penguin_df['species'])
plt.axvline(bill_length, label='Your Instance')
plt.legend()
plt.title('Bill Length by Species')
st.pyplot(ax)

fig, ax = plt.subplots()
ax = sns.displot(x=penguin_df['bill_depth_mm'], hue=penguin_df['species'])
plt.axvline(bill_depth, label='Your Instance')
plt.legend()
plt.title('Bill Depth by Species')
st.pyplot(ax)

fig, ax = plt.subplots()
ax = sns.displot(x=penguin_df['flipper_length_mm'], hue=penguin_df['species'])
plt.axvline(flipper_length, label='Your Instance')
plt.legend()
plt.title('Flipper Length by Species')
st.pyplot(ax)
