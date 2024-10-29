import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.title('⚙️ Machine Learning Penguin App')

st.info('This app builds a machine learning model for data on penguins!')

with st.expander("Data"):
  st.write("**Raw Data**")
  df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/penguins_cleaned.csv')
  df

  st.write("**x-axis data**")
  x_raw = df.drop("species", axis = 1)
  x_raw
  
  st.write("**y-axis data**")
  y_raw = df.species
  y_raw
  
with st.expander("Data visualisation"):
  st.scatter_chart(
    data=df,
    x="bill_length_mm",
    y="body_mass_g",
    color="species"
  )


with st.sidebar:
  st.header("Input Features")
  island = st.selectbox("Island", ("Biscoe", "Dream", "Torgerson"))
  bill_length_mm = st.slider("Bill length (mm)", 32.1, 59.6, 43.9)
  bill_depth_mm = st.slider("Bill depth (mm)", 13.1, 21.5, 17.2)
  flipper_length_mm = st.slider("Flipper length (mm)", 172.0, 231.0, 201.0)
  body_mass_g = st.slider("Body mass (g)", 2700.0, 6300.0, 4207.0)
  gender = st.selectbox("Gender", ("male", "female"))

#build df for slider input
data = {"island": island,
        "bill_length_mm": bill_length_mm,
        "bill_depth_mm": bill_depth_mm,
        "flipper_length_mm": flipper_length_mm,
        "body_mass_g": body_mass_g,
        "sex": gender,
}
input_df = pd.DataFrame(data, index=[0])
input_penguins = pd.concat([input_df, x_raw], axis = 0)

with st.expander("Input features"):
  st.write("**Input Penguin**")
  input_df
  st.write("**Combined Penguin Data**")
  input_penguins
  
#Data prep
#Encode X
encode = ["island", "sex"]
df_penguins = pd.get_dummies(input_penguins, prefix=encode)

x = df_penguins[1:]
input_row = df_penguins[:1]

#Encode Y
target_mapper = {"Adelie": 0, "Chinstrap": 1, "Gentoo": 2}

def target_encode(val):
  return target_mapper[val]

y = y_raw.apply(target_encode)
  

with st.expander("Data preparation"):
  st.write("**Encoded x (input penguin)**")
  input_row
  st.write("**Encoded y**")
  y
  
# Training the model

clf = RandomForestClassifier()
clf.fit(x, y)

# Apply model
prediction = clf.predict(input_row)
prediction_probability = clf.predict_proba(input_row)

df_prediction_probability = pd.DataFrame(prediction_probability)
df_prediction_probability.columns = ["Adelie", "Chinstrap", "Gentoo"]
df_prediction_probability.rename(columns={0: "Adelie", 1: "Chinstrap", 2: "Gentoo"})

# Display predicted species
st.subheader("Predicted Species")
st.dataframe(df_prediction_probability,
        column_config={
        "Adelie": st.column_config.ProgressColumn(
            "Adelie",
            format="%f",
            width = "medium",
            min_value=0,
            max_value=1,
        ),
        "Chinstrap": st.column_config.ProgressColumn(
            "Chinstrap",
            format="%f",
            width = "medium",
            min_value=0,
            max_value=1,
        ),
        "Gentoo": st.column_config.ProgressColumn(
            "Gentoo",
            format="%f",
            width = "medium",
            min_value=0,
            max_value=1,
        },
)



penguin_species = np.array(["Adelie", "Chinstrap", "Gentoo"])
st.success(penguin_species[prediction][0])

# Display Progress Column
