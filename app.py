import pickle
import json
import streamlit as st
import pandas as pd

st.set_page_config(page_title='Mushroom', layout = 'wide', page_icon='🍄')

left_col, center_col, right_col = st.columns((1,2,1))


with center_col:
    st.header('Predict If A Mushroom Is Safe Or Not')

@st.cache_resource
def get_resources(std_scaler_filename, model_filename):
    model = pickle.load(open(model_filename, 'rb'))
    scaler = pickle.load(open(std_scaler_filename, 'rb'))
    return scaler, model

@st.cache_data
def get_data(data_path):
    return pd.read_csv(data_path)

std_scaler_filename = 'resources/standard_scaler.pkl'
model_filename = 'resources/svc_model_predictor_mushrooms.pkl'
data_path = 'mushrooms.csv'

standard_scaler, model =  get_resources(std_scaler_filename, model_filename)
df = get_data(data_path)

cap_shape_dict = {
    'Bell': 'b',
    'Conical': 'c',
    'Convex': 'x',
    'Flat': 'f',
    'Knobbed': 'k',
    'Sunken': 's',
}

cap_surface_dict = {
    'Fibrous': 'f',
    'Grooves': 'g',
    'Scaly': 'y',
    'Smooth': 's',
}

cap_color_dict = {
    'Brown': 'n',
    'Buff': 'b',
    'Cinnamon': 'c',
    'Gray': 'g',
    'Green': 'r',
    'Pink': 'p',
    'Purple': 'u',
    'Red': 'e',
    'White': 'w',
    'Yellow': 'y',
}

bruises_dict = {
    'Bruises': 't',
    'No': 'f',
}

odor_dict = {
    'Almond': 'a',
    'Anise': 'l',
    'Creosote': 'c',
    'Fishy': 'y',
    'Foul': 'f',
    'Musty': 'm',
    'None': 'n',
    'Pungent': 'p',
    'Spicy': 's',
}

gill_attachment_dict = {
    'Attached': 'a',
    'Descending': 'd',
    'Free': 'f',
    'Notched': 'n',
}

gill_spacing_dict = {
    'Close': 'c',
    'Crowded': 'w',
    'Distant': 'd',
}

gill_size_dict = {
    'Broad': 'b',
    'Narrow': 'n',
}

gill_color_dict = {
    'Black': 'k',
    'Brown': 'n',
    'Buff': 'b',
    'Chocolate': 'h',
    'Gray': 'g',
    'Green': 'r',
    'Orange': 'o',
    'Pink': 'p',
    'Purple': 'u',
    'Red': 'e',
    'White': 'w',
    'Yellow': 'y',
}

stalk_shape_dict = {
    'Enlarging': 'e',
    'Tapering': 't',
}

stalk_root_dict = {
    'Bulbous': 'b',
    'Club': 'c',
    'Cup': 'u',
    'Equal': 'e',
    'Rhizomorphs': 'z',
    'Rooted': 'r',
    'Missing': '?',
}

stalk_surface_above_ring_dict = {
    'Fibrous': 'f',
    'Scaly': 'y',
    'Silky': 'k',
    'Smooth': 's',
}

stalk_surface_below_ring_dict = {
    'Fibrous': 'f',
    'Scaly': 'y',
    'Silky': 'k',
    'Smooth': 's',
}

stalk_color_above_ring_dict = {
    'Brown': 'n',
    'Buff': 'b',
    'Cinnamon': 'c',
    'Gray': 'g',
    'Orange': 'o',
    'Pink': 'p',
    'Red': 'e',
    'White': 'w',
    'Yellow': 'y',
}

stalk_color_below_ring_dict = {
    'Brown': 'n',
    'Buff': 'b',
    'Cinnamon': 'c',
    'Gray': 'g',
    'Orange': 'o',
    'Pink': 'p',
    'Red': 'e',
    'White': 'w',
    'Yellow': 'y',
}

veil_type_dict = {
    'Partial': 'p',
}

veil_color_dict = {
    'Brown': 'n',
    'Orange': 'o',
    'White': 'w',
}

ring_number_dict = {
    'None': 'n',
    'One': 'o',
    'Two': 't',
}

ring_type_dict = {
    'Cobwebby': 'c',
    'Evanescent': 'e',
    'Flaring': 'f',
    'Large': 'l',
    'None': 'n',
    'Pendant': 'p',
    'Sheathing': 's',
    'Zone': 'z',
}

spore_print_color_dict = {
    'Black': 'k',
    'Brown': 'n',
    'Buff': 'b',
    'Chocolate': 'h',
    'Green': 'r',
    'Orange': 'o',
    'Purple': 'u',
    'White': 'w',
    'Yellow': 'y',
}

population_dict = {
    'Abundant': 'a',
    'Clustered': 'c',
    'Numerous': 'n',
    'Scattered': 's',
    'Several': 'v',
    'Solitary': 'y',
}

habitat_dict = {
    'Grasses': 'g',
    'Leaves': 'l',
    'Meadows': 'm',
    'Paths': 'p',
    'Urban': 'u',
    'Waste': 'w',
    'Woods': 'd',
}



cap_shape = st.selectbox('Cap Shape', options=cap_shape_dict)
cap_surface = st.selectbox('Cap Surface', options=cap_surface_dict)
cap_color = st.selectbox('Cap Color', options=cap_color_dict)
bruises = st.selectbox('Bruises', options=bruises_dict)
odor = st.selectbox('Odor', options=odor_dict)
gill_attachment = st.selectbox('Gill Attachment', options=gill_attachment_dict)
gill_spacing = st.selectbox('Gill Spacing', options=gill_spacing_dict)
gill_size = st.selectbox('Gill Size', options=gill_size_dict)
gill_color = st.selectbox('Gill Color', options=gill_color_dict)
stalk_shape = st.selectbox('Stalk Shape', options=stalk_shape_dict)
stalk_root = st.selectbox('Stalk Root', options=stalk_root_dict)
stalk_surface_above_ring = st.selectbox('Stalk Surface Above Ring', options=stalk_surface_above_ring_dict)
stalk_surface_below_ring = st.selectbox('Stalk Surface Below Ring', options=stalk_surface_below_ring_dict)
stalk_color_above_ring = st.selectbox('Stalk Color Above Ring', options=stalk_color_above_ring_dict)
stalk_color_below_ring = st.selectbox('Stalk Color Below Ring', options=stalk_color_below_ring_dict)
veil_type = st.selectbox('Veil Type', options=veil_type_dict)
veil_color = st.selectbox('Veil Color', options=veil_color_dict)
ring_number = st.selectbox('Ring Number', options=ring_number_dict)
ring_type = st.selectbox('Ring Type', options=ring_type_dict)
spore_print_color = st.selectbox('Spore Print Color', options=spore_print_color_dict)
population = st.selectbox('Population', options=population_dict)
habitat = st.selectbox('Habitat', options=habitat_dict)
button = st.button('Get Results')

input_data = {
    'cap-shape': cap_shape_dict[cap_shape],
    'cap-surface': cap_surface_dict[cap_surface],
    'cap-color': cap_color_dict[cap_color],
    'bruises': bruises_dict[bruises],
    'odor': odor_dict[odor],
    'gill-attachment': gill_attachment_dict[gill_attachment],
    'gill-spacing': gill_spacing_dict[gill_spacing],
    'gill-size': gill_size_dict[gill_size],
    'gill-color': gill_color_dict[gill_color],
    'stalk-shape': stalk_shape_dict[stalk_shape],
    'stalk-root': stalk_root_dict[stalk_root],
    'stalk-surface-above-ring': stalk_surface_above_ring_dict[stalk_surface_above_ring],
    'stalk-surface-below-ring': stalk_surface_below_ring_dict[stalk_surface_below_ring],
    'stalk-color-above-ring': stalk_color_above_ring_dict[stalk_color_above_ring],
    'stalk-color-below-ring': stalk_color_below_ring_dict[stalk_color_below_ring],
    'veil-type': veil_type_dict[veil_type],
    'veil-color': veil_color_dict[veil_color],
    'ring-number': ring_number_dict[ring_number],
    'ring-type': ring_type_dict[ring_type],
    'spore-print-color': spore_print_color_dict[spore_print_color],
    'population': population_dict[population],
    'habitat': habitat_dict[habitat]
}

# Convert the dictionary to a DataFrame

encoding_file_path = 'resources/encoded_dict.json'
@st.cache_data
def get_encoding_json_file(encoding_file_path):
    with open(encoding_file_path, 'r') as json_file:
        return json.load(json_file)

encoded_dict = get_encoding_json_file(encoding_file_path)

ready_input_data = dict()
for i, v in input_data.items():
    ready_input_data[i] = int(encoded_dict[v])

df = pd.DataFrame([ready_input_data])
scaled_df = standard_scaler.transform(df)

if button:
    pred = model.predict(scaled_df)
    if pred == 1:
        st.write('Poisonous')
    else:
        st.write('Edible')

