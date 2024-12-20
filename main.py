import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
from helper_functions import label_to_character_mapping, colors, predict_new_feature_vector, visualize_feature_extraction, visualize_data_distribution
from feature_extractors import FeatureExtractor
from movement_simulators import ExecuteAllPourMovements

st.set_page_config(layout="wide", page_icon='logo.png')
st.title("Hindi MNIST Classification using Water Accumulation method")

st.markdown("""#### This project is a simple implementation of a machine learning model for classifying handwritten digits in the Hindi language using the Water Accumulation method. The machine learning model is trained on the features extracted from the input images using the Water Accumulation method. This model is used to predict the correct label for a given input image. This project was inspired by this [video](https://youtu.be/CC4G_xKK2g8?si=MbyYaG73GaxDrL3X) from [PickentCode](https://www.youtube.com/@PickentCode). The code in reference is available [here](https://github.com/PickentCode/KNN-Digit-Recognition). 
            
#### I used Python with Streamlit instead of JavaScript to create this project. Other differences being that I used [Hindi MNIST](https://www.kaggle.com/datasets/imbikramsaha/hindi-mnist/data) dataset instead of MNIST dataset and I used other classifiers instead of KNN.
""")

st.header("Sample Characters (0-9)")
st.image("all_digits_img.jpeg", caption="Sample Characters", use_container_width=True)

st.header("Draw a character")
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=30,
    stroke_color="White",
    background_color="black",
    width=400,
    height=400,
    drawing_mode="freedraw",
    key="canvas"
)

display_traversal = st.checkbox("Display Input Data Feature Extraction Graphs", value=True)

new_feature_vector = None

if canvas_result.image_data is not None:
    new_image = Image.fromarray(canvas_result.image_data)
    new_image = new_image.resize((32, 32)).convert("L")
    new_image = np.asarray(new_image)
    new_image = (new_image > 128)
    if new_image.sum() != 0:
        st.header("Predicted Character")
        prediction_row = st.columns(2)
        prediction_row[1].image(new_image.astype(np.uint8) * 255, caption="Input image given to the model!")
        exec_pour_movements = ExecuteAllPourMovements(display_traversal)
        pour_movements = exec_pour_movements.get_objects(new_image)
        feature_extractor = FeatureExtractor()
        new_feature_vector = feature_extractor.extract_features(new_image, *pour_movements)
        input_prediction = predict_new_feature_vector(new_feature_vector)
        prediction_row[0].markdown(f"### The predicted character is {label_to_character_mapping[input_prediction]} in Hindi which is equivalent to {input_prediction} in Arabic number system.")

st.header("Data Distribution Graph")
dist_options = [st.columns(2) for _ in range(2)]
labels_to_include = dist_options[0][0].multiselect("Select Labels", list(range(10)), list(range(10)))
data_marker_size = dist_options[0][1].slider("Dataset Marker Size", 0, 500, 200, 50)
input_marker_size = dist_options[1][0].slider("Input Marker Size", 200, 2000, 1000, 50)
num_samples = dist_options[1][1].slider("Number of Samples to Visualize", 100, 3000, 500, 100)
fig, scatter_lst, new_scatter = visualize_data_distribution(labels_to_include, new_feature_vector, n_samples=num_samples)
for scatter in scatter_lst:
    scatter.set_sizes(np.ones(len(scatter.get_offsets())) * data_marker_size)
if new_scatter is not None:
    new_scatter.set_sizes(np.ones(len(new_scatter.get_offsets())) * input_marker_size)
    new_scatter.set_color(colors[input_prediction])
    new_scatter.set_edgecolors("black")
st.pyplot(fig, pad_inches=0)


if new_feature_vector is not None and display_traversal:
    st.header("Input Data Feature Extraction Graphs")
    visualize_feature_extraction(*pour_movements)

st.header("About me")
st.markdown("This project was made by [**Sidharth D**](https://github.com/Sidharth-Darwin).")
