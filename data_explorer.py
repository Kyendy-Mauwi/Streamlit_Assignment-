import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def import_csv_from_url(url):
    try:
        df = pd.read_csv(url, header=None, names=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"])
        return df
    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    # Replace the URL with the Iris dataset CSV URL
    csv_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

    # Call the function to import the CSV from the URL
    data_frame = import_csv_from_url(csv_url)

    # Check if the data_frame is not None (i.e., the import was successful)
    if data_frame is not None:
        st.title("Iris Dataset")

        # Question 1: Show the average sepal length for each species
        avg_sepal_length = data_frame.groupby('class')['sepal_length'].mean()
        st.header("Question 1: Average Sepal Length for Each Species")
        st.write(avg_sepal_length)

        # Question 2: Display a scatter plot comparing two features
        st.header("Question 2: Scatter Plot Comparing Two Features")
        feature1 = st.selectbox("Select the first feature", data_frame.columns[:-1])
        feature2 = st.selectbox("Select the second feature", data_frame.columns[:-1])
        sns.scatterplot(data=data_frame, x=feature1, y=feature2, hue='class')
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        st.pyplot()

        # Question 3: Filter data based on species
        st.header("Question 3: Filter Data Based on Species")
        selected_species = st.selectbox("Select species to filter", data_frame['class'].unique())
        filtered_data = data_frame[data_frame['class'] == selected_species]
        st.write(filtered_data)

        # Question 4: Display a pairplot for the selected species
        st.header("Question 4: Pairplot for the Selected Species")
        sns.pairplot(filtered_data, hue='class')
        st.pyplot()

        # Question 5: Show the distribution of a selected feature
        st.header("Question 5: Distribution of a Selected Feature")
        selected_feature = st.selectbox("Select feature for distribution", data_frame.columns[:-1])
        sns.histplot(data=data_frame, x=selected_feature, hue='class', kde=True)
        plt.xlabel(selected_feature)
        st.pyplot()

if __name__ == "__main__":
    main()
