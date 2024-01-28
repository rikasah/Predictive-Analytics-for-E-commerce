import streamlit as st
import pandas as pd
import requests
from joblib import load
import matplotlib.pyplot as plt
import seaborn as sns

# Set the page title and favicon
st.set_page_config(page_title="E-commerce Recommender App", page_icon="🛍️")

# Display the version of the app
st.write("App Version: 1.0.0")

# Load the dataset
df = pd.read_csv('https://github.com/rikasah/Predictive-Analytics-for-E-commerce/raw/main/fake_data.csv')

# Function to download and load the SVD model
def load_svd_model(url):
    response = requests.get(url)
    if response.status_code == 200:
        with open("svd_model.joblib", "wb") as f:
            f.write(response.content)
        return load("svd_model.joblib")
    else:
        return None

# Load the SVD model from the file
svd_model_url = 'https://github.com/rikasah/Predictive-Analytics-for-E-commerce/raw/main/svd_model.joblib'
loaded_svd_model = load_svd_model(svd_model_url)

# Check if the model was loaded successfully
if loaded_svd_model is not None:
    st.success("SVD Model loaded successfully")
else:
    st.error("Failed to load the SVD model. Please check the URL.")

def get_top_n_recommendations(model, user_id, n=5):
    # Function to get recommendations (unchanged)

def main():
    st.title("Product Recommendation App")

    # User input for customer ID
    user_id_input = st.number_input("Enter customer ID:", min_value=int(df['customer_id'].min()), max_value=int(df['customer_id'].max()))

    # User input for top N recommendations
    top_n_input = st.number_input("Enter the number of recommendations (top N):", min_value=1, value=5)

    # Get recommendations when the user clicks the "Get Recommendations" button
    if st.button("Get Recommendations"):
        recommendations = get_top_n_recommendations(loaded_svd_model, user_id_input, n=top_n_input)
        
        # Display the recommendations
        if recommendations:
            st.success(f"Top {top_n_input} recommended products for customer with ID {user_id_input}:")
            for product_id, estimated_rating in recommendations:
                st.write(f"Product ID: {product_id}, Estimated Rating: {estimated_rating}")

    # Visualization for Average Ratings per Category
    st.subheader("Average Ratings per Category")
    avg_ratings_per_category = df.groupby('category')['rating'].mean()
    st.bar_chart(avg_ratings_per_category)

    # Visualization for Average Prices per Category
    st.subheader("Average Prices per Category")
    avg_prices_per_category = df.groupby('category')['price'].mean()
    st.bar_chart(avg_prices_per_category)

    # Visualization for Most Bought Category Item
    st.subheader("Most Bought Category Item")
    most_bought_category_item = df['category'].value_counts().idxmax()
    st.write(f"The most bought category item is: {most_bought_category_item}")

if __name__ == '__main__':
    main()
