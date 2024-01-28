import streamlit as st
import pandas as pd
from surprise.dump import load
from surprise.model_selection import train_test_split
from surprise import SVD
from surprise import Dataset, Reader
import requests
from joblib import dump, load
import matplotlib.pyplot as plt

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
    # Verify the loaded model
    st.write(loaded_svd_model)
else:
    st.error("Failed to load the SVD model. Please check the URL.")

# Streamlit app
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

    # Visualizations
    if st.checkbox("Show Visualizations"):
        # Choose which visualization to display
        visualization_option = st.radio("Select Visualization:", ["Daily Purchases Line Chart", "Average Price per Category Bar Chart"])

        # Check if 'category' column exists in the DataFrame
        if 'category' in df.columns:
            # Check if 'ratings' column exists in the DataFrame
            if 'ratings' in df.columns:
                # Average rating per category
                avg_rating_per_category = df.groupby('category')['ratings'].mean().reset_index()
                st.bar_chart(avg_rating_per_category.set_index('category'))
            else:
                st.warning("The 'ratings' column does not exist in the dataset.")

            # Check if 'price' column exists in the DataFrame
            if 'price' in df.columns:
                # Average price per category
                avg_price_per_category = df.groupby('category')['price'].mean().reset_index()
                st.bar_chart(avg_price_per_category.set_index('category'))
            else:
                st.warning("The 'price' column does not exist in the dataset.")

            # Most buying category item
            most_buying_category = df['category'].mode().iloc[0]
            st.write("Most buying category item:", most_buying_category)
        else:
            st.warning("The 'category' column does not exist in the dataset.")

        # Display line chart for daily purchases
        if visualization_option == "Daily Purchases Line Chart":
            df['purchase_date'] = pd.to_datetime(df['purchase_date'])
            daily_purchases = df.groupby(df['purchase_date'].dt.date)['product_id'].count().reset_index()
            daily_purchases.columns = ['Date', 'Daily Purchases']
            st.line_chart(daily_purchases.set_index('Date'))

        # Display average price per category bar chart
        elif visualization_option == "Average Price per Category Bar Chart":
            if 'price' in df.columns:
                avg_price_per_category = df.groupby('category')['price'].mean().reset_index()
                st.bar_chart(avg_price_per_category.set_index('category'))
            else:
                st.warning("The 'price' column does not exist in the dataset.")

if __name__ == '__main__':
    main()
