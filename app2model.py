import streamlit as st
import pandas as pd
from surprise.dump import load
from surprise.model_selection import train_test_split
from surprise import SVD, KNNBasic
from surprise import Dataset, Reader
import requests
from joblib import dump, load
from surprise.accuracy import rmse
from surprise.model_selection import cross_validate

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

# Function to download and load the KNN model
def load_knn_model(url):
    response = requests.get(url)
    if response.status_code == 200:
        with open("knn_model.pkl", "wb") as f:
            f.write(response.content)
        return load("knn_model.pkl")
    else:
        return None

# Load the SVD model from the file
svd_model_url = 'https://github.com/rikasah/Predictive-Analytics-for-E-commerce/raw/main/svd_model.joblib'
loaded_svd_model = load_svd_model(svd_model_url)

# Load the KNN model from the file
knn_model_url = 'https://github.com/rikasah/Predictive-Analytics-for-E-commerce/raw/main/knn_model.pkl'
loaded_knn_model = load_knn_model(knn_model_url)

# Check if the models were loaded successfully
if loaded_svd_model is not None:
    st.success("SVD Model loaded successfully")
    # Verify the loaded model
    st.write(loaded_svd_model)
else:
    st.error("Failed to load the SVD model. Please check the URL.")

if loaded_knn_model is not None:
    st.success("KNN Model loaded successfully")
    # Verify the loaded model
    st.write(loaded_knn_model)
else:
    st.error("Failed to load the KNN model. Please check the URL.")

# Function to get top N recommendations from a model
def get_top_n_recommendations(model, user_id, n=5):
    # Check if the user ID is in the dataset
    if user_id not in df['customer_id'].unique():
        st.warning(f"No data available for customer with ID {user_id}. Please check if the customer has provided ratings.")
        return []

    # Get a list of all product IDs
    all_products = df['product_id'].unique()

    # Get the products the user has already purchased
    purchased_products = df[df['customer_id'] == user_id]['product_id'].tolist()

    # Remove the purchased products from the list
    to_predict = [prod for prod in all_products if prod not in purchased_products]

    # Check if there are items to predict
    if not to_predict:
        st.warning(f"No products available for recommendation for customer with ID {user_id}.")
        return []

    # Predict ratings for products that the user has not purchased
    predictions = [model.predict(user_id, prod) for prod in to_predict]

    # Sort predictions by estimated rating
    predictions.sort(key=lambda x: x.est, reverse=True)

    # Display the top N recommended products
    top_n_recommendations = predictions[:n]
    if not top_n_recommendations:
        st.warning(f"No recommendations available for customer with ID {user_id}. Please check if the customer has provided ratings.")
        return []
    else:
        recommended_products = [(recommendation.iid, recommendation.est) for recommendation in top_n_recommendations]
        return recommended_products

# Streamlit app
def main():
    st.title("Product Recommendation App")

    # User input for customer ID
    user_id_input = st.number_input("Enter customer ID:", min_value=int(df['customer_id'].min()), max_value=int(df['customer_id'].max()))

    # User input for top N recommendations
    top_n_input = st.number_input("Enter the number of recommendations (top N):", min_value=1, value=5)

    # Get recommendations when the user clicks the "Get Recommendations" button
    if st.button("Get Recommendations"):
        # Recommendations from SVD model
        recommendations_svd = get_top_n_recommendations(loaded_svd_model, user_id_input, n=top_n_input)
        
        # Recommendations from KNN model
        recommendations_knn = get_top_n_recommendations(loaded_knn_model, user_id_input, n=top_n_input)

        # Display the recommendations
        if recommendations_svd:
            st.success(f"Top {top_n_input} SVD recommendations for customer with ID {user_id_input}:")
            for product_id, estimated_rating in recommendations_svd:
                st.write(f"Product ID: {product_id}, Estimated Rating: {estimated_rating}")

        if recommendations_knn:
            st.success(f"Top {top_n_input} KNN recommendations for customer with ID {user_id_input}:")
            for product_id, estimated_rating in recommendations_knn:
                st.write(f"Product ID: {product_id}, Estimated Rating: {estimated_rating}")

    # Evaluation matrix button
    if st.button("Show Evaluation Matrix"):
        # Evaluate the SVD model
        svd_data = Dataset.load_from_df(df[['customer_id', 'product_id', 'rating']], Reader())
        svd_trainset = svd_data.build_full_trainset()
        svd_eval = cross_validate(loaded_svd_model, svd_data, measures=['RMSE'], cv=5, verbose=True)
        st.write("SVD Model Evaluation Matrix:")
        st.write(svd_eval)

        # Evaluate the KNN model
        knn_data = Dataset.load_from_df(df[['customer_id', 'product_id', 'rating']], Reader())
        knn_trainset = knn_data.build_full_trainset()
        knn_eval = cross_validate(loaded_knn_model, knn_data, measures=['RMSE'], cv=5, verbose=True)
        st.write("KNN Model Evaluation Matrix:")
        st.write(knn_eval)

if __name__ == '__main__':
    main()