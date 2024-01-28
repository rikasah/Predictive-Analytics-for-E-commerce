import streamlit as st
import pandas as pd
from surprise.dump import load
from surprise.model_selection import train_test_split
from surprise import SVD
from surprise import Dataset, Reader
import requests
import pickle

# Display the version of the app
st.write("App Version: 1.0.0")

# Load the dataset
df = pd.read_csv('https://github.com/rikasah/Predictive-Analytics-for-E-commerce/raw/main/fake_data.csv')

# Load the SVD model from the URL
svd_model_url = 'https://github.com/rikasah/Predictive-Analytics-for-E-commerce/raw/main/svd_model.pkl'
response = requests.get(svd_model_url)
loaded_svd_model_content = pickle.loads(response.content)

# Print the loaded content to understand its structure
st.write("Loaded SVD Model Content:", loaded_svd_model_content)

# Assuming the content is a tuple, get the second element
loaded_svd_model = loaded_svd_model_content[1]

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
        recommendations = get_top_n_recommendations(loaded_svd_model, user_id_input, n=top_n_input)
        
        # Display the recommendations
        if recommendations:
            st.success(f"Top {top_n_input} recommended products for customer with ID {user_id_input}:")
            for product_id, estimated_rating in recommendations:
                st.write(f"Product ID: {product_id}, Estimated Rating: {estimated_rating}")

if __name__ == '__main__':
    main()
