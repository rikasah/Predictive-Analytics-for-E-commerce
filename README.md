# Product Recommendation App

This Streamlit app provides product recommendations for customers based on a collaborative filtering model (SVD - Singular Value Decomposition). Additionally, it includes visualizations to explore average ratings and prices per category, purchases in each category, and daily purchases.

## Instructions to Run the App

Follow these steps to run the Product Recommendation App locally:

### 1. Clone the Repository

```bash
git clone https://github.com/rikasah/Predictive-Analytics-for-E-commerce.git
cd Predictive-Analytics-for-E-commerce
```

### 2. Install Dependencies

Ensure you have Python installed. Then, create a virtual environment and install the required packages using `pip`:

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: .\venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the Streamlit App

```bash
streamlit run appv2.py
```

This command will launch the app, and you can access it by visiting the provided URL in your web browser.

## Usage

1. Enter a customer ID and the desired number of recommendations.
2. Click the "Get Recommendations" button to see top product recommendations.
3. Use the sidebar to choose visualizations such as average rating per category, average price per category, purchases in each category, or daily purchases.
4. Leave feedback in the sidebar for the app.

## Feedback

Feel free to leave your feedback using the provided textarea in the sidebar. Your input is valuable for improving the app.
