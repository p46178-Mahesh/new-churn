import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Configure page
st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    page_icon="🔮",
    layout="wide"
)

st.title("🔮 Customer Churn Prediction (Leakage-Free)")
st.write("This dashboard uses a clean Machine Learning model with all data leakage features (like Recency, RFM scores, Risk Levels, and Customer Segments) completely removed for real-world reliability.")

# Load pipeline
@st.cache_resource
def load_pipeline():
    return joblib.load("clean_customer_churn_model.pkl")

try:
    pipeline = load_pipeline()
    st.sidebar.success("Pipeline loaded successfully!")
except Exception as e:
    st.sidebar.error(f"Error loading pipeline: {e}")
    st.stop()

# Form for inputs
st.subheader("Enter Customer Details")

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Demographics & Loyalty")
        age = st.number_input("Customer Age", min_value=12, max_value=110, value=35)
        gender = st.selectbox("Gender", ["Female", "Male"])
        country = st.selectbox("Country", ["United States", "Canada", "United Kingdom", "Germany", "France", "China", "India", "Brazil"])
        continent = st.selectbox("Continent", ["North America", "Europe", "Asia", "South America", "Africa", "Oceania"])
        loyalty_tier = st.selectbox("Loyalty Tier", ["Bronze", "Silver", "Gold", "Platinum"])
        loyalty_points = st.number_input("Loyalty Points", min_value=0, value=150)
        clv = st.number_input("Customer Lifetime Value (USD)", min_value=0.0, value=250.0)

    with col2:
        st.markdown("### Session Activity")
        total_sessions = st.number_input("Total Sessions", min_value=0, value=25)
        total_page_views = st.number_input("Total Page Views", min_value=0, value=120)
        avg_session_dur = st.number_input("Avg Session Duration (Mins)", min_value=0.0, value=12.5)
        products_viewed = st.number_input("Products Viewed", min_value=0, value=45)
        products_added = st.number_input("Products Added To Cart", min_value=0, value=10)
        cart_abandonments = st.number_input("Cart Abandonments", min_value=0, value=3)
        purchase_freq = st.selectbox("Purchase Frequency", ["Very Low", "Low", "Medium", "High", "Very High"])
        sat_score = st.slider("Satisfaction Score (1-5)", 1.0, 5.0, 4.0)

    with col3:
        st.markdown("### Orders & Spending")
        total_orders = st.number_input("Total Orders", min_value=0, value=8)
        completed_orders = st.number_input("Completed Orders", min_value=0, value=6)
        cancelled_orders = st.number_input("Cancelled Orders", min_value=0, value=1)
        returned_orders = st.number_input("Returned Orders", min_value=0, value=1)
        total_items = st.number_input("Total Items Purchased", min_value=0, value=15)
        total_spending = st.number_input("Total Spending (USD)", min_value=0.0, value=450.0)
        avg_order_value = st.number_input("Average Order Value (USD)", min_value=0.0, value=75.0)

    st.markdown("### Behavioral Details & Engagement Preferences")
    b1, b2, b3 = st.columns(3)
    with b1:
        discount_count = st.number_input("Discount Usage Count", min_value=0, value=2)
        discount_pct = st.slider("Average Discount %", 0.0, 100.0, 15.0)
        coupon_count = st.number_input("Coupon Usage Count", min_value=0, value=1)
        wishlist_items = st.number_input("Wishlist Items", min_value=0, value=5)
    with b2:
        product_reviews = st.number_input("Product Reviews Given", min_value=0, value=2)
        avg_rating_given = st.slider("Average Product Rating Given", 1.0, 5.0, 4.2)
        preferred_category = st.selectbox("Preferred Category", ["Electronics", "Clothing", "Home & Kitchen", "Beauty", "Sports"])
        preferred_payment = st.selectbox("Preferred Payment Method", ["Credit Card", "PayPal", "Debit Card", "Bank Transfer", "Cash"])
    with b3:
        support_contacts = st.number_input("Customer Support Contacts", min_value=0, value=2)
        complaint_count = st.number_input("Complaint Count", min_value=0, value=0)
        preferred_device = st.selectbox("Preferred Device", ["Mobile", "Desktop", "Tablet"])
        preferred_marketing = st.selectbox("Preferred Marketing Channel", ["Email Marketing", "Social Media", "SEO", "PPC"])

    # Extended fields expected by pipeline
    email_open = st.slider("Email Open Rate (%)", 0.0, 100.0, 40.0)
    email_click = st.slider("Email Click Rate (%)", 0.0, 100.0, 10.0)
    push_engagement = st.slider("Push Notification Engagement (%)", 0.0, 100.0, 25.0)
    social_engagement = st.slider("Social Media Engagement Score (1-10)", 1, 10, 5)
    city = "New York"
    region = "New York"

    submit = st.form_submit_button("Predict Churn Risk")

if submit:
    # Construct the raw user input row matches exactly X_clean columns format
    user_input = {
        'Customer_Age': age,
        'Gender': gender,
        'Country': country,
        'Continent': continent,
        'City': city,
        'Region': region,
        'Total_Sessions': total_sessions,
        'Total_Page_Views': total_page_views,
        'Average_Session_Duration_Minutes': avg_session_dur,
        'Products_Viewed': products_viewed,
        'Products_Added_To_Cart': products_added,
        'Cart_Abandonment_Count': cart_abandonments,
        'Total_Orders': total_orders,
        'Completed_Orders': completed_orders,
        'Cancelled_Orders': cancelled_orders,
        'Returned_Orders': returned_orders,
        'Total_Items_Purchased': total_items,
        'Total_Spending_USD': total_spending,
        'Average_Order_Value_USD': avg_order_value,
        'Discount_Usage_Count': discount_count,
        'Average_Discount_Percentage': discount_pct,
        'Coupon_Usage_Count': coupon_count,
        'Wishlist_Items': wishlist_items,
        'Product_Reviews_Count': product_reviews,
        'Average_Product_Rating': avg_rating_given,
        'Customer_Support_Contacts': support_contacts,
        'Complaint_Count': complaint_count,
        'Preferred_Category': preferred_category,
        'Preferred_Payment_Method': preferred_payment,
        'Preferred_Device': preferred_device,
        'Preferred_Marketing_Channel': preferred_marketing,
        'Email_Open_Rate_Percentage': email_open,
        'Email_Click_Rate_Percentage': email_click,
        'Push_Notification_Engagement_Percentage': push_engagement,
        'Social_Media_Engagement_Score': social_engagement,
        'Loyalty_Points': loyalty_points,
        'Loyalty_Tier': loyalty_tier,
        'Customer_Lifetime_Value_USD': clv,
        'Purchase_Frequency': purchase_freq,
        'Customer_Satisfaction_Score': sat_score
    }

    input_df = pd.DataFrame([user_input])
    
    # Predict
    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0][1]

    st.markdown("--- ")
    st.subheader("Prediction Results")
    if prediction == 1:
        st.error(f"🚨 **High Churn Risk!** Probability: **{probability:.2%}**")
    else:
        st.success(f"✅ **Loyal / Active Customer** Churn Probability: **{probability:.2%}**")
