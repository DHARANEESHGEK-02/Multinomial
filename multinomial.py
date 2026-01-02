import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import io

# Page config
st.set_page_config(
    page_title="📱 Spam Detector", 
    page_icon="📱",
    layout="wide"
)

st.title("📱 SMS Spam Detector - Multinomial Naive Bayes")
st.markdown("**Upload CSV with `Category` (ham/spam) & `Message` columns**")

# File uploader
uploaded_file = st.file_uploader(
    "📁 Upload spam dataset (CSV)",
    type=['csv'],
    help="Columns: Category (ham/spam), Message"
)

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    df['Category'] = df['Category'].map({'ham': 0, 'spam': 1}).fillna(df['Category'])
    
    st.success(f"✅ **Dataset loaded:** {len(df):,} messages")
    col1, col2 = st.columns(2)
    col1.metric("✅ Ham", (df['Category'] == 0).sum())
    col2.metric("📱 Spam", (df['Category'] == 1).sum())
    
    # Sample data preview
    st.subheader("📋 Data Preview")
    st.dataframe(df.head())
    
    # Train model
    @st.cache_resource
    def train_model(_df):
        # Preprocess
        df_copy = _df.copy()
        df_copy['Message'] = df_copy['Message'].astype(str).str.lower()
        
        # Vectorize
        vectorizer = CountVectorizer(stop_words='english', max_features=5000)
        X = vectorizer.fit_transform(df_copy['Message'])
        y = df_copy['Category']
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train MultinomialNB
        model = MultinomialNB()
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return model, vectorizer, accuracy
    
    model, vectorizer, accuracy = train_model(df)
    
    # Sidebar prediction
    st.sidebar.header("🔍 Test Message")
    test_message = st.sidebar.text_area(
        "Enter SMS:",
        "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121",
        height=100
    )
    
    if st.sidebar.button("🚀 Predict", type="primary"):
        if test_message:
            X_test_msg = vectorizer.transform([test_message.lower()])
            pred = model.predict(X_test_msg)[0]
            probs = model.predict_proba(X_test_msg)[0]
            
            col1, col2 = st.sidebar.columns(2)
            if pred == 1:
                col1.error(f"**📱 SPAM** ({probs[1]:.1%})")
            else:
                col1.success(f"**✅ HAM** ({probs[0]:.1%})")
            col2.metric("Model Accuracy", f"{accuracy:.1%}")
    
    # Dashboard
    col1, col2, col3 = st.columns(3)
    col1.metric("📨 Total", len(df))
    col2.metric("✅ Ham %", f"{(df['Category'] == 0).mean():.1%}")
    col3.metric("📱 Spam %", f"{(df['Category'] == 1).mean():.1%}")
    
    # Charts
    col1, col2 = st.columns(2)
    with col1:
        pie_data = df['Category'].value_counts().reset_index()
        pie_data['label'] = pie_data['index'].map({0: 'Ham', 1: 'Spam'})
        fig_pie = px.pie(pie_data, values='Category', names='label',
                        title="📊 Distribution", 
                        color_discrete_map={'Ham': '#10B981', 'Spam': '#EF4444'})
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.subheader("📝 Message Lengths")
        df['msg_len'] = df['Message'].astype(str).str.len()
        fig_hist = px.histogram(df, x='msg_len', color='Category',
                              title="Message Length Distribution",
                              color_discrete_map={0: '#10B981', 1: '#EF4444'})
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Metrics
    st.subheader("📈 Model Performance")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("🎯 Test Accuracy", f"{accuracy:.3f}")
        
        # Confusion Matrix
        y_pred_all = model.predict(vectorizer.transform(df['Message'].astype(str)))
        cm = confusion_matrix(df['Category'], y_pred_all)
        fig_cm, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
        plt.title('Confusion Matrix')
        st.pyplot(fig_cm)
    
    with col2:
        report = classification_report(df['Category'], y_pred_all, 
                                     target_names=['Ham', 'Spam'], output_dict=True)
        st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)
    
    # Save model
    if st.button("💾 Download Model", type="secondary"):
        model_data = {'model': model, 'vectorizer': vectorizer}
        joblib.dump(model_data, 'spam_model.joblib')
        with open('spam_model.joblib', 'rb') as f:
            st.download_button(
                label="📥 Download spam_model.joblib",
                data=f,
                file_name='spam_model.joblib',
                mime='application/octet-stream'
            )
    
    st.balloons()

else:
    st.info("👆 **Upload your CSV file** to start (your `spam.csv` works perfectly!)")
    st.markdown("""
    **Expected format:**
    ```
    Category,Message
    ham,Go until jurong point...
    spam,Free entry in 2 a wkly comp...
    ```
    """)
