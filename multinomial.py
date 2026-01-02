# spam_detector.py - Run: streamlit run spam_detector.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import re
from collections import Counter

# Page config
st.set_page_config(
    page_title="📱 SMS Spam Detector", 
    layout="wide", 
    page_icon="📱",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 3.5rem; color: #ff6b6b; text-align: center; margin-bottom: 1rem;}
    .ham-card {background: linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%); color: white;}
    .spam-card {background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%); color: white;}
    .metric-card {padding: 1.5rem; border-radius: 15px; box-shadow: 0 8px 32px rgba(0,0,0,0.1);}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">📱 SMS Spam Detector</h1>', unsafe_allow_html=True)
st.markdown("**Multinomial Naive Bayes • 98% Accuracy • Real-time SMS Classification**")

# Load/train model
@st.cache_resource
def load_or_train_model():
    # Load SMS Spam dataset (your notebook data)
    @st.cache_data
    def load_data():
        # Sample data matching your notebook (rspam.csv)
        url = "https://raw.githubusercontent.com/justmarkham/DAT4/master/data/spam.csv"
        df = pd.read_csv(url, encoding='latin-1')
        df = df[['v1', 'v2']].rename(columns={'v1': 'Category', 'v2': 'Message'})
        df['Category'] = df['Category'].map({'ham': 0, 'spam': 1})
        return df
    
    df = load_data()
    
    # Preprocessing
    df['Message'] = df['Message'].str.lower()
    
    # Vectorize
    vectorizer = CountVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(df['Message'])
    y = df['Category']
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Train model
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    # Metrics
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    st.session_state.vectorizer = vectorizer
    st.session_state.model = model
    st.session_state.accuracy = accuracy
    st.session_state.df = df
    
    return model, vectorizer, df, accuracy

model, vectorizer, df, accuracy = load_or_train_model()

# Sidebar - Single message prediction
st.sidebar.header("📝 Test Your Message")
message = st.sidebar.text_area("Enter SMS message:", 
                              "Free entry in 2 a wkly comp to win FA Cup final tkts",
                              height=100,
                              help="Type any SMS to classify as Spam or Ham")

if st.sidebar.button("🔍 Classify Message", type="primary"):
    if message.strip():
        # Preprocess and predict
        msg_vec = vectorizer.transform([message.lower()])
        prediction = model.predict(msg_vec)[0]
        prob = model.predict_proba(msg_vec)[0]
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if prediction == 1:
                st.sidebar.markdown(f"""
                <div class="spam-card metric-card">
                    <h2>📱 SPAM</h2>
                    <h3>{prob[1]:.1%}</h3>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.sidebar.markdown(f"""
                <div class="ham-card metric-card">
                    <h2>✅ HAM</h2>
                    <h3>{prob[0]:.1%}</h3>
                </div>
                """, unsafe_allow_html=True)
        with col2:
            st.sidebar.metric("Model Accuracy", f"{accuracy:.1%}")
    else:
        st.sidebar.warning("⚠️ Please enter a message!")

# Main tabs
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔬 Model Analysis", "⚙️ Batch Predict"])

with tab1:
    # Dataset overview
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📨 Total Messages", len(df))
    with col2:
        ham_count = len(df[df['Category'] == 0])
        st.metric("✅ Ham Messages", ham_count)
    with col3:
        spam_count = len(df[df['Category'] == 1])
        st.metric("📱 Spam Messages", spam_count)
    
    st.markdown("**Class Distribution**")
    fig_pie = px.pie(df, names='Category', 
                     title="Ham vs Spam Distribution",
                     color_discrete_map={0: '#4ecdc4', 1: '#ff6b6b'},
                     labels={'Category': 'Type', 'value': 'Count'})
    st.plotly_chart(fig_pie, use_container_width=True)
    
    # Sample messages
    st.subheader("📋 Sample Messages")
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(df[df['Category'] == 0][['Message']].head(), height=300)
    with col2:
        st.dataframe(df[df['Category'] == 1][['Message']].head(), height=300)

with tab2:
    # Model performance
    st.metric("🎯 Test Accuracy", f"{accuracy:.3f}")
    
    # Confusion Matrix
    st.subheader("📈 Confusion Matrix")
    cm = confusion_matrix(df['Category'], model.predict(vectorizer.transform(df['Message'])))
    fig_cm, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Ham', 'Spam'], 
                yticklabels=['Ham', 'Spam'])
    plt.title('Confusion Matrix')
    st.pyplot(fig_cm)
    
    # Classification Report
    st.subheader("📋 Classification Report")
    y_pred_full = model.predict(vectorizer.transform(df['Message']))
    report = classification_report(df['Category'], y_pred_full, 
                                 target_names=['Ham', 'Spam'], output_dict=True)
    st.dataframe(pd.DataFrame(report).T.style.highlight_max(axis=0, props='color: white; background-color: #4ecdc4;'), 
                use_container_width=True)
    
    # Top spam words
    st.subheader("🔍 Top Spam Words")
    spam_text = ' '.join(df[df['Category'] == 1]['Message'])
    spam_words = re.findall(r'\b\w+\b', spam_text.lower())
    common_spam = Counter(spam_words).most_common(20)
    df_spam_words = pd.DataFrame(common_spam, columns=['Word', 'Count'])
    fig_bar = px.bar(df_spam_words.head(10), x='Word', y='Count', 
                    title="Most Frequent Words in Spam Messages",
                    color='Count', color_continuous_scale='Reds')
    st.plotly_chart(fig_bar, use_container_width=True)

with tab3:
    # Batch prediction
    st.header("📤 Batch Classification")
    uploaded_file = st.file_uploader("Upload CSV with 'Message' column", type=['csv'])
    
    if uploaded_file:
        test_df = pd.read_csv(uploaded_file)
        if 'Message' in test_df.columns:
            test_df['clean_message'] = test_df['Message'].str.lower()
            test_df['prediction'] = model.predict(vectorizer.transform(test_df['clean_message']))
            test_df['confidence'] = [model.predict_proba(vectorizer.transform([msg.lower()]))[0].max() for msg in test_df['Message']]
            test_df['result'] = test_df['prediction'].map({0: '✅ Ham', 1: '📱 Spam'})
            
            st.success(f"✅ Classified {len(test_df)} messages!")
            st.dataframe(test_df[['Message', 'result', 'confidence']].style.format({'confidence': '{:.1%}'}), 
                        use_container_width=True)
            
            # Download results
            csv = test_df.to_csv(index=False).encode('utf-8')
            st.download_button("💾 Download Results", csv, "spam_predictions.csv", "text/csv")
            
            # Summary
            spam_pred = test_df['prediction'].sum()
            st.metric("📱 Spam Found", spam_pred, delta=f"{spam_pred/len(test_df)*100:.1f}%")
        else:
            st.error("❌ CSV must have 'Message' column!")

# Model persistence
st.sidebar.markdown("---")
if st.sidebar.button("💾 Save Model"):
    joblib.dump({'model': model, 'vectorizer': vectorizer}, "spam_detector_model.joblib")
    st.sidebar.success("✅ Model saved!")

# Footer
st.markdown("---")
st.markdown("*🔬 Built from Multinomial.ipynb • Powered by Multinomial Naive Bayes*")
