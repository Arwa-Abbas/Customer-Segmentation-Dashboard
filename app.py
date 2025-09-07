import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Customer Segmentation Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Root Variables */
    :root {
        --primary-color: #ff6b35;
        --secondary-color: #f7931e;
        --accent-color: #4ecdc4;
        --bg-primary: #0f0f23;
        --bg-secondary: #1a1a2e;
        --bg-tertiary: #16213e;
        --text-primary: #ffffff;
        --text-secondary: #b8c5d6;
        --border-color: #2d3748;
        --shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
        --shadow-hover: 0 8px 30px rgba(255, 107, 53, 0.3);
        --glow: 0 0 20px rgba(255, 107, 53, 0.3);
    }

    /* Main App Container */
    .stApp {
        background: linear-gradient(135deg, var(--bg-primary) 0%, #0a0a1f 50%, var(--bg-secondary) 100%);
        color: var(--text-primary);
        font-family: 'Inter', sans-serif;
    }

    /* Fixed Sidebar Styling */
    .css-1d391kg, .css-1y4p8pa, .css-sidebar {
        background: linear-gradient(180deg, var(--bg-secondary) 0%, var(--bg-primary) 100%) !important;
        border-right: 2px solid var(--border-color) !important;
        box-shadow: 2px 0 10px rgba(0,0,0,0.3) !important;
    }
    
    .sidebar .sidebar-content {
        background: transparent !important;
    }
    
    /* Ensure sidebar visibility */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--bg-secondary) 0%, var(--bg-primary) 100%) !important;
        border-right: 2px solid var(--border-color) !important;
    }
    
    /* Custom Header */
    .dashboard-header {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color), var(--accent-color));
        padding: 30px;
        border-radius: 20px;
        margin-bottom: 30px;
        text-align: center;
        box-shadow: var(--shadow), var(--glow);
        animation: slideInDown 0.8s ease-out;
        position: relative;
        overflow: hidden;
    }
    
    .dashboard-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
        animation: shine 3s infinite;
    }
    
    .dashboard-title {
        font-size: 3rem;
        font-weight: 800;
        color: white;
        margin: 0;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.5);
        position: relative;
        z-index: 1;
    }
    
    .dashboard-subtitle {
        font-size: 1.2rem;
        color: rgba(255,255,255,0.9);
        margin: 10px 0 0 0;
        font-weight: 500;
        position: relative;
        z-index: 1;
    }

    /* Enhanced Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-tertiary) 100%);
        padding: 25px;
        border-radius: 15px;
        border: 1px solid var(--border-color);
        box-shadow: var(--shadow);
        text-align: center;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        margin-bottom: 20px;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 107, 53, 0.1), transparent);
        transition: left 0.5s;
    }
    
    .metric-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: var(--shadow-hover);
        border-color: var(--primary-color);
        box-shadow: var(--shadow), var(--glow);
    }
    
    .metric-card:hover::before {
        left: 100%;
    }

    /* Chart Container Styling */
    .chart-container {
        background: linear-gradient(135deg, var(--bg-secondary), var(--bg-tertiary));
        padding: 30px;
        border-radius: 20px;
        border: 1px solid var(--border-color);
        box-shadow: var(--shadow);
        margin-bottom: 25px;
        transition: all 0.3s ease;
        animation: fadeInUp 0.6s ease-out;
        position: relative;
    }
    
    .chart-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color), var(--accent-color));
        border-radius: 20px 20px 0 0;
    }
    
    .chart-container:hover {
        transform: translateY(-5px);
        box-shadow: var(--shadow-hover);
        border-color: var(--primary-color);
    }
    
    .chart-title {
        font-size: 1.4rem;
        font-weight: 700;
        background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 25px;
        text-align: center;
    }

    /* Streamlit Native Elements Override */
    .stMetric {
        background: none !important;
        border: none !important;
        box-shadow: none !important;
    }
    
    .stMetric > div {
        background: linear-gradient(135deg, var(--bg-secondary), var(--bg-tertiary)) !important;
        padding: 25px !important;
        border-radius: 15px !important;
        border: 1px solid var(--border-color) !important;
        box-shadow: var(--shadow) !important;
        transition: all 0.3s ease !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    .stMetric > div::before {
        content: '' !important;
        position: absolute !important;
        top: 0 !important;
        left: 0 !important;
        right: 0 !important;
        height: 3px !important;
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color)) !important;
    }
    
    .stMetric > div:hover {
        transform: translateY(-5px) scale(1.02) !important;
        box-shadow: var(--shadow-hover) !important;
        border-color: var(--primary-color) !important;
    }
    
    .stMetric label {
        color: var(--text-secondary) !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }
    
    .stMetric [data-testid="metric-value"] {
        color: var(--primary-color) !important;
        font-weight: 800 !important;
        font-size: 2.2rem !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3) !important;
    }
    
    .stMetric [data-testid="metric-delta"] {
        color: var(--accent-color) !important;
        font-weight: 500 !important;
    }

    /* Enhanced Sidebar */
    .sidebar-header {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: var(--shadow);
        position: relative;
        overflow: hidden;
    }
    
    .sidebar-header::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--accent-color), transparent);
    }
    
    /* Enhanced Buttons */
    .stButton > button {
        background: linear-gradient(45deg, var(--primary-color), var(--secondary-color)) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 15px 30px !important;
        font-weight: 700 !important;
        transition: all 0.3s ease !important;
        box-shadow: var(--shadow) !important;
        font-family: 'Inter', sans-serif !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    .stButton > button::before {
        content: '' !important;
        position: absolute !important;
        top: 0 !important;
        left: -100% !important;
        width: 100% !important;
        height: 100% !important;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent) !important;
        transition: left 0.5s !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.05) !important;
        box-shadow: var(--shadow-hover) !important;
        background: linear-gradient(45deg, var(--secondary-color), var(--accent-color)) !important;
    }
    
    .stButton > button:hover::before {
        left: 100% !important;
    }
    
    /* Enhanced Sliders */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color)) !important;
        border-radius: 10px !important;
    }
    
    .stSlider > div > div > div {
        background: var(--bg-tertiary) !important;
        border-radius: 10px !important;
    }
    
    /* Enhanced Selectbox */
    .stSelectbox > div > div {
        background: var(--bg-tertiary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 10px !important;
        color: var(--text-primary) !important;
    }
    
    /* Enhanced Multiselect */
    .stMultiSelect > div > div {
        background: var(--bg-tertiary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 10px !important;
    }
    
    /* Enhanced Dataframe */
    .stDataFrame {
        background: var(--bg-secondary) !important;
        border-radius: 15px !important;
        padding: 20px !important;
        border: 1px solid var(--border-color) !important;
        box-shadow: var(--shadow) !important;
    }

    /* Enhanced Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--bg-secondary) !important;
        border-radius: 15px !important;
        padding: 8px !important;
        gap: 5px !important;
        box-shadow: var(--shadow) !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        border-radius: 10px !important;
        color: var(--text-secondary) !important;
        transition: all 0.3s ease !important;
        padding: 15px 25px !important;
        font-weight: 600 !important;
        border: 1px solid transparent !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 107, 53, 0.1) !important;
        border-color: var(--primary-color) !important;
        transform: translateY(-2px) !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(45deg, var(--primary-color), var(--secondary-color)) !important;
        color: white !important;
        box-shadow: var(--shadow) !important;
        border-color: var(--primary-color) !important;
    }
    
    /* Enhanced Expander */
    .streamlit-expanderHeader {
        background: var(--bg-secondary) !important;
        border-radius: 10px !important;
        border: 1px solid var(--border-color) !important;
        color: var(--text-primary) !important;
        font-weight: 600 !important;
    }
    
    .streamlit-expanderContent {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 0 0 10px 10px !important;
        border-top: none !important;
    }

    /* Animations */
    @keyframes slideInDown {
        from {
            transform: translateY(-30px);
            opacity: 0;
        }
        to {
            transform: translateY(0);
            opacity: 1;
        }
    }
    
    @keyframes fadeInUp {
        from {
            transform: translateY(20px);
            opacity: 0;
        }
        to {
            transform: translateY(0);
            opacity: 1;
        }
    }
    
    @keyframes shine {
        0% { transform: translateX(-100%) translateY(-100%) rotate(30deg); }
        100% { transform: translateX(100%) translateY(100%) rotate(30deg); }
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }

    /* Custom Segment Cards */
    .segment-card {
        background: linear-gradient(135deg, var(--bg-tertiary), var(--bg-secondary));
        border: 1px solid var(--border-color);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .segment-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
    }
    
    .segment-card:hover {
        transform: translateY(-3px);
        box-shadow: var(--shadow-hover);
        border-color: var(--primary-color);
    }
    
    .segment-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--primary-color);
        margin-bottom: 10px;
    }
    
    .segment-stats {
        color: var(--text-secondary);
        font-size: 0.9rem;
        line-height: 1.6;
    }
    
    /* Grid Layout for Segments */
    .segments-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 20px;
        margin-top: 20px;
    }
    
    /* Progress Bars */
    .progress-container {
        background: var(--bg-tertiary);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border: 1px solid var(--border-color);
    }
    
    .progress-bar {
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
        height: 8px;
        border-radius: 5px;
        transition: width 1s ease-in-out;
        box-shadow: 0 2px 8px rgba(255, 107, 53, 0.3);
    }
    
    /* Enhanced Warning/Error Messages */
    .stAlert {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 10px !important;
        color: var(--text-primary) !important;
    }
    
    /* Plotly Chart Enhancements */
    .js-plotly-plot {
        border-radius: 15px !important;
        overflow: hidden !important;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Mall_Customers.csv")
        df.columns = df.columns.str.strip()
        return df
    except FileNotFoundError:
        st.error("Mall_Customers.csv not found. Please upload the file.")
        st.stop()

df = load_data()

try:
    income_col = [c for c in df.columns if "income" in c.lower()][0]
    score_col = [c for c in df.columns if "score" in c.lower()][0]
    age_col = [c for c in df.columns if "age" in c.lower()][0]
    gender_col = [c for c in df.columns if "gender" in c.lower()][0]
except IndexError:
    st.error("Required columns not found. Ensure your CSV has Income, Spending Score, Age, and Gender columns.")
    st.stop()

# Dashboard Header
st.markdown("""
    <div class="dashboard-header">
        <h1 class="dashboard-title">Customer Segmentation Analytics</h1>
        <p class="dashboard-subtitle">AI-Powered Customer Intelligence & Business Intelligence Platform</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar Controls
with st.sidebar:
    st.markdown("""
        <div class="sidebar-header">
            <h2 style="margin: 0; color: white; font-size: 1.5rem; font-weight: 700;">Control Center</h2>
            <p style="margin: 5px 0 0 0; color: rgba(255,255,255,0.8); font-size: 0.9rem;">Configure your analysis</p>
        </div>
    """, unsafe_allow_html=True)

    # Refresh Data Button
    if st.button("Refresh Analysis", help="Recalculate all metrics and segments"):
        st.cache_data.clear()
        st.rerun()

    st.markdown("### Data Filters")
    
    income_range = st.slider(
        "Annual Income Range (k$)",
        int(df[income_col].min()),
        int(df[income_col].max()),
        (int(df[income_col].min()), int(df[income_col].max())),
        help="Filter customers by income range"
    )

    age_range = st.slider(
        " Age Range",
        int(df[age_col].min()),
        int(df[age_col].max()),
        (int(df[age_col].min()), int(df[age_col].max())),
        help="Filter customers by age range"
    )

    spending_range = st.slider(
        "Spending Score Range",
        int(df[score_col].min()),
        int(df[score_col].max()),
        (int(df[score_col].min()), int(df[score_col].max())),
        help="Filter customers by spending behavior"
    )

    genders = st.multiselect(
        "Gender Selection",
        df[gender_col].unique(),
        df[gender_col].unique(),
        help="Select gender categories to include"
    )

    st.markdown("### ML Configuration")
    
    n_clusters = st.slider(
        "Number of Clusters",
        2, 8, 5,
        help="Choose optimal number of customer segments"
    )
    
    algorithm = st.selectbox(
        "Clustering Algorithm",
        ["K-Means", "K-Means++"],
        index=1,
        help="Select clustering initialization method"
    )
    
    st.markdown("### ℹQuick Stats")
    st.info(f"**Dataset:** {len(df):,} total customers")
    st.info(f" **Filtered:** {len(df[(df[income_col] >= income_range[0]) & (df[income_col] <= income_range[1]) & (df[age_col] >= age_range[0]) & (df[age_col] <= age_range[1]) & (df[score_col] >= spending_range[0]) & (df[score_col] <= spending_range[1]) & (df[gender_col].isin(genders))]):,} customers")

# Apply Filters
df_filtered = df[
    (df[income_col] >= income_range[0]) & (df[income_col] <= income_range[1]) &
    (df[age_col] >= age_range[0]) & (df[age_col] <= age_range[1]) &
    (df[score_col] >= spending_range[0]) & (df[score_col] <= spending_range[1]) &
    (df[gender_col].isin(genders))
].copy()

if df_filtered.empty:
    st.warning(" No data matches current filters. Please adjust your selections.")
    st.stop()

@st.cache_data
def perform_clustering(data, n_clusters, algorithm_type):
    X = data[[income_col, score_col]].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    init_method = 'k-means++' if algorithm_type == 'K-Means++' else 'random'
    kmeans = KMeans(n_clusters=n_clusters, init=init_method, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    return clusters, kmeans, scaler

clusters, kmeans_model, scaler = perform_clustering(df_filtered, n_clusters, algorithm)
df_filtered["Cluster"] = clusters

# Key Performance Indicators
st.markdown("## Key Performance Indicators")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    total_customers = len(df_filtered)
    st.metric(
        label="Total Customers",
        value=f"{total_customers:,}",
        delta=f"{total_customers - len(df)}" if len(df_filtered) != len(df) else None
    )

with col2:
    avg_income = df_filtered[income_col].mean()
    st.metric(
        label="Avg Income",
        value=f"${avg_income:.0f}k",
        delta=f"{avg_income - df[income_col].mean():.1f}k"
    )

with col3:
    avg_score = df_filtered[score_col].mean()
    st.metric(
        label="Avg Spending Score",
        value=f"{avg_score:.1f}",
        delta=f"{avg_score - df[score_col].mean():.1f}"
    )

with col4:
    avg_age = df_filtered[age_col].mean()
    st.metric(
        label="Avg Age",
        value=f"{avg_age:.1f} yrs",
        delta=f"{avg_age - df[age_col].mean():.1f}"
    )

with col5:
    segments = df_filtered["Cluster"].nunique()
    st.metric(
        label="Active Segments",
        value=f"{segments}",
        delta=None
    )

# Customer Analytics Overview
st.markdown("## Customer Analytics Overview")

# Create tabs for different analysis views
tab1, tab2, tab3, tab4 = st.tabs(["Segmentation", "Demographics", "Trends", "Data Explorer"])

with tab1:
    col_seg1, col_seg2 = st.columns([2, 1])
    
    with col_seg1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Customer Segmentation Map</div>', unsafe_allow_html=True)
        
        # Enhanced Scatter Plot with better segment names
        df_filtered['Segment_Name'] = df_filtered['Cluster'].apply(lambda x: f'Segment {x}')
        
        fig_segments = px.scatter(
            df_filtered,
            x=income_col,
            y=score_col,
            color="Segment_Name",
            size=age_col,
            hover_data={
                'Segment_Name': True,
                age_col: True,
                gender_col: True
            },
            title="",
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        
        fig_segments.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_family="Inter",
            font_color="#ffffff",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(26, 26, 46, 0.8)",
                bordercolor="var(--border-color)",
                borderwidth=1
            ),
            xaxis_title="Annual Income (k$)",
            yaxis_title="Spending Score (1-100)",
            height=500
        )
        
        fig_segments.update_traces(
            marker=dict(
                line=dict(width=2, color='rgba(255,255,255,0.3)'),
                opacity=0.8
            )
        )
        
        st.plotly_chart(fig_segments, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_seg2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Segment Distribution</div>', unsafe_allow_html=True)
        
        # Cluster Size Analysis
        cluster_counts = df_filtered["Cluster"].value_counts().sort_index()
        
        fig_cluster_pie = px.pie(
            values=cluster_counts.values,
            names=[f"Segment {i}" for i in cluster_counts.index],
            title="",
            color_discrete_sequence=px.colors.sequential.Oranges_r
        )
        
        fig_cluster_pie.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_family="Inter",
            showlegend=True,
            legend=dict(
                orientation="h", 
                yanchor="bottom", 
                y=-0.2,
                xanchor="center",
                x=0.5
            )
        )
        
        st.plotly_chart(fig_cluster_pie, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Horizontal Segment Insights Grid
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">📋 Segment Insights Dashboard</div>', unsafe_allow_html=True)
    
    # Create horizontal grid of segment cards
    segments_data = []
    for cluster_id in sorted(df_filtered["Cluster"].unique()):
        cluster_data = df_filtered[df_filtered["Cluster"] == cluster_id]
        avg_income_cluster = cluster_data[income_col].mean()
        avg_score_cluster = cluster_data[score_col].mean()
        avg_age_cluster = cluster_data[age_col].mean()
        size = len(cluster_data)
        
        segments_data.append({
            'id': cluster_id,
            'size': size,
            'avg_income': avg_income_cluster,
            'avg_score': avg_score_cluster,
            'avg_age': avg_age_cluster,
            'percentage': (size / len(df_filtered)) * 100
        })
    
    # Create columns for horizontal layout
    num_segments = len(segments_data)
    cols = st.columns(num_segments)
    
    for i, (col, segment) in enumerate(zip(cols, segments_data)):
        with col:
            # Determine segment characteristics for better naming
            if segment['avg_income'] > df_filtered[income_col].mean() and segment['avg_score'] > df_filtered[score_col].mean():
                segment_type = "Premium"
                segment_color = "#ff6b35"
            elif segment['avg_income'] > df_filtered[income_col].mean() and segment['avg_score'] <= df_filtered[score_col].mean():
                segment_type = "Conservative"
                segment_color = "#4ecdc4"
            elif segment['avg_income'] <= df_filtered[income_col].mean() and segment['avg_score'] > df_filtered[score_col].mean():
                segment_type = "Aspirational"
                segment_color = "#f7931e"
            else:
                segment_type = "Budget-Conscious"
                segment_color = "#9b59b6"
            
            st.markdown(f"""
                <div class="segment-card">
                    <div class="segment-header"> Segment {segment['id']} - {segment_type}</div>
                    <div class="segment-stats">
                        <strong>Size:</strong> {segment['size']} customers ({segment['percentage']:.1f}%)<br>
                        <strong>Avg Income:</strong> ${segment['avg_income']:.0f}k<br>
                        <strong>Avg Spending:</strong> {segment['avg_score']:.1f}<br>
                        <strong>Avg Age:</strong> {segment['avg_age']:.1f} years
                    </div>
                    <div class="progress-container">
                        <div class="progress-bar" style="width: {segment['percentage']}%; background: {segment_color};"></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    col_demo1, col_demo2 = st.columns(2)
    
    with col_demo1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Age Distribution by Gender</div>', unsafe_allow_html=True)
        
        fig_age_gender = px.histogram(
            df_filtered,
            x=age_col,
            color=gender_col,
            barmode="overlay",
            opacity=0.7,
            title="",
            color_discrete_sequence=['#ff6b35', '#4ecdc4']
        )
        
        fig_age_gender.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_family="Inter",
            xaxis_title="Age",
            yaxis_title="Count",
            legend=dict(
                orientation="h",
                yanchor="top",
                y=1.1,
                xanchor="center",
                x=0.5
            )
        )
        
        st.plotly_chart(fig_age_gender, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Income vs Age Correlation
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Income vs Age Relationship</div>', unsafe_allow_html=True)
        
        fig_income_age = px.scatter(
            df_filtered,
            x=age_col,
            y=income_col,
            color=gender_col,
            trendline="ols",
            title="",
            color_discrete_sequence=['#ff6b35', '#4ecdc4']
        )
        
        fig_income_age.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_family="Inter",
            xaxis_title="Age",
            yaxis_title="Annual Income (k$)",
            legend=dict(
                orientation="h",
                yanchor="top",
                y=1.1,
                xanchor="center",
                x=0.5
            )
        )
        
        st.plotly_chart(fig_income_age, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_demo2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Spending Patterns by Gender</div>', unsafe_allow_html=True)
        
        fig_spending_gender = px.box(
            df_filtered,
            x=gender_col,
            y=score_col,
            color=gender_col,
            title="",
            color_discrete_sequence=['#ff6b35', '#4ecdc4']
        )
        
        fig_spending_gender.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_family="Inter",
            xaxis_title="Gender",
            yaxis_title="Spending Score",
            showlegend=False
        )
        
        st.plotly_chart(fig_spending_gender, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Heatmap of correlations
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Feature Correlation Matrix</div>', unsafe_allow_html=True)
        
        # Calculate correlation matrix
        corr_data = df_filtered[[age_col, income_col, score_col]].corr()
        
        fig_corr = px.imshow(
            corr_data,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdYlBu_r",
            title=""
        )
        
        fig_corr.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_family="Inter"
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    col_trend1, col_trend2 = st.columns(2)
    
    with col_trend1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Spending Trends by Age Groups</div>', unsafe_allow_html=True)
        
        # Create age groups
        df_filtered['Age_Group'] = pd.cut(
            df_filtered[age_col],
            bins=[0, 25, 35, 45, 55, 100],
            labels=['18-25', '26-35', '36-45', '46-55', '55+']
        )
        
        age_spending = df_filtered.groupby(['Age_Group', gender_col])[score_col].mean().reset_index()
        
        fig_age_spending = px.bar(
            age_spending,
            x='Age_Group',
            y=score_col,
            color=gender_col,
            barmode='group',
            title="",
            color_discrete_sequence=['#ff6b35', '#4ecdc4']
        )
        
        fig_age_spending.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_family="Inter",
            xaxis_title="Age Groups",
            yaxis_title="Average Spending Score",
            legend=dict(
                orientation="h",
                yanchor="top",
                y=1.1,
                xanchor="center",
                x=0.5
            )
        )
        
        st.plotly_chart(fig_age_spending, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Income Distribution by Cluster
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Income Distribution by Segment</div>', unsafe_allow_html=True)
        
        fig_income_cluster = px.violin(
            df_filtered,
            x="Segment_Name",
            y=income_col,
            color="Segment_Name",
            title="",
            color_discrete_sequence=px.colors.sequential.Oranges_r
        )
        
        fig_income_cluster.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_family="Inter",
            xaxis_title="Customer Segment",
            yaxis_title="Annual Income (k$)",
            showlegend=False,
            xaxis={'tickangle': 0}  # Horizontal labels
        )
        
        st.plotly_chart(fig_income_cluster, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_trend2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Customer Value Matrix</div>', unsafe_allow_html=True)
        
        # Calculate customer value (income * spending score)
        df_filtered['Customer_Value'] = df_filtered[income_col] * df_filtered[score_col] / 100
        
        fig_value_matrix = px.scatter(
            df_filtered,
            x=income_col,
            y=score_col,
            size='Customer_Value',
            color=age_col,
            title="",
            color_continuous_scale="Viridis",
            size_max=20
        )
        
        fig_value_matrix.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_family="Inter",
            xaxis_title="Annual Income (k$)",
            yaxis_title="Spending Score"
        )
        
        st.plotly_chart(fig_value_matrix, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Radar Chart for Segment Comparison
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Segment Characteristics Radar</div>', unsafe_allow_html=True)
        
        # Calculate segment averages
        segment_stats = df_filtered.groupby('Cluster').agg({
            age_col: 'mean',
            income_col: 'mean',
            score_col: 'mean'
        }).reset_index()
        
        # Normalize values for radar chart
        from sklearn.preprocessing import MinMaxScaler
        scaler_radar = MinMaxScaler()
        segment_stats_norm = segment_stats.copy()
        segment_stats_norm[['age_norm', 'income_norm', 'score_norm']] = scaler_radar.fit_transform(
            segment_stats[[age_col, income_col, score_col]]
        )
        
        # Create radar chart
        fig_radar = go.Figure()
        
        categories = ['Age', 'Income', 'Spending Score']
        
        for idx, row in segment_stats_norm.iterrows():
            values = [row['age_norm'], row['income_norm'], row['score_norm']]
            values += values[:1]  
            
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                fill='toself',
                name=f'Segment {row["Cluster"]}',
                opacity=0.7
            ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    gridcolor="#333",
                    linecolor="#333"
                ),
                angularaxis=dict(
                    gridcolor="#333",
                    linecolor="#333"
                )
            ),
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_family="Inter",
            showlegend=True,
            legend=dict(
                orientation="h", 
                yanchor="bottom", 
                y=-0.2,
                xanchor="center",
                x=0.5
            )
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

with tab4:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">Interactive Data Explorer</div>', unsafe_allow_html=True)
    
    # Search and filter options
    col_search1, col_search2, col_search3 = st.columns(3)
    
    with col_search1:
        search_cluster = st.selectbox(
            "Filter by Segment",
            ["All"] + [f"Segment {i}" for i in sorted(df_filtered["Cluster"].unique())]
        )
    
    with col_search2:
        sort_by = st.selectbox(
            "Sort by",
            [age_col, income_col, score_col, "Customer_Value"]
        )
    
    with col_search3:
        sort_order = st.selectbox(
            "Sort Order",
            ["Descending", "Ascending"]
        )
    
    # Apply data explorer filters
    df_display = df_filtered.copy()
    
    if search_cluster != "All":
        cluster_num = int(search_cluster.split()[-1])
        df_display = df_display[df_display["Cluster"] == cluster_num]
    
    # Sort data
    ascending = sort_order == "Ascending"
    df_display = df_display.sort_values(by=sort_by, ascending=ascending)
    
    # Display enhanced dataframe
    st.dataframe(
        df_display[[age_col, gender_col, income_col, score_col, "Cluster", "Customer_Value"]].round(2),
        use_container_width=True,
        hide_index=True,
        column_config={
            age_col: st.column_config.NumberColumn("Age", format="%d"),
            income_col: st.column_config.NumberColumn("Income", format="$%dk"),
            score_col: st.column_config.NumberColumn("Spending Score", format="%.1f"),
            "Customer_Value": st.column_config.NumberColumn("Customer Value", format="%.2f"),
            "Cluster": st.column_config.NumberColumn("Segment", format="Segment %d")
        }
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Demographics Summary Table
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">Demographic Summary Statistics</div>', unsafe_allow_html=True)
    
    demo_summary = df_filtered.groupby([gender_col, 'Cluster']).agg({
        age_col: ['mean', 'count'],
        income_col: 'mean',
        score_col: 'mean'
    }).round(2)
    
    demo_summary.columns = ['Avg Age', 'Count', 'Avg Income', 'Avg Spending Score']
    demo_summary = demo_summary.reset_index()
    
    st.dataframe(demo_summary, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Advanced Analytics Section
st.markdown("## Advanced Analytics")

col_advanced1, col_advanced2 = st.columns(2)

with col_advanced1:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">3D Customer Segmentation</div>', unsafe_allow_html=True)
    
    fig_3d = px.scatter_3d(
        df_filtered,
        x=age_col,
        y=income_col,
        z=score_col,
        color="Segment_Name",
        size='Customer_Value',
        title="",
        color_discrete_sequence=px.colors.qualitative.Set1,
        opacity=0.8
    )
    
    fig_3d.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_family="Inter",
        scene=dict(
            xaxis_title="Age",
            yaxis_title="Income (k$)",
            zaxis_title="Spending Score",
            bgcolor="rgba(0,0,0,0)",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0,
            xanchor="center",
            x=0.5
        )
    )
    
    st.plotly_chart(fig_3d, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_advanced2:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">Segment Comparison Matrix</div>', unsafe_allow_html=True)
    
    # Create comparison heatmap
    comparison_matrix = df_filtered.groupby('Cluster').agg({
        age_col: 'mean',
        income_col: 'mean',
        score_col: 'mean',
        'Customer_Value': 'mean'
    }).round(2)
    
    # Rename index for better display
    comparison_matrix.index = [f'Segment {i}' for i in comparison_matrix.index]
    
    fig_heatmap = px.imshow(
        comparison_matrix.T,
        title="",
        color_continuous_scale="Oranges",
        aspect="auto",
        text_auto=True
    )
    
    fig_heatmap.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_family="Inter",
        xaxis_title="Customer Segments",
        yaxis_title="Metrics",
        xaxis={'tickangle': 0}  # Horizontal labels
    )
    
    st.plotly_chart(fig_heatmap, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Customer Lifetime Value Estimation
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">Estimated Customer Lifetime Value</div>', unsafe_allow_html=True)
    
    # Simple CLV calculation (Income * Spending Score * Age Factor)
    df_filtered['CLV_Estimate'] = (
        df_filtered[income_col] * 
        df_filtered[score_col] * 
        (100 - df_filtered[age_col]) / 100
    ) / 100
    
    clv_by_cluster = df_filtered.groupby('Cluster')['CLV_Estimate'].mean().reset_index()
    clv_by_cluster['Segment_Name'] = clv_by_cluster['Cluster'].apply(lambda x: f'Segment {x}')
    
    fig_clv = px.bar(
        clv_by_cluster,
        x='Segment_Name',
        y='CLV_Estimate',
        title="",
        color='CLV_Estimate',
        color_continuous_scale="Oranges",
        text='CLV_Estimate'
    )
    
    fig_clv.update_traces(texttemplate='$%{text:.2f}', textposition='outside')
    
    fig_clv.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_family="Inter",
        xaxis_title="Customer Segment",
        yaxis_title="Estimated CLV",
        showlegend=False,
        xaxis={'tickangle': 0}  # Horizontal labels
    )
    
    st.plotly_chart(fig_clv, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Real-time Insights
st.markdown("## Real-time Insights")

col_insights1, col_insights2, col_insights3, col_insights4 = st.columns(4)

with col_insights1:
    high_value_customers = len(df_filtered[df_filtered['Customer_Value'] > df_filtered['Customer_Value'].quantile(0.8)])
    st.metric(
        label="High-Value Customers",
        value=f"{high_value_customers}",
        delta=f"{high_value_customers/len(df_filtered)*100:.1f}% of total"
    )

with col_insights2:
    avg_clv = df_filtered['CLV_Estimate'].mean()
    st.metric(
        label="Avg Customer Value",
        value=f"${avg_clv:.2f}",
        delta="Estimated lifetime value"
    )

with col_insights3:
    dominant_segment = df_filtered["Cluster"].mode().iloc[0]
    segment_size = len(df_filtered[df_filtered["Cluster"] == dominant_segment])
    st.metric(
        label="Dominant Segment",
        value=f"Segment {dominant_segment}",
        delta=f"{segment_size} customers"
    )

with col_insights4:
    income_diversity = df_filtered[income_col].std() / df_filtered[income_col].mean()
    st.metric(
        label="Income Diversity",
        value=f"{income_diversity:.2f}",
        delta="Coefficient of variation"
    )

# Performance Summary
with st.expander("Analysis Summary & Model Performance", expanded=False):
    col_perf1, col_perf2 = st.columns(2)
    
    with col_perf1:
        st.markdown("### Clustering Performance")
        
        # Calculate silhouette score approximation
        from sklearn.metrics import silhouette_score
        try:
            X_scaled = scaler.transform(df_filtered[[income_col, score_col]])
            sil_score = silhouette_score(X_scaled, clusters)
            
            # Performance indicator
            if sil_score > 0.5:
                performance_color = "#4ecdc4"
                performance_status = "Excellent"
            elif sil_score > 0.3:
                performance_color = "#f7931e"
                performance_status = "Good"
            else:
                performance_color = "#e74c3c"
                performance_status = "Fair"
            
            st.markdown(f"""
                <div style="background: var(--bg-secondary); padding: 25px; border-radius: 15px; border: 1px solid var(--border-color); box-shadow: var(--shadow);">
                    <h4 style="color: var(--primary-color); margin-top: 0; display: flex; align-items: center;">
                        Model Metrics
                    </h4>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 15px;">
                        <div>
                            <p style="margin: 5px 0;"><strong>Silhouette Score:</strong></p>
                            <p style="color: {performance_color}; font-size: 1.2rem; font-weight: 700;">{sil_score:.3f} ({performance_status})</p>
                        </div>
                        <div>
                            <p style="margin: 5px 0;"><strong>Segments:</strong> {n_clusters}</p>
                            <p style="margin: 5px 0;"><strong>Algorithm:</strong> {algorithm}</p>
                            <p style="margin: 5px 0;"><strong>Data Points:</strong> {len(df_filtered):,}</p>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Could not calculate performance metrics: {str(e)}")
    
    with col_perf2:
        st.markdown("### Data Quality Insights")
        
        missing_data = df_filtered.isnull().sum().sum()
        completeness = (1 - missing_data / (len(df_filtered) * len(df_filtered.columns))) * 100
        
        st.markdown(f"""
            <div style="background: var(--bg-secondary); padding: 25px; border-radius: 15px; border: 1px solid var(--border-color); box-shadow: var(--shadow);">
                <h4 style="color: var(--primary-color); margin-top: 0;">📊 Data Quality</h4>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 15px;">
                    <div>
                        <p style="margin: 5px 0;"><strong>Completeness:</strong></p>
                        <p style="color: #4ecdc4; font-size: 1.2rem; font-weight: 700;">{completeness:.1f}%</p>
                    </div>
                    <div>
                        <p style="margin: 5px 0;"><strong>Missing Values:</strong> {missing_data}</p>
                        <p style="margin: 5px 0;"><strong>Unique Customers:</strong> {len(df_filtered):,}</p>
                        <p style="margin: 5px 0;"><strong>Features Used:</strong> {len([income_col, score_col])}</p>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

# Additional Trend Analysis
st.markdown("## Trend Analysis")

col_trend1, col_trend2 = st.columns(2)

with col_trend1:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">Spending Score Trends by Age</div>', unsafe_allow_html=True)
    
    # Spending trends by age
    spending_by_age = df_filtered.groupby(age_col)[score_col].mean().reset_index()
    
    fig_spending_trend = px.line(
        spending_by_age,
        x=age_col,
        y=score_col,
        title="",
        markers=True,
        line_shape='spline'
    )
    
    fig_spending_trend.update_traces(
        line_color='#ff6b35',
        marker_color='#ff6b35',
        marker_size=10,
        line_width=3
    )
    
    fig_spending_trend.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_family="Inter",
        xaxis_title="Age",
        yaxis_title="Average Spending Score"
    )
    
    st.plotly_chart(fig_spending_trend, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_trend2:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">Income Trends by Age</div>', unsafe_allow_html=True)
    
    income_by_age = df_filtered.groupby(age_col)[income_col].mean().reset_index()
    
    fig_income_trend = px.area(
        income_by_age,
        x=age_col,
        y=income_col,
        title="",
        color_discrete_sequence=['#4ecdc4']
    )
    
    fig_income_trend.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_family="Inter",
        xaxis_title="Age",
        yaxis_title="Average Income (k$)"
    )
    
    fig_income_trend.update_traces(
        fill='tonexty',
        fillcolor='rgba(78, 205, 196, 0.3)',
        line_color='#4ecdc4',
        line_width=3
    )
    
    st.plotly_chart(fig_income_trend, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Segment Performance Dashboard
st.markdown("## Segment Performance Dashboard")

# Create cluster performance metrics
cluster_metrics = df_filtered.groupby('Cluster').agg({
    age_col: 'mean',
    income_col: 'mean',
    score_col: 'mean',
    gender_col: 'count'
}).round(2)
cluster_metrics.columns = ['Avg Age', 'Avg Income', 'Avg Spending', 'Size']
cluster_metrics = cluster_metrics.reset_index()
cluster_metrics['Segment_Name'] = cluster_metrics['Cluster'].apply(lambda x: f'Segment {x}')

# Create horizontal bar charts for better segment comparison
col_perf1, col_perf2 = st.columns(2)

with col_perf1:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">Segment Performance Metrics</div>', unsafe_allow_html=True)
    
    # Create subplots for better organization
    fig_performance = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Average Age', 'Average Income', 'Average Spending', 'Segment Size'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Add traces with horizontal orientation
    fig_performance.add_trace(
        go.Bar(x=cluster_metrics['Segment_Name'], y=cluster_metrics['Avg Age'], 
               name='Age', marker_color='#ff6b35'),
        row=1, col=1
    )
    
    fig_performance.add_trace(
        go.Bar(x=cluster_metrics['Segment_Name'], y=cluster_metrics['Avg Income'], 
               name='Income', marker_color='#4ecdc4'),
        row=1, col=2
    )
    
    fig_performance.add_trace(
        go.Bar(x=cluster_metrics['Segment_Name'], y=cluster_metrics['Avg Spending'], 
               name='Spending', marker_color='#f7931e'),
        row=2, col=1
    )
    
    fig_performance.add_trace(
        go.Bar(x=cluster_metrics['Segment_Name'], y=cluster_metrics['Size'], 
               name='Size', marker_color='#9b59b6'),
        row=2, col=2
    )
    
    # Update layout for horizontal labels
    fig_performance.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_family="Inter",
        showlegend=False,
        height=500
    )
    
    # Make all x-axis labels horizontal
    fig_performance.update_xaxes(tickangle=0)
    
    st.plotly_chart(fig_performance, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_perf2:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">Segment Characteristics Overview</div>', unsafe_allow_html=True)
    
    # Create a comprehensive segment overview
    for cluster_id in sorted(df_filtered["Cluster"].unique()):
        cluster_data = df_filtered[df_filtered["Cluster"] == cluster_id]
        avg_income_cluster = cluster_data[income_col].mean()
        avg_score_cluster = cluster_data[score_col].mean()
        avg_age_cluster = cluster_data[age_col].mean()
        size = len(cluster_data)
        percentage = (size / len(df_filtered)) * 100
        
        # Determine segment type and characteristics
        if avg_income_cluster > df_filtered[income_col].mean() and avg_score_cluster > df_filtered[score_col].mean():
            segment_type = "Premium Customers"
            segment_color = "#ff6b35"
            segment_desc = "High income, high spending"
        elif avg_income_cluster > df_filtered[income_col].mean() and avg_score_cluster <= df_filtered[score_col].mean():
            segment_type = "Conservative Spenders"
            segment_color = "#4ecdc4"
            segment_desc = "High income, low spending"
        elif avg_income_cluster <= df_filtered[income_col].mean() and avg_score_cluster > df_filtered[score_col].mean():
            segment_type = "Aspirational Buyers"
            segment_color = "#f7931e"
            segment_desc = "Lower income, high spending"
        else:
            segment_type = "Budget-Conscious"
            segment_color = "#9b59b6"
            segment_desc = "Lower income, low spending"
        
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, var(--bg-tertiary), var(--bg-secondary)); 
                        border: 1px solid {segment_color}; 
                        border-radius: 15px; 
                        padding: 20px; 
                        margin: 15px 0; 
                        box-shadow: var(--shadow);
                        transition: all 0.3s ease;">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 15px;">
                    <h4 style="margin: 0; color: {segment_color}; font-size: 1.1rem;">Segment {cluster_id}</h4>
                    <span style="background: {segment_color}; color: white; padding: 5px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600;">{segment_type}</span>
                </div>
                <p style="color: var(--text-secondary); font-size: 0.9rem; margin-bottom: 10px; font-style: italic;">{segment_desc}</p>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                    <div>
                        <p style="margin: 2px 0; font-size: 0.85rem;"><strong>Size:</strong> {size} ({percentage:.1f}%)</p>
                        <p style="margin: 2px 0; font-size: 0.85rem;"><strong>Avg Age:</strong> {avg_age_cluster:.1f} years</p>
                    </div>
                    <div>
                        <p style="margin: 2px 0; font-size: 0.85rem;"><strong>Income:</strong> ${avg_income_cluster:.0f}k</p>
                        <p style="margin: 2px 0; font-size: 0.85rem;"><strong>Spending:</strong> {avg_score_cluster:.1f}</p>
                    </div>
                </div>
                <div style="background: var(--bg-primary); border-radius: 8px; padding: 8px; margin-top: 10px;">
                    <div style="width: {percentage}%; height: 6px; background: linear-gradient(90deg, {segment_color}, {segment_color}80); border-radius: 3px; box-shadow: 0 2px 8px rgba(255, 107, 53, 0.3);"></div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer with enhanced styling
st.markdown("---")
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.markdown(f"""
    <div style="background: linear-gradient(135deg, var(--bg-secondary), var(--bg-tertiary)); 
                padding: 25px; 
                border-radius: 15px; 
                text-align: center; 
                color: var(--text-secondary); 
                font-size: 0.9rem; 
                margin: 30px 0 20px 0;
                border: 1px solid var(--border-color);
                box-shadow: var(--shadow);">
        <div style="display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap; gap: 20px;">
            <div><strong>Customer Analytics Dashboard</strong></div>
            <div>Last Updated: {current_time}</div>
            <div>Analyzing {len(df_filtered):,} customers</div>
            <div>{n_clusters} segments identified</div>
            <p> Made by Arwa Abbas </p>
        </div>
    </div>
""", unsafe_allow_html=True)

# Add JavaScript 
st.markdown("""
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        // Add loading animation for charts
        const charts = document.querySelectorAll('.js-plotly-plot');
        charts.forEach((chart, index) => {
            chart.style.opacity = '0';
            chart.style.transform = 'translateY(20px)';
            chart.style.transition = 'all 0.6s ease-out';
            
            setTimeout(() => {
                chart.style.opacity = '1';
                chart.style.transform = 'translateY(0)';
            }, index * 200);
        });
        
        // Add hover effects to metrics
        const metrics = document.querySelectorAll('[data-testid="metric-container"]');
        metrics.forEach(metric => {
            metric.addEventListener('mouseenter', function() {
                this.style.transform = 'scale(1.05) translateY(-5px)';
                this.style.zIndex = '10';
            });
            
            metric.addEventListener('mouseleave', function() {
                this.style.transform = 'scale(1) translateY(0)';
                this.style.zIndex = '1';
            });
        });
        
        // Add segment card hover effects
        const segmentCards = document.querySelectorAll('.segment-card');
        segmentCards.forEach(card => {
            card.addEventListener('mouseenter', function() {
                this.style.transform = 'translateY(-5px) scale(1.02)';
                this.style.zIndex = '10';
            });
            
            card.addEventListener('mouseleave', function() {
                this.style.transform = 'translateY(0) scale(1)';
                this.style.zIndex = '1';
            });
        });
        
        // Smooth scroll for navigation
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                document.querySelector(this.getAttribute('href')).scrollIntoView({
                    behavior: 'smooth'
                });
            });
        });
    });
    </script>
""", unsafe_allow_html=True)
