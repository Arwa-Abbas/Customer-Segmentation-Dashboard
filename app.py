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
        --bg-primary: #0a0a0a;
        --bg-secondary: #1a1a1a;
        --bg-tertiary: #2a2a2a;
        --text-primary: #ffffff;
        --text-secondary: #b0b0b0;
        --border-color: #333333;
        --shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        --shadow-hover: 0 8px 30px rgba(255, 107, 53, 0.2);
    }

    /* Main App Container */
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
        color: var(--text-primary);
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar Styling */
    .css-1d391kg, .css-1y4p8pa {
        background: linear-gradient(180deg, #1a1a1a 0%, #0f0f0f 100%);
        border-right: 2px solid var(--border-color);
    }
    
    /* Custom Header */
    .dashboard-header {
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 30px;
        text-align: center;
        box-shadow: var(--shadow);
        animation: slideInDown 0.8s ease-out;
    }
    
    .dashboard-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: white;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .dashboard-subtitle {
        font-size: 1.1rem;
        color: rgba(255,255,255,0.9);
        margin: 5px 0 0 0;
        font-weight: 400;
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
    }
    
    .metric-card:hover::before {
        left: 100%;
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: var(--primary-color);
        margin: 10px 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: var(--text-secondary);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-icon {
        font-size: 2rem;
        margin-bottom: 10px;
        opacity: 0.8;
    }

    /* Chart Container Styling */
    .chart-container {
        background: var(--bg-secondary);
        padding: 25px;
        border-radius: 15px;
        border: 1px solid var(--border-color);
        box-shadow: var(--shadow);
        margin-bottom: 25px;
        transition: all 0.3s ease;
        animation: fadeInUp 0.6s ease-out;
    }
    
    .chart-container:hover {
        transform: translateY(-3px);
        box-shadow: var(--shadow-hover);
    }
    
    .chart-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: var(--primary-color);
        margin-bottom: 20px;
        text-align: center;
    }

    /* Streamlit Native Elements Override */
    .stMetric {
        background: none !important;
        border: none !important;
        box-shadow: none !important;
    }
    
    .stMetric > div {
        background: var(--bg-secondary) !important;
        padding: 20px !important;
        border-radius: 15px !important;
        border: 1px solid var(--border-color) !important;
        box-shadow: var(--shadow) !important;
        transition: all 0.3s ease !important;
    }
    
    .stMetric > div:hover {
        transform: translateY(-5px) !important;
        box-shadow: var(--shadow-hover) !important;
    }
    
    .stMetric label {
        color: var(--text-secondary) !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }
    
    .stMetric .metric-value {
        color: var(--primary-color) !important;
        font-weight: 700 !important;
        font-size: 2rem !important;
    }

    /* Sidebar Enhancements */
    .sidebar-header {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: var(--shadow);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 25px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: var(--shadow);
        font-family: 'Inter', sans-serif;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 107, 53, 0.4);
        background: linear-gradient(45deg, var(--secondary-color), var(--primary-color));
    }
    
    /* Sliders */
    .stSlider > div > div > div > div {
        background-color: var(--primary-color) !important;
    }
    
    /* Dataframe */
    .stDataFrame {
        background: var(--bg-secondary);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid var(--border-color);
        box-shadow: var(--shadow);
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
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .pulse-animation {
        animation: pulse 2s infinite;
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
    }
    
    /* Custom Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--bg-secondary);
        border-radius: 10px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: var(--text-secondary);
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
        color: white;
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


st.markdown("""
    <div class="dashboard-header">
        <h1 class="dashboard-title">Customer Segmentation Analytics</h1>
        <p class="dashboard-subtitle">ML-Powered Customer Intelligence Dashboard</p>
    </div>
""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown("""
        <div class="sidebar-header">
            <h2 style="margin: 0; color: white; font-size: 1.4rem;">Controls</h2>
        </div>
    """, unsafe_allow_html=True)

    # Refresh Data Button
    if st.button("Refresh Analysis", help="Recalculate all metrics and segments"):
        st.cache_data.clear()
        st.experimental_rerun()  # <-- use experimental_rerun instead of st.rerun()

    # Now define your sliders AFTER the refresh button
    income_range = st.slider(
        "Annual Income Range (k$)",
        int(df[income_col].min()),
        int(df[income_col].max()),
        (int(df[income_col].min()), int(df[income_col].max())),
        help="Filter customers by income range"
    )

    age_range = st.slider(
        "Age Range",
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
    
    # Refresh Data Button
    if st.button("Refresh Analysis", help="Recalculate all metrics and segments"):
        st.cache_data.clear()
        st.rerun()

# Apply Filters
df_filtered = df[
    (df[income_col] >= income_range[0]) & (df[income_col] <= income_range[1]) &
    (df[age_col] >= age_range[0]) & (df[age_col] <= age_range[1]) &
    (df[score_col] >= spending_range[0]) & (df[score_col] <= spending_range[1]) &
    (df[gender_col].isin(genders))
].copy()

if df_filtered.empty:
    st.warning("No data matches current filters. Please adjust your selections.")
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


st.markdown("## Customer Analytics Overview")

# Create tabs for different analysis views
tab1, tab2, tab3, tab4 = st.tabs(["Segmentation", "Demographics", "Trends", "Data Explorer"])

with tab1:
    col_seg1, col_seg2 = st.columns([2, 1])
    
    with col_seg1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Customer Segmentation Map</div>', unsafe_allow_html=True)
        
        # Enhanced Scatter Plot
        fig_segments = px.scatter(
            df_filtered,
            x=income_col,
            y=score_col,
            color="Cluster",
            size=age_col,
            hover_data={
                'Cluster': True,
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
                xanchor="right",
                x=1
            ),
            xaxis_title="Annual Income (k$)",
            yaxis_title="Spending Score (1-100)"
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
            legend=dict(orientation="v", x=1.05)
        )
        
        st.plotly_chart(fig_cluster_pie, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Segment Summary Stats
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Segment Insights</div>', unsafe_allow_html=True)
        
        for cluster_id in sorted(df_filtered["Cluster"].unique()):
            cluster_data = df_filtered[df_filtered["Cluster"] == cluster_id]
            avg_income_cluster = cluster_data[income_col].mean()
            avg_score_cluster = cluster_data[score_col].mean()
            
            st.markdown(f"""
                <div style="background: var(--bg-tertiary); padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid var(--primary-color);">
                    <h4 style="margin: 0; color: var(--primary-color);">Segment {cluster_id}</h4>
                    <p style="margin: 5px 0; color: var(--text-secondary); font-size: 0.9rem;">
                        {len(cluster_data)} customers<br>
                        ${avg_income_cluster:.0f}k avg income<br>
                        {avg_score_cluster:.1f} avg score
                    </p>
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
            yaxis_title="Count"
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
            yaxis_title="Annual Income (k$)"
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
            yaxis_title="Average Spending Score"
        )
        
        st.plotly_chart(fig_age_spending, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Income Distribution by Cluster
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Income Distribution by Segment</div>', unsafe_allow_html=True)
        
        fig_income_cluster = px.violin(
            df_filtered,
            x="Cluster",
            y=income_col,
            color="Cluster",
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
            showlegend=False
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
            values += values[:1]  # Complete the circle
            
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                fill='toself',
                name=f'Segment {row["Cluster"]}',
                opacity=0.6
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
            legend=dict(orientation="h", yanchor="bottom", y=-0.1)
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    col_demo1, col_demo2, col_demo3 = st.columns(3)
    
    with col_demo1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Gender Distribution</div>', unsafe_allow_html=True)
        
        gender_counts = df_filtered[gender_col].value_counts()
        
        fig_gender = px.pie(
            values=gender_counts.values,
            names=gender_counts.index,
            title="",
            color_discrete_sequence=['#ff6b35', '#4ecdc4']
        )
        
        fig_gender.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_family="Inter",
            showlegend=True
        )
        
        st.plotly_chart(fig_gender, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_demo2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Age Demographics</div>', unsafe_allow_html=True)
        
        fig_age_dist = px.histogram(
            df_filtered,
            x=age_col,
            nbins=15,
            title="",
            color_discrete_sequence=['#ff6b35']
        )
        
        fig_age_dist.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_family="Inter",
            xaxis_title="Age",
            yaxis_title="Count",
            bargap=0.1
        )
        
        st.plotly_chart(fig_age_dist, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_demo3:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Income Brackets</div>', unsafe_allow_html=True)
        
        # Create income brackets
        df_filtered['Income_Bracket'] = pd.cut(
            df_filtered[income_col],
            bins=5,
            labels=['Low', 'Low-Mid', 'Medium', 'Mid-High', 'High']
        )
        
        income_bracket_counts = df_filtered['Income_Bracket'].value_counts()
        
        fig_income_brackets = px.bar(
            x=income_bracket_counts.index,
            y=income_bracket_counts.values,
            title="",
            color=income_bracket_counts.values,
            color_continuous_scale="Oranges"
        )
        
        fig_income_brackets.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_family="Inter",
            xaxis_title="Income Bracket",
            yaxis_title="Count",
            showlegend=False
        )
        
        st.plotly_chart(fig_income_brackets, use_container_width=True)
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

with tab3:
    col_trend1, col_trend2 = st.columns(2)
    
    with col_trend1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Spending Score Trends</div>', unsafe_allow_html=True)
        
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
            marker_size=8
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
        
        # Income trends
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
        
        st.plotly_chart(fig_income_trend, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_trend2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Cluster Performance Metrics</div>', unsafe_allow_html=True)
        
        # Cluster performance metrics
        cluster_metrics = df_filtered.groupby('Cluster').agg({
            age_col: 'mean',
            income_col: 'mean',
            score_col: 'mean',
            gender_col: 'count'
        }).round(2)
        cluster_metrics.columns = ['Avg Age', 'Avg Income', 'Avg Spending', 'Size']
        cluster_metrics = cluster_metrics.reset_index()
        
        # Create a multi-metric bar chart
        fig_cluster_metrics = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Average Age', 'Average Income', 'Average Spending', 'Cluster Size'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Add traces
        fig_cluster_metrics.add_trace(
            go.Bar(x=cluster_metrics['Cluster'], y=cluster_metrics['Avg Age'], 
                   name='Age', marker_color='#ff6b35'),
            row=1, col=1
        )
        
        fig_cluster_metrics.add_trace(
            go.Bar(x=cluster_metrics['Cluster'], y=cluster_metrics['Avg Income'], 
                   name='Income', marker_color='#4ecdc4'),
            row=1, col=2
        )
        
        fig_cluster_metrics.add_trace(
            go.Bar(x=cluster_metrics['Cluster'], y=cluster_metrics['Avg Spending'], 
                   name='Spending', marker_color='#f7931e'),
            row=2, col=1
        )
        
        fig_cluster_metrics.add_trace(
            go.Bar(x=cluster_metrics['Cluster'], y=cluster_metrics['Size'], 
                   name='Size', marker_color='#9b59b6'),
            row=2, col=2
        )
        
        fig_cluster_metrics.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_family="Inter",
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig_cluster_metrics, use_container_width=True)
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
        
        fig_clv = px.bar(
            clv_by_cluster,
            x='Cluster',
            y='CLV_Estimate',
            title="",
            color='CLV_Estimate',
            color_continuous_scale="Oranges"
        )
        
        fig_clv.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_family="Inter",
            xaxis_title="Customer Segment",
            yaxis_title="Estimated CLV",
            showlegend=False
        )
        
        st.plotly_chart(fig_clv, use_container_width=True)
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
        color="Cluster",
        size='Customer_Value',
        title="",
        color_discrete_sequence=px.colors.qualitative.Set1,
        opacity=0.7
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
            bgcolor="rgba(0,0,0,0)"
        ),
        height=500
    )
    
    st.plotly_chart(fig_3d, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_advanced2:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">Segment Comparison Matrix</div>', unsafe_allow_html=True)
    
    # Heatmap of segment characteristics
    pivot_data = df_filtered.groupby(['Cluster', gender_col]).agg({
        age_col: 'mean',
        income_col: 'mean',
        score_col: 'mean'
    }).round(1)
    
    # Create comparison heatmap
    comparison_matrix = df_filtered.groupby('Cluster').agg({
        age_col: 'mean',
        income_col: 'mean',
        score_col: 'mean',
        'Customer_Value': 'mean'
    }).round(2)
    
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
        yaxis_title="Metrics"
    )
    
    st.plotly_chart(fig_heatmap, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

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
# ----------------------------
with st.expander("🔍 Analysis Summary & Model Performance", expanded=False):
    col_perf1, col_perf2 = st.columns(2)
    
    with col_perf1:
        st.markdown("### Clustering Performance")
        
        # Calculate silhouette score approximation
        from sklearn.metrics import silhouette_score
        try:
            X_scaled = scaler.transform(df_filtered[[income_col, score_col]])
            sil_score = silhouette_score(X_scaled, clusters)
            
            st.markdown(f"""
                <div style="background: var(--bg-secondary); padding: 20px; border-radius: 10px; border: 1px solid var(--border-color);">
                    <h4 style="color: var(--primary-color); margin-top: 0;">Model Metrics</h4>
                    <p><strong>Silhouette Score:</strong> {sil_score:.3f}</p>
                    <p><strong>Number of Segments:</strong> {n_clusters}</p>
                    <p><strong>Algorithm:</strong> {algorithm}</p>
                    <p><strong>Data Points:</strong> {len(df_filtered):,}</p>
                </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Could not calculate performance metrics: {str(e)}")
    
    with col_perf2:
        st.markdown("### Data Quality Insights")
        
        missing_data = df_filtered.isnull().sum().sum()
        completeness = (1 - missing_data / (len(df_filtered) * len(df_filtered.columns))) * 100
        
        st.markdown(f"""
            <div style="background: var(--bg-secondary); padding: 20px; border-radius: 10px; border: 1px solid var(--border-color);">
                <h4 style="color: var(--primary-color); margin-top: 0;">Data Quality</h4>
                <p><strong>Data Completeness:</strong> {completeness:.1f}%</p>
                <p><strong>Missing Values:</strong> {missing_data}</p>
                <p><strong>Unique Customers:</strong> {len(df_filtered):,}</p>
                <p><strong>Features Used:</strong> {len([income_col, score_col])}</p>
            </div>
        """, unsafe_allow_html=True)

# ----------------------------
# Footer with Timestamp
# ----------------------------
st.markdown("---")
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.markdown(f"""
    <div style="text-align: center; color: var(--text-secondary); font-size: 0.9rem; padding: 20px;">
        Customer Analytics Dashboard | Last Updated: {current_time} | 
        Analyzing {len(df_filtered):,} customers across {n_clusters} segments
    </div>
""", unsafe_allow_html=True)

# ----------------------------
# Custom JavaScript for Enhanced Interactivity
# ----------------------------
st.markdown("""
    <script>
    // Add smooth scrolling and enhanced interactions
    document.addEventListener('DOMContentLoaded', function() {
        // Add loading animation
        const charts = document.querySelectorAll('.js-plotly-plot');
        charts.forEach(chart => {
            chart.style.opacity = '0';
            chart.style.transform = 'translateY(20px)';
            chart.style.transition = 'all 0.6s ease-out';
            
            setTimeout(() => {
                chart.style.opacity = '1';
                chart.style.transform = 'translateY(0)';
            }, Math.random() * 500);
        });
        
        // Add hover effects to metrics
        const metrics = document.querySelectorAll('[data-testid="metric-container"]');
        metrics.forEach(metric => {
            metric.addEventListener('mouseenter', function() {
                this.style.transform = 'scale(1.05)';
            });
            
            metric.addEventListener('mouseleave', function() {
                this.style.transform = 'scale(1)';
            });
        });
    });
    </script>
""", unsafe_allow_html=True)
