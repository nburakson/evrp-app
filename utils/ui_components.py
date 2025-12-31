"""
UI Components and Styling for EVRP Application
Professional styling and reusable components
"""

import streamlit as st


def apply_custom_css():
    """Apply custom CSS for professional look"""
    st.markdown("""
    <style>
    /* Main theme colors */
    :root {
        --primary-color: #D31027;
        --secondary-color: #EA384D;
        --success-color: #06D6A0;
        --warning-color: #F77F00;
        --danger-color: #C1121F;
        --bg-light: #F8F9FA;
        --text-dark: #2D3748;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #D31027 0%, #EA384D 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-header p {
        font-size: 1.1rem;
        margin-top: 0.5rem;
        opacity: 0.95;
    }
    
    /* Card styling */
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        border-left: 4px solid var(--primary-color);
        margin-bottom: 1rem;
    }
    
    .success-card {
        background: linear-gradient(135deg, #06D6A0 0%, #05B48C 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    
    .warning-card {
        background: linear-gradient(135deg, #FFC371 0%, #FF5F6D 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    
    /* Metric cards */
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
    }
    
    div[data-testid="metric-container"] {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        border: 1px solid #E2E8F0;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #D31027 0%, #EA384D 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #F7FAFC;
        padding: 10px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        border: 1px solid #E2E8F0;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #D31027 0%, #EA384D 100%);
        color: white;
        border: none;
    }
    
    /* DataFrame styling - Excel benzeri görünüm */
    .dataframe {
        border: 1px solid #E2E8F0 !important;
        border-radius: 10px !important;
        overflow: hidden !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08) !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
    }
    
    .dataframe thead tr th {
        background: linear-gradient(135deg, #D31027 0%, #EA384D 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        padding: 14px 16px !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        text-align: left !important;
        text-transform: uppercase !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.5px !important;
    }
    
    .dataframe tbody tr {
        border-bottom: 1px solid #E2E8F0 !important;
    }
    
    .dataframe tbody tr:nth-child(even) {
        background-color: #F9FAFB !important;
    }
    
    .dataframe tbody tr:nth-child(odd) {
        background-color: white !important;
    }
    
    .dataframe tbody tr:hover {
        background-color: #FEF2F2 !important;
        transition: all 0.2s ease !important;
        transform: scale(1.001) !important;
    }
    
    .dataframe tbody td {
        padding: 12px 16px !important;
        border: 1px solid #E2E8F0 !important;
        color: #2D3748 !important;
        font-size: 0.9rem !important;
    }
    
    /* Tablo container */
    div[data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
    }
    
    div[data-testid="stDataFrame"] > div {
        border-radius: 10px !important;
    }
    
    /* Input styling */
    .stTextInput>div>div>input,
    .stNumberInput>div>div>input,
    .stSelectbox>div>div>select {
        border-radius: 8px;
        border: 2px solid #E2E8F0;
        padding: 0.75rem;
    }
    
    .stTextInput>div>div>input:focus,
    .stNumberInput>div>div>input:focus {
        border-color: #D31027;
        box-shadow: 0 0 0 3px rgba(211, 16, 39, 0.1);
    }
    
    /* Slider styling */
    .stSlider>div>div>div>div {
        background: linear-gradient(135deg, #D31027 0%, #EA384D 100%);
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #06D6A0 0%, #05B48C 100%);
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #F7FAFC 0%, #EDF2F7 100%);
    }
    
    /* Section dividers */
    hr {
        margin: 2rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #E2E8F0, transparent);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #F7FAFC;
        border-radius: 8px;
        font-weight: 600;
        border: 1px solid #E2E8F0;
    }
    
    /* File uploader */
    .stFileUploader>div>div {
        border: 2px dashed #667eea;
        border-radius: 8px;
        background-color: #F7FAFC;
    }
    
    /* Custom scroll bar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #F7FAFC;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Alert boxes - Daha belirgin ve profesyonel */
    .stAlert {
        border-radius: 10px !important;
        border-left: 5px solid !important;
        padding: 1.2rem 1.5rem !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08) !important;
        font-weight: 500 !important;
        margin: 1rem 0 !important;
    }
    
    /* Info alert - BİLGİ (Mavi) */
    div[data-baseweb="notification"][kind="info"] {
        background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%) !important;
        border-left-color: #2196F3 !important;
        color: #0D47A1 !important;
    }
    
    /* Warning alert - UYARI (Turuncu/Sarı) */
    div[data-baseweb="notification"][kind="warning"] {
        background: linear-gradient(135deg, #FFF3E0 0%, #FFE0B2 100%) !important;
        border-left-color: #FF9800 !important;
        color: #E65100 !important;
    }
    
    /* Success alert - BAŞARILI (Yeşil) */
    div[data-baseweb="notification"][kind="success"] {
        background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%) !important;
        border-left-color: #4CAF50 !important;
        color: #1B5E20 !important;
    }
    
    /* Error alert - HATA (Kırmızı) */
    div[data-baseweb="notification"][kind="error"] {
        background: linear-gradient(135deg, #FFEBEE 0%, #FFCDD2 100%) !important;
        border-left-color: #F44336 !important;
        color: #B71C1C !important;
    }
    </style>
    """, unsafe_allow_html=True)


def render_header(title, subtitle=None):
    """Render a professional header"""
    st.markdown(f"""
    <div class="main-header">
        <h1>🚚 {title}</h1>
        {f'<p>{subtitle}</p>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)


def info_card(title, content, icon="ℹ️"):
    """Render an info card"""
    st.markdown(f"""
    <div class="info-card">
        <h3>{icon} {title}</h3>
        <p>{content}</p>
    </div>
    """, unsafe_allow_html=True)


def success_card(content):
    """Render a success card"""
    st.markdown(f"""
    <div class="success-card">
        <p style="margin: 0; font-size: 1.1rem;">✅ {content}</p>
    </div>
    """, unsafe_allow_html=True)


def warning_card(content):
    """Render a warning card"""
    st.markdown(f"""
    <div class="warning-card">
        <p style="margin: 0; font-size: 1.1rem;">⚠️ {content}</p>
    </div>
    """, unsafe_allow_html=True)


def metric_card(label, value, delta=None, icon="📊"):
    """Render a styled metric card"""
    delta_html = f'<p style="color: #06D6A0; font-size: 0.9rem; margin-top: 0.5rem;">{delta}</p>' if delta else ''
    
    st.markdown(f"""
    <div style="background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); text-align: center;">
        <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
        <div style="color: #718096; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem;">{label}</div>
        <div style="font-size: 2rem; font-weight: 700; color: #2D3748;">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)
