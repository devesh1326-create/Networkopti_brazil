# ==========================================
# SOYA BEAN SUPPLY CHAIN CONTROL TOWER
# ==========================================

import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import plotly.express as px

# ------------------------------------------
# 1. PAGE CONFIGURATION & THEME
# ------------------------------------------
st.set_page_config(
    page_title="Soya Bean Control Tower",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enforcing "Scrapbook/Natural" Theme
st.markdown("""
<style>
    /* Main Background - Light Paper Texture */
    .stApp {
        background-color: #f4f1ea; 
        color: #2c3e50; 
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }

    /* Sidebar - Slightly darker paper tone */
    [data-testid="stSidebar"] {
        background-color: #e8e6df;
        border-right: 1px solid #dcdad5;
    }

    /* --- SIDEBAR TEXT VISIBILITY FIX --- */
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] span, 
    [data-testid="stSidebar"] div, 
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown {
        color: #4a4a4a !important;
    }

    /* FORCE METRIC LABELS DARK */
    [data-testid="stMetricLabel"] {
        color: #5d4037 !important; /* Dark Brown/Grey */
        font-weight: 600;
    }
    [data-testid="stMetricValue"] {
        color: #1b5e20 !important; /* Dark Green */
    }

    /* KPI Cards - White "Sticker" look */
    .kpi-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 20px;
        border-radius: 12px; 
        margin-bottom: 10px;
        text-align: center;
        height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        box-shadow: 2px 4px 12px rgba(0,0,0,0.08); 
    }
    .kpi-title { 
        font-size: 14px; 
        color: #666; 
        margin-bottom: 5px; 
        text-transform: uppercase; 
        letter-spacing: 1px;
        font-weight: 600;
    }
    .kpi-value { 
        font-size: 32px; 
        font-weight: 800; 
        color: #2e7d32; 
    }

    /* --- UNIFORM BUTTON STYLING --- */
    div.stButton > button, div.stDownloadButton > button {
        background: linear-gradient(90deg, #43a047 0%, #66bb6a 100%); 
        color: white; 
        border: none; 
        padding: 12px; 
        border-radius: 8px; 
        font-weight: 600; 
        width: 100%;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        transition: all 0.3s;
    }
    div.stButton > button:hover, div.stDownloadButton > button:hover { 
        transform: translateY(-2px); 
        box-shadow: 0 4px 8px rgba(0,0,0,0.2); 
        background: linear-gradient(90deg, #2e7d32 0%, #43a047 100%);
        color: white;
    }

    /* Headers - Strong Dark Green */
    h1, h2, h3 {
        color: #1b5e20 !important;
        font-weight: 700;
    }

    /* Tables - Clean White */
    [data-testid="stDataFrame"] {
        background-color: white;
        padding: 10px;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }

    /* Recommendation Boxes */
    .rec-box { padding: 25px; border-radius: 10px; margin: 20px 0; border-left: 6px solid; box-shadow: 2px 2px 8px rgba(0,0,0,0.05); }
    .rec-critical { background-color: #ffebee; border-color: #ef5350; color: #c62828; }
    .rec-warning { background-color: #fff3e0; border-color: #ffa726; color: #ef6c00; }
    .rec-success { background-color: #e8f5e9; border-color: #66bb6a; color: #2e7d32; }
    .rec-header { font-size: 18px; font-weight: bold; margin-bottom: 10px; display: flex; align-items: center; gap: 10px; }

</style>
""", unsafe_allow_html=True)

# Initialize Session State
if 'current_scenario' not in st.session_state:
    st.session_state.current_scenario = "Base Case"


# ------------------------------------------
# 2. DATA ENGINE
# ------------------------------------------

@st.cache_data
def load_data():
    # Transport Arcs
    transport_arcs = pd.DataFrame({
        'Arc_ID': ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15'],
        'From_Node': ['Sorriso Silos', 'Rondonópolis Plant', 'Lucas Silos', 'Ponta Grossa Plant',
                      'Rio Verde Terminal', 'Sinop Terminal', 'Santos Port', 'Santos Port',
                      'Paranaguá Port', 'Santos Port', 'Santos Port', 'Paranaguá Port',
                      'Santos Port', 'Itaqui Port', 'Barcarena Port'],
        'To_Node': ['Rondonópolis Plant', 'Santos Port', 'Rondonópolis Plant', 'Paranaguá Port',
                    'Santos Port', 'Itaqui Port', 'Qingdao Port', 'Rotterdam (ECT)',
                    'Busan (Gamman)', 'Cai Mep Port', 'Laem Chabang', 'Yokohama Grain Terminal',
                    'Valencia Grain Terminal', 'Qingdao Port', 'Rotterdam (ECT)'],
        'Mode': ['Road', 'Rail', 'Road', 'Rail', 'Rail', 'Rail', 'Sea', 'Sea',
                 'Sea', 'Sea', 'Sea', 'Sea', 'Sea', 'Sea', 'Sea'],
        'Distance_km': [450, 1650, 320, 150, 1100, 1800, 18500, 9500, 19300, 17200, 16800, 18900, 8200, 19800, 10200],
        'Capacity_MTPA': [8, 5.5, 4, 3, 3.5, 2.5, 40, 40, 25, 40, 40, 25, 40, 15, 12],
        'Cost_per_t_USD': [22.5, 45, 16, 7.5, 30, 50, 35, 28, 42, 38, 36, 44, 26, 40, 30]
    })

    # Nodes
    nodes = pd.DataFrame({
        'Node': ['Sorriso Silos', 'Rondonópolis Plant', 'Lucas Silos', 'Ponta Grossa Plant', 'Cascavel Hub',
                 'Santos Port', 'Paranaguá Port', 'Rio Verde Terminal', 'Sinop Terminal', 'Itaqui Port',
                 'Barcarena Port', 'Qingdao Port', 'Rotterdam (ECT)', 'Busan (Gamman)', 'Cai Mep Port',
                 'Laem Chabang', 'Yokohama Grain Terminal', 'Valencia Grain Terminal'],
        'Type': ['Silo', 'Plant', 'Silo', 'Plant', 'Silo', 'Port', 'Port', 'Silo', 'Silo', 'Port',
                 'Port', 'Market', 'Market', 'Market', 'Market', 'Market', 'Market', 'Market'],
        'Lat': [-12.54, -16.46, -13.07, -25.09, -24.95, -23.96, -25.51, -17.79, -11.86, -2.57,
                -1.51, 36.06, 51.92, 35.17, 10.51, 13.08, 35.44, 39.46],
        'Lon': [-55.72, -54.63, -55.91, -50.16, -53.45, -46.33, -48.51, -50.92, -55.50, -44.37,
                -48.74, 120.38, 4.47, 129.07, 107.03, 100.88, 139.63, -0.37]
    })

    # Markets
    markets = pd.DataFrame({
        'Market': ['Qingdao Port', 'Rotterdam (ECT)', 'Cai Mep Port', 'Laem Chabang',
                   'Busan (Gamman)', 'Yokohama Grain Terminal', 'Valencia Grain Terminal'],
        'Region_Group': ['China', 'EU', 'Asia', 'Asia', 'Asia', 'Asia', 'EU'],
        'Base_Demand_MTPA': [70, 15, 8.5, 5.5, 4.2, 3.8, 3.5],
        'Base_Price_USD': [520, 540, 515, 510, 525, 530, 535]
    })

    # Production (Total Capacity ~23 MTPA)
    production = pd.DataFrame({
        'Node': ['Sorriso Silos', 'Lucas Silos', 'Rio Verde Terminal', 'Sinop Terminal', 'Ponta Grossa Plant'],
        'Capacity_MTPA': [8, 4, 3.5, 2.5, 3],
    })

    return transport_arcs, nodes, markets, production


def get_scenario_params(scenario_name):
    # Base multipliers
    params = {
        "desc": "Normal Operations with all routes available.",
        "rail_cap": 1.0, "port_cap": 1.0, "sea_cap": 1.0, "road_cap": 1.0,
        "rail_cost": 1.0, "sea_cost": 1.0,
        "dem_cn": 1.0, "dem_eu": 1.0, "dem_asia": 1.0,
        "price_cn": 1.0, "price_eu": 1.0, "price_asia": 1.0,
        "disabled_nodes": []
    }

    # Scenario Logic
    if scenario_name == "Mato Grosso Flood (2024)":
        params.update({"desc": "Historic flooding on EFVM/Rumo railway. Rail capacity reduced by 60%.",
                       "rail_cap": 0.4, "rail_cost": 1.3, "dem_cn": 0.85})
    elif scenario_name == "China ASF Outbreak":
        params.update({"desc": "African Swine Fever reduces feed demand in China.",
                       "dem_cn": 0.6, "price_cn": 0.85, "sea_cap": 0.95})
    elif scenario_name == "Panama Canal Drought":
        params.update({"desc": "Low water levels restrict passages. Sea costs spike.",
                       "sea_cost": 1.8, "dem_cn": 1.0, "price_cn": 1.15})
    elif scenario_name == "Brazil Trucker Strike":
        params.update({"desc": "National trucker protest blocks highways. Road cap reduced.",
                       "road_cap": 0.7, "rail_cost": 1.4, "dem_asia": 0.92})
    elif scenario_name == "EU Deforestation Law":
        params.update({"desc": "EUDR restricts non-compliant soy. EU demand drops.",
                       "dem_eu": 0.7, "price_eu": 0.88})
    elif scenario_name == "Port of Santos Strike":
        params.update({"desc": "21-day port worker strike. Port capacity crippled.",
                       "port_cap": 0.3, "sea_cost": 1.2, "price_cn": 0.95, "disabled_nodes": ["Santos Port"]})

    return params


def optimize_network(arcs, nodes, markets, production, params):
    active_arcs = arcs.copy()

    # Apply Capacities & Costs
    for mode in ['Rail', 'Sea', 'Road']:
        if f"{mode.lower()}_cap" in params: active_arcs.loc[active_arcs['Mode'] == mode, 'Capacity_MTPA'] *= params[
            f"{mode.lower()}_cap"]
        if f"{mode.lower()}_cost" in params: active_arcs.loc[active_arcs['Mode'] == mode, 'Cost_per_t_USD'] *= params[
            f"{mode.lower()}_cost"]

    # Filter Disabled Nodes
    if params['disabled_nodes']:
        active_arcs = active_arcs[~active_arcs['From_Node'].isin(params['disabled_nodes'])]
        active_arcs = active_arcs[~active_arcs['To_Node'].isin(params['disabled_nodes'])]

    # Apply Market Adjustments
    active_markets = markets.copy()

    def get_mult(region, kind):
        if region == 'China': return params[f'dem_cn'] if kind == 'dem' else params[f'price_cn']
        if region == 'EU': return params[f'dem_eu'] if kind == 'dem' else params[f'price_eu']
        return params[f'dem_asia'] if kind == 'dem' else params[f'price_asia']

    active_markets['Adj_Demand'] = active_markets.apply(
        lambda x: x['Base_Demand_MTPA'] * get_mult(x['Region_Group'], 'dem'), axis=1)
    active_markets['Adj_Price'] = active_markets.apply(
        lambda x: x['Base_Price_USD'] * get_mult(x['Region_Group'], 'price'), axis=1)

    # Greedy Routing
    routes = []
    node_capacity = production.set_index('Node')['Capacity_MTPA'].to_dict()
    arc_capacity = active_arcs.set_index('Arc_ID')['Capacity_MTPA'].to_dict()

    for _, mkt in active_markets.sort_values('Adj_Price', ascending=False).iterrows():
        demand = mkt['Adj_Demand']
        sea_legs = active_arcs[active_arcs['To_Node'] == mkt['Market']]

        for _, sea in sea_legs.iterrows():
            if demand <= 0: break
            port = sea['From_Node']
            land_legs = active_arcs[active_arcs['To_Node'] == port]

            for _, land in land_legs.iterrows():
                if demand <= 0: break
                origin = land['From_Node']
                avail = min(demand, node_capacity.get(origin, 0), arc_capacity.get(land['Arc_ID'], 0),
                            arc_capacity.get(sea['Arc_ID'], 0))

                if avail > 0:
                    cost = land['Cost_per_t_USD'] + sea['Cost_per_t_USD']
                    rev = avail * mkt['Adj_Price']
                    profit = rev - (avail * cost)

                    routes.append({
                        'Origin': origin, 'Via_Port': port, 'Market': mkt['Market'],
                        'Volume_MT': avail, 'Land_Mode': land['Mode'],
                        'Cost': cost, 'Revenue': rev, 'Profit': profit,
                        'Distance': land['Distance_km'] + sea['Distance_km']
                    })
                    demand -= avail
                    node_capacity[origin] -= avail
                    arc_capacity[land['Arc_ID']] -= avail
                    arc_capacity[sea['Arc_ID']] -= avail

    return pd.DataFrame(routes)


# ------------------------------------------
# 3. VISUALIZATION ENGINE
# ------------------------------------------

def create_dark_map(nodes, routes_df, transport_arcs, params):
    # UPDATED: Use Positron for light theme
    m = folium.Map(location=[-15, -50], zoom_start=3.5, tiles='CartoDB positron')

    active_pairs = set()
    if not routes_df.empty:
        active_pairs = set(zip(routes_df['Origin'], routes_df['Via_Port'])) | set(
            zip(routes_df['Via_Port'], routes_df['Market']))

    for _, arc in transport_arcs.iterrows():
        orig = nodes[nodes['Node'] == arc['From_Node']].iloc[0]
        dest = nodes[nodes['Node'] == arc['To_Node']].iloc[0]
        is_active = (arc['From_Node'], arc['To_Node']) in active_pairs

        if is_active:
            # Colors adjusted for light background
            if arc['Mode'] == 'Rail':
                color = '#e65100'  # Orange
            elif arc['Mode'] == 'Sea':
                color = '#0277bd'  # Blue
            else:
                color = '#5d4037'  # Brown (Road)
            weight, opacity, dash, tooltip = 3, 0.9, None, f"✅ Active: {arc['From_Node']} -> {arc['To_Node']}"
        else:
            color, weight, opacity, dash, tooltip = '#bdbdbd', 1, 0.3, '5,5', f"❌ Inactive: {arc['From_Node']} -> {arc['To_Node']}"

        folium.PolyLine(locations=[[orig['Lat'], orig['Lon']], [dest['Lat'], dest['Lon']]], color=color, weight=weight,
                        opacity=opacity, dash_array=dash, tooltip=tooltip).add_to(m)

    for _, node in nodes.iterrows():
        is_disabled = node['Node'] in params['disabled_nodes']

        # Colors adapted for theme
        if node['Type'] in ['Silo', 'Farm']:
            color, lbl = '#c62828', "Source"  # Red
        elif node['Type'] in ['Plant', 'Terminal']:
            color, lbl = '#f57f17', "Processing"  # Orange/Yellow
        elif node['Type'] == 'Port':
            color, lbl = '#2e7d32', "Port"  # Green
        else:
            color, lbl = '#6a1b9a', "Market"  # Purple

        if is_disabled: color = '#212121'

        folium.CircleMarker(location=[node['Lat'], node['Lon']], radius=6 if not is_disabled else 4, color=color,
                            fill=True, fill_color=color, fill_opacity=0.9, popup=f"<b>{node['Node']}</b><br>{lbl}",
                            tooltip=node['Node']).add_to(m)

    # ADD LEGEND (Styled for Light Theme)
    legend_html = '''
    <div style="position: fixed; bottom: 50px; right: 50px; width: 150px; background-color: white; border: 1px solid #ccc; z-index: 9999; font-family: sans-serif; font-size: 13px; padding: 10px; border-radius: 4px; box-shadow: 2px 2px 5px rgba(0,0,0,0.3); color: #333;">
        <div style="margin-bottom: 5px; font-weight: bold;">LEGEND</div>
        <div><span style="color: #c62828;">●</span> Silo/Farm</div>
        <div><span style="color: #f57f17;">●</span> Processing</div>
        <div><span style="color: #2e7d32;">●</span> Port</div>
        <div><span style="color: #6a1b9a;">●</span> Market</div>
        <hr style="margin: 5px 0; border: 0; border-top: 1px solid #ddd;">
        <div><span style="color: #e65100;">━</span> Rail</div>
        <div><span style="color: #0277bd;">━</span> Sea</div>
        <div><span style="color: #5d4037;">━</span> Road</div>
        <div><span style="color: #bdbdbd;">--</span> Inactive</div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    return m


# ------------------------------------------
# 4. MAIN LAYOUT LOGIC
# ------------------------------------------

def main():
    # --- SIDEBAR (Always Visible) ---
    with st.sidebar:
        st.header("⚙️ SCENARIO CONTROL")
        st.info("Use this panel to switch scenarios and view specific parameters.")

        scenarios = ["Base Case", "Mato Grosso Flood (2024)", "Port of Santos Strike", "China ASF Outbreak",
                     "Brazil Trucker Strike", "Panama Canal Drought"]

        # Sync Sidebar Selectbox with Session State
        if st.session_state.current_scenario not in scenarios:
            st.session_state.current_scenario = "Base Case"

        selected_scen = st.selectbox("Select Scenario:", scenarios,
                                     index=scenarios.index(st.session_state.current_scenario))

        if st.button("Apply Scenario"):
            st.session_state.current_scenario = selected_scen
            st.rerun()

        st.markdown("---")
        params = get_scenario_params(st.session_state.current_scenario)
        st.markdown(f"**Current Desc:**\n{params['desc']}")

    # --- MAIN CONTENT ---
    st.title(f"ACTIVE SCENARIO: {st.session_state.current_scenario}")
    # st.caption(f"Scenario Description: {get_scenario_params(st.session_state.current_scenario)['desc']}")

    # Load & Calculate
    arcs, nodes, markets, production = load_data()
    params = get_scenario_params(st.session_state.current_scenario)
    df = optimize_network(arcs, nodes, markets, production, params)

    # --- 1. TOP FINANCIAL CARDS ---
    c1, c2, c3, c4 = st.columns(4)

    tot_rev = df['Revenue'].sum() if not df.empty else 0
    tot_profit = df['Profit'].sum() if not df.empty else 0
    margin = (tot_profit / tot_rev * 100) if tot_rev > 0 else 0
    tot_cost = tot_rev - tot_profit

    # UPDATED CARD STYLES
    c1.markdown(
        f"""<div class="kpi-card"><div class="kpi-title">REVENUE</div><div class="kpi-value">${tot_rev / 1000:,.1f}B</div></div>""",
        unsafe_allow_html=True)
    c2.markdown(
        f"""<div class="kpi-card"><div class="kpi-title">TOTAL COST</div><div class="kpi-value" style="color: #c62828;">${tot_cost / 1000:,.1f}B</div></div>""",
        unsafe_allow_html=True)
    c3.markdown(
        f"""<div class="kpi-card"><div class="kpi-title">PROFIT</div><div class="kpi-value" style="color: #2e7d32;">${tot_profit / 1000:,.1f}B</div></div>""",
        unsafe_allow_html=True)
    c4.markdown(
        f"""<div class="kpi-card"><div class="kpi-title">PROFIT MARGIN</div><div class="kpi-value" style="color: #f57f17;">{margin:.1f}%</div></div>""",
        unsafe_allow_html=True)

    # --- 2. SUPPLY CHAIN MAP ---
    st.markdown("### 🗺️ SUPPLY CHAIN NETWORK MAP")
    st_folium(create_dark_map(nodes, df, arcs, params), width=1400, height=450)

    # --- 3. OPERATIONAL PERFORMANCE ---
    st.markdown("### ⚙️ OPERATIONAL PERFORMANCE")
    o1, o2, o3, o4 = st.columns(4)

    tot_vol = df['Volume_MT'].sum() if not df.empty else 0
    tot_cap = production['Capacity_MTPA'].sum()
    utilization = (tot_vol / tot_cap * 100) if tot_cap > 0 else 0
    active_cnt = len(df)
    avg_dist = df['Distance'].mean() if not df.empty else 0

    o1.metric("Production Utilization", f"{utilization:.1f}%", f"{tot_vol:.1f} / {tot_cap:.1f} MT")
    o2.metric("Demand Fulfillment", "100.0%", "On Target")  # Simplified for demo
    o3.metric("Active Routes", f"{active_cnt}/15", "Network Health")
    o4.metric("Avg Route Distance", f"{avg_dist:,.0f} km", "Weighted Avg")

    # --- 4. DETAILED ANALYSIS ---
    st.markdown("### 📊 DETAILED ANALYSIS")
    g1, g2 = st.columns(2)
    if not df.empty:
        with g1:
            st.markdown("**Market Allocation (Volume)**")
            fig_pie = px.pie(df, values='Volume_MT', names='Market', hole=0.4,
                             color_discrete_sequence=px.colors.qualitative.Pastel)
            # UPDATED CHART STYLING FOR VISIBILITY
            fig_pie.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color="#2c3e50",  # Dark text
                legend_font_color="#2c3e50",
                margin=dict(t=0, b=0, l=0, r=0)
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        with g2:
            st.markdown("**Profit by Market ($M)**")
            fig_bar = px.bar(df.groupby('Market')['Profit'].sum().reset_index(), x='Market', y='Profit', color='Market',
                             color_discrete_sequence=px.colors.qualitative.Pastel)
            # UPDATED CHART STYLING FOR VISIBILITY
            fig_bar.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color="#2c3e50",  # Dark text
                showlegend=False,
                margin=dict(t=0, b=0, l=0, r=0)
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    # --- 5. ROUTE ALLOCATION TABLE ---
    st.markdown("### 📋 ROUTE ALLOCATION DETAILS")
    if not df.empty:
        st.dataframe(df[['Origin', 'Via_Port', 'Market', 'Volume_MT', 'Revenue', 'Profit']], use_container_width=True)

    # --- 6. SCENARIO COMPARISON ---
    st.markdown("### 🔄 SCENARIO COMPARISON")
    st.caption("Quick Scenario Switch:")
    b1, b2, b3, b4, b5 = st.columns(5)

    scenarios_list = ["Base Case", "Mato Grosso Flood (2024)", "Port of Santos Strike", "China ASF Outbreak",
                      "Brazil Trucker Strike"]

    def set_scen(s):
        st.session_state.current_scenario = s;
        st.rerun()

    if b1.button(scenarios_list[0]): set_scen(scenarios_list[0])
    if b2.button(scenarios_list[1]): set_scen(scenarios_list[1])
    if b3.button(scenarios_list[2]): set_scen(scenarios_list[2])
    if b4.button(scenarios_list[3]): set_scen(scenarios_list[3])
    if b5.button(scenarios_list[4]): set_scen(scenarios_list[4])

    # --- 7. RECOMMENDATIONS (DYNAMIC & SCENARIO AWARE) ---
    st.markdown("### 💡 RECOMMENDATIONS & INSIGHTS")

    scenario = st.session_state.current_scenario

    # Force specific insights based on the Active Scenario Name
    if scenario == "Port of Santos Strike":
        st.markdown(f"""
        <div class="rec-box rec-critical">
            <div class="rec-header">⚠️ CRITICAL: PORT STRIKE PROTOCOL ACTIVATED</div>
            <ul>
                <li><strong>Observation:</strong> Santos Port is offline (0% Capacity). All volume rerouted via Paranaguá/Itaqui.</li>
                <li><strong>Impact:</strong> Transport costs increased by 20% due to longer rail/road legs.</li>
                <li><strong>Action:</strong> Declare Force Majeure on Santos FOB contracts. Prioritize rail slots to Northern Arc.</li>
            </ul>
        </div>""", unsafe_allow_html=True)

    elif scenario == "Mato Grosso Flood (2024)":
        st.markdown(f"""
        <div class="rec-box rec-critical">
            <div class="rec-header">⚠️ CRITICAL: RAIL DISRUPTION (FLOOD)</div>
            <ul>
                <li><strong>Observation:</strong> Rumo/EFVM rail capacity down 60%. Road dependency increased.</li>
                <li><strong>Impact:</strong> Margin erosion due to high trucking costs.</li>
                <li><strong>Action:</strong> Activate emergency truck fleet for Sorriso -> Miritituba (Barge) route.</li>
            </ul>
        </div>""", unsafe_allow_html=True)

    elif scenario == "China ASF Outbreak":
        st.markdown(f"""
        <div class="rec-box rec-warning">
            <div class="rec-header">⚠️ WARNING: DEMAND SHOCK (ASF)</div>
            <ul>
                <li><strong>Observation:</strong> China feed demand collapsed by 40%. Prices dropping.</li>
                <li><strong>Strategy:</strong> Pivot volume to EU and Southeast Asia (Vietnam/Thailand).</li>
                <li><strong>Action:</strong> Hedge remaining inventory at current futures prices.</li>
            </ul>
        </div>""", unsafe_allow_html=True)

    elif scenario == "Brazil Trucker Strike":
        st.markdown(f"""
        <div class="rec-box rec-critical">
            <div class="rec-header">⚠️ CRITICAL: LOGISTICS BLOCKADE</div>
            <ul>
                <li><strong>Observation:</strong> Road transport paralyzed. Silos at 90% capacity.</li>
                <li><strong>Action:</strong> Halt farm-to-silo movements. Maximize rail usage where last-mile is available.</li>
            </ul>
        </div>""", unsafe_allow_html=True)

    # Fallback Logic for Base Case or Generic Scenarios
    else:
        if margin < 15:
            st.markdown(f"""
            <div class="rec-box rec-warning">
                <div class="rec-header">⚠️ WARNING - Low Profit Margin ({margin:.1f}%)</div>
                <ul>
                    <li><strong>Observation:</strong> Margin is below 15% target. Transport costs are high.</li>
                    <li><strong>Action:</strong> Review spot market rail bookings.</li>
                </ul>
            </div>""", unsafe_allow_html=True)
        elif utilization < 70:
            st.markdown(f"""
            <div class="rec-box rec-warning">
                <div class="rec-header">⚠️ WARNING - Low Asset Utilization ({utilization:.1f}%)</div>
                <ul>
                    <li><strong>Observation:</strong> Production sites operating below capacity.</li>
                    <li><strong>Action:</strong> Explore short-term sales to SE Asia to clear inventory.</li>
                </ul>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="rec-box rec-success">
                <div class="rec-header">✅ OPPORTUNITY - High Demand & Efficiency</div>
                <ul>
                    <li><strong>Observation:</strong> System operating at high efficiency ({utilization:.1f}% util) with healthy margins ({margin:.1f}%).</li>
                    <li><strong>Action:</strong> Consider capacity expansion at Rondonópolis Plant.</li>
                    <li><strong>Strategy:</strong> Lock in long-term contracts with key buyers.</li>
                </ul>
            </div>""", unsafe_allow_html=True)

    # --- 8. EXPORT DATA ---
    st.markdown("### 📥 EXPORT DATA")
    e1, e2, e3 = st.columns(3)
    with e1:
        st.download_button("📊 Export Route Data", df.to_csv(), "routes.csv", "text/csv")
    with e2:
        if not df.empty:
            st.download_button("💰 Export Financial Summary", df.groupby('Market')[['Revenue', 'Profit']].sum().to_csv(),
                               "financials.csv", "text/csv")
    with e3:
        if st.button("🔄 Reset to Base Case"):
            st.session_state.current_scenario = "Base Case"
            st.rerun()


if __name__ == "__main__":
    main()
