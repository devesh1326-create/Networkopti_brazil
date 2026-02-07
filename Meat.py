# ==========================================
# BOVINE MEAT SUPPLY CHAIN CONTROL TOWER
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
    page_title="Bovine Meat Control Tower",
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

if 'current_scenario' not in st.session_state:
    st.session_state.current_scenario = "Base Case"


# ------------------------------------------
# 2. DATA ENGINE
# ------------------------------------------

@st.cache_data
def load_data():
    # Simplified arc structure for bovine meat
    transport_arcs = pd.DataFrame({
        'Arc_ID': ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15'],
        'From_Node': ['Mato Grosso Ranches', 'Mato Grosso do Sul Ranches', 'Campo Grande Plant',
                      'Goiás Ranches', 'Rio Verde Plant', 'Paraná Ranches',
                      'Santos Cold Port', 'Santos Cold Port', 'Santos Cold Port',
                      'Paranaguá Cold Port', 'Santos Cold Port', 'Itaqui Cold Port',
                      'Santos Cold Port', 'Santos Cold Port', 'Rondonópolis Plant'],
        'To_Node': ['Campo Grande Plant', 'Campo Grande Plant', 'Santos Cold Port',
                    'Rio Verde Plant', 'Santos Cold Port', 'Paranaguá Cold Port',
                    'Shanghai Port', 'Hong Kong Port', 'Jebel Ali Port',
                    'Rotterdam Cold Store', 'Los Angeles Port', 'Egypt Port Said',
                    'Santiago Port', 'Moscow Port', 'Santos Cold Port'],
        'Mode': ['Road', 'Road', 'Refrigerated Truck', 'Road', 'Refrigerated Truck', 'Road',
                 'Refrigerated Ship', 'Refrigerated Ship', 'Refrigerated Ship',
                 'Refrigerated Ship', 'Refrigerated Ship', 'Refrigerated Ship',
                 'Refrigerated Ship', 'Refrigerated Ship', 'Refrigerated Truck'],
        'Distance_km': [700, 200, 1100, 300, 1000, 500, 18500, 18200, 11500, 10000, 10500, 9200, 4500, 15800, 400],
        'Capacity_MTPA': [1.2, 1.5, 2, 0.7, 0.8, 0.5, 1.5, 1, 0.8, 0.6, 0.4, 0.4, 0.3, 0.2, 0.7],
        'Cost_per_t_USD': [85, 35, 120, 45, 110, 60, 180, 175, 150, 140, 160, 130, 170, 190, 50]
    })

    nodes = pd.DataFrame({
        'Node': ['Mato Grosso Ranches', 'Mato Grosso do Sul Ranches', 'Goiás Ranches', 'Paraná Ranches',
                 'Campo Grande Plant', 'Rio Verde Plant', 'Rondonópolis Plant',
                 'Santos Cold Port', 'Paranaguá Cold Port', 'Itaqui Cold Port',
                 'Shanghai Port', 'Hong Kong Port', 'Jebel Ali Port', 'Rotterdam Cold Store',
                 'Los Angeles Port', 'Egypt Port Said', 'Santiago Port', 'Moscow Port'],
        'Type': ['Ranch', 'Ranch', 'Ranch', 'Ranch', 'Plant', 'Plant', 'Plant',
                 'Port', 'Port', 'Port', 'Market', 'Market', 'Market', 'Market', 'Market', 'Market', 'Market',
                 'Market'],
        'Lat': [-12.0, -20.5, -16.0, -25.0, -20.45, -17.8, -16.47,
                -23.96, -25.43, -2.53, 31.23, 22.32, 25.27, 51.92, 33.74, 31.26, -33.45, 55.75],
        'Lon': [-55.5, -54.6, -49.3, -51.5, -54.64, -50.92, -54.62,
                -46.33, -48.43, -44.30, 121.47, 114.17, 55.30, 4.47, -118.27, 32.29, -70.67, 37.62]
    })

    markets = pd.DataFrame({
        'Market': ['Shanghai Port', 'Hong Kong Port', 'Jebel Ali Port', 'Rotterdam Cold Store',
                   'Los Angeles Port', 'Egypt Port Said', 'Santiago Port', 'Moscow Port'],
        'Region_Group': ['Asia', 'Asia', 'Middle East', 'Europe', 'Americas', 'Africa', 'Americas', 'Europe'],
        'Base_Demand_MTPA': [1.8, 1.0, 0.8, 0.6, 0.4, 0.4, 0.3, 0.2],
        'Base_Price_USD': [5500, 5600, 5200, 5800, 5700, 5100, 5400, 5300]
    })

    production = pd.DataFrame({
        'Node': ['Mato Grosso Ranches', 'Mato Grosso do Sul Ranches', 'Goiás Ranches', 'Paraná Ranches'],
        'Capacity_MTPA': [1.8, 1.5, 1.0, 0.6],
    })

    return transport_arcs, nodes, markets, production


def get_scenario_params(scenario_name):
    params = {
        "desc": "Normal Operations with all routes available.",
        "road_cap": 1.0, "port_cap": 1.0, "sea_cap": 1.0,
        "road_cost": 1.0, "sea_cost": 1.0,
        "dem_cn": 1.0, "dem_eu": 1.0, "dem_me": 1.0, "dem_usa": 1.0,
        "price_cn": 1.0, "price_eu": 1.0, "price_me": 1.0, "price_usa": 1.0,
        "disabled_nodes": []
    }

    if scenario_name == "Foot-and-Mouth Outbreak":
        params.update({"desc": "FMD outbreak in Mato Grosso. Export bans to premium markets.",
                       "disabled_nodes": ["Mato Grosso Ranches"], "dem_cn": 1.1, "price_cn": 1.05})

    elif scenario_name == "JBS Plant Sanitation Issue":
        params.update({"desc": "Health inspection shuts down Campo Grande Plant temporarily.",
                       "disabled_nodes": ["Campo Grande Plant"], "road_cap": 0.8})

    elif scenario_name == "China Beef Ban (Political)":
        params.update({"desc": "Diplomatic tension leads to temporary China import suspension.",
                       "dem_cn": 0.1, "price_cn": 0.9, "dem_me": 1.15, "dem_eu": 1.1})

    elif scenario_name == "Brazil Trucker Strike":
        params.update({"desc": "National highway blockade disrupts cattle transport to plants.",
                       "road_cap": 0.7, "road_cost": 1.3})

    elif scenario_name == "Global Container Shortage":
        params.update({"desc": "Refrigerated container shortage drives up sea freight costs.",
                       "sea_cost": 1.4, "port_cap": 0.85})

    return params


def optimize_network(arcs, nodes, markets, production, params):
    active_arcs = arcs.copy()

    # Apply Capacities & Costs
    for mode in ['Road', 'Refrigerated Truck', 'Refrigerated Ship']:
        if mode in ['Road', 'Refrigerated Truck']:
            active_arcs.loc[active_arcs['Mode'] == mode, 'Capacity_MTPA'] *= params['road_cap']
            active_arcs.loc[active_arcs['Mode'] == mode, 'Cost_per_t_USD'] *= params['road_cost']
        elif mode == 'Refrigerated Ship':
            active_arcs.loc[active_arcs['Mode'] == mode, 'Capacity_MTPA'] *= params['sea_cap']
            active_arcs.loc[active_arcs['Mode'] == mode, 'Cost_per_t_USD'] *= params['sea_cost']

    # Filter Disabled Nodes
    if params['disabled_nodes']:
        active_arcs = active_arcs[~active_arcs['From_Node'].isin(params['disabled_nodes'])]
        active_arcs = active_arcs[~active_arcs['To_Node'].isin(params['disabled_nodes'])]

    # Apply Market Adjustments
    active_markets = markets.copy()

    def get_mult(region, kind):
        if region == 'Asia': return params[f'dem_cn'] if kind == 'dem' else params[f'price_cn']
        if region == 'Europe': return params[f'dem_eu'] if kind == 'dem' else params[f'price_eu']
        if region == 'Middle East': return params[f'dem_me'] if kind == 'dem' else params[f'price_me']
        if region == 'Americas': return params[f'dem_usa'] if kind == 'dem' else params[f'price_usa']
        return 1.0

    active_markets['Adj_Demand'] = active_markets.apply(
        lambda x: x['Base_Demand_MTPA'] * get_mult(x['Region_Group'], 'dem'), axis=1)
    active_markets['Adj_Price'] = active_markets.apply(
        lambda x: x['Base_Price_USD'] * get_mult(x['Region_Group'], 'price'), axis=1)

    # Adjust production for disabled nodes
    active_production = production.copy()
    if params['disabled_nodes']:
        active_production = active_production[~active_production['Node'].isin(params['disabled_nodes'])]

    # Greedy Routing
    routes = []
    node_capacity = active_production.set_index('Node')['Capacity_MTPA'].to_dict()
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

                # Handle multi-hop
                origin_cap = node_capacity.get(origin, 0)
                if origin_cap == 0:
                    upstream_arcs = active_arcs[active_arcs['To_Node'] == origin]
                    for _, up_arc in upstream_arcs.iterrows():
                        upstream_origin = up_arc['From_Node']
                        origin_cap = max(origin_cap, node_capacity.get(upstream_origin, 0))
                        if origin_cap > 0:
                            origin = upstream_origin
                            break

                avail = min(demand, origin_cap, arc_capacity.get(land['Arc_ID'], 0),
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
                    if origin in node_capacity:
                        node_capacity[origin] -= avail
                    arc_capacity[land['Arc_ID']] -= avail
                    arc_capacity[sea['Arc_ID']] -= avail

    return pd.DataFrame(routes)


# ------------------------------------------
# 3. VISUALIZATION ENGINE
# ------------------------------------------

def create_map(nodes, routes_df, transport_arcs, params):
    # UPDATED: Use Positron for light theme
    m = folium.Map(location=[-15, -50], zoom_start=2.5, tiles='CartoDB positron')

    active_pairs = set()
    if not routes_df.empty:
        active_pairs = set(zip(routes_df['Origin'], routes_df['Via_Port'])) | set(
            zip(routes_df['Via_Port'], routes_df['Market']))

    for _, arc in transport_arcs.iterrows():
        try:
            orig = nodes[nodes['Node'] == arc['From_Node']].iloc[0]
            dest = nodes[nodes['Node'] == arc['To_Node']].iloc[0]
        except:
            continue

        is_active = (arc['From_Node'], arc['To_Node']) in active_pairs

        if is_active:
            # Colors adjusted for light background
            if arc['Mode'] in ['Road', 'Refrigerated Truck']:
                color = '#e65100'  # Orange/Reddish for land
            elif arc['Mode'] == 'Refrigerated Ship':
                color = '#0277bd'  # Strong Blue for sea
            else:
                color = '#f57f17'
            weight, opacity, dash, tooltip = 3, 0.9, None, f"✅ Active: {arc['From_Node']} -> {arc['To_Node']}"
        else:
            color, weight, opacity, dash, tooltip = '#bdbdbd', 1, 0.3, '5,5', f"❌ Inactive"

        folium.PolyLine(locations=[[orig['Lat'], orig['Lon']], [dest['Lat'], dest['Lon']]], color=color, weight=weight,
                        opacity=opacity, dash_array=dash, tooltip=tooltip).add_to(m)

    for _, node in nodes.iterrows():
        is_disabled = node['Node'] in params['disabled_nodes']

        # New Color Scheme
        if node['Type'] == 'Ranch':
            color, lbl = '#5d4037', "Ranch"  # Brown
        elif node['Type'] == 'Plant':
            color, lbl = '#e65100', "Processing Plant"  # Orange
        elif node['Type'] == 'Port':
            color, lbl = '#2e7d32', "Cold Port"  # Green
        else:
            color, lbl = '#6a1b9a', "Market"  # Purple

        if is_disabled: color = '#212121'

        folium.CircleMarker(location=[node['Lat'], node['Lon']], radius=6 if not is_disabled else 4, color=color,
                            fill=True, fill_color=color, fill_opacity=0.9, popup=f"<b>{node['Node']}</b><br>{lbl}",
                            tooltip=node['Node']).add_to(m)

    # ADD LEGEND
    legend_html = '''
    <div style="position: fixed; bottom: 50px; right: 50px; width: 180px; background-color: white; border: 1px solid #ccc; z-index: 9999; font-family: sans-serif; font-size: 13px; padding: 10px; border-radius: 4px; box-shadow: 2px 2px 5px rgba(0,0,0,0.3); color: #333;">
        <div style="margin-bottom: 5px; font-weight: bold;">LEGEND</div>
        <div><span style="color: #5d4037;">●</span> Ranch</div>
        <div><span style="color: #e65100;">●</span> Processing Plant</div>
        <div><span style="color: #2e7d32;">●</span> Cold Port</div>
        <div><span style="color: #6a1b9a;">●</span> Market</div>
        <hr style="margin: 5px 0; border: 0; border-top: 1px solid #ddd;">
        <div><span style="color: #e65100;">━</span> Road/Truck</div>
        <div><span style="color: #0277bd;">━</span> Reefer Ship</div>
        <div><span style="color: #bdbdbd;">--</span> Inactive Route</div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    return m


# ------------------------------------------
# 4. MAIN LAYOUT LOGIC
# ------------------------------------------

def main():
    with st.sidebar:
        st.header("⚙️ SCENARIO CONTROL")
        st.info("Use this panel to switch scenarios and view specific parameters.")

        scenarios = ["Base Case", "Foot-and-Mouth Outbreak", "JBS Plant Sanitation Issue",
                     "China Beef Ban (Political)", "Brazil Trucker Strike", "Global Container Shortage"]

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

    st.title(f"🥩 BOVINE MEAT SUPPLY CHAIN CONTROL TOWER")
    st.subheader(f"ACTIVE SCENARIO: {st.session_state.current_scenario}")

    arcs, nodes, markets, production = load_data()
    params = get_scenario_params(st.session_state.current_scenario)
    df = optimize_network(arcs, nodes, markets, production, params)

    # --- KPI CARDS ---
    c1, c2, c3, c4 = st.columns(4)
    tot_rev = df['Revenue'].sum() if not df.empty else 0
    tot_profit = df['Profit'].sum() if not df.empty else 0
    margin = (tot_profit / tot_rev * 100) if tot_rev > 0 else 0
    tot_cost = tot_rev - tot_profit

    # UPDATED CARD STYLES
    c1.markdown(
        f"""<div class="kpi-card"><div class="kpi-title">REVENUE</div><div class="kpi-value">${tot_rev:,.0f}M</div></div>""",
        unsafe_allow_html=True)
    c2.markdown(
        f"""<div class="kpi-card"><div class="kpi-title">TOTAL COST</div><div class="kpi-value" style="color: #c62828;">${tot_cost:,.0f}M</div></div>""",
        unsafe_allow_html=True)
    c3.markdown(
        f"""<div class="kpi-card"><div class="kpi-title">PROFIT</div><div class="kpi-value" style="color: #2e7d32;">${tot_profit:,.0f}M</div></div>""",
        unsafe_allow_html=True)
    c4.markdown(
        f"""<div class="kpi-card"><div class="kpi-title">PROFIT MARGIN</div><div class="kpi-value" style="color: #f57f17;">{margin:.1f}%</div></div>""",
        unsafe_allow_html=True)

    # --- MAP ---
    st.markdown("### 🗺️ SUPPLY CHAIN NETWORK MAP")
    st_folium(create_map(nodes, df, arcs, params), width=1400, height=450,
              key=f"map_{st.session_state.current_scenario}")

    # --- PERFORMANCE ---
    st.markdown("### ⚙️ OPERATIONAL PERFORMANCE")
    o1, o2, o3, o4 = st.columns(4)
    tot_vol = df['Volume_MT'].sum() if not df.empty else 0
    tot_cap = production['Capacity_MTPA'].sum()
    utilization = (tot_vol / tot_cap * 100) if tot_cap > 0 else 0
    active_cnt = len(df)
    avg_dist = df['Distance'].mean() if not df.empty else 0

    o1.metric("Production Utilization", f"{utilization:.1f}%", f"{tot_vol:.1f} / {tot_cap:.1f} MTPA")
    o2.metric("Demand Fulfillment", f"{(tot_vol / markets['Base_Demand_MTPA'].sum() * 100):.1f}%", "Target")
    o3.metric("Active Routes", f"{active_cnt}/15", "Network Health")
    o4.metric("Avg Route Distance", f"{avg_dist:,.0f} km", "Weighted Avg")

    # --- CHARTS ---
    st.markdown("### 📊 DETAILED ANALYSIS")
    g1, g2 = st.columns(2)
    if not df.empty:
        with g1:
            st.markdown("**Market Allocation (Volume)**")
            fig_pie = px.pie(df, values='Volume_MT', names='Market', hole=0.4,
                             color_discrete_sequence=px.colors.qualitative.Pastel)
            # UPDATED CHART STYLING
            fig_pie.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color="#2c3e50",
                legend_font_color="#2c3e50",
                margin=dict(t=0, b=0, l=0, r=0)
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        with g2:
            st.markdown("**Profit by Market ($M)**")
            fig_bar = px.bar(df.groupby('Market')['Profit'].sum().reset_index(), x='Market', y='Profit', color='Market',
                             color_discrete_sequence=px.colors.qualitative.Pastel)
            # UPDATED CHART STYLING
            fig_bar.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color="#2c3e50",
                showlegend=False,
                margin=dict(t=0, b=0, l=0, r=0)
            )
            st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.warning("⚠️ No routes generated for current scenario.")

    # --- TABLE ---
    st.markdown("### 📋 ROUTE ALLOCATION DETAILS")
    if not df.empty:
        st.dataframe(df[['Origin', 'Via_Port', 'Market', 'Volume_MT', 'Revenue', 'Profit']], use_container_width=True)
    else:
        st.info("No active routes in this scenario.")

    # --- SCENARIOS ---
    st.markdown("### 🔄 SCENARIO COMPARISON")
    st.caption("Quick Scenario Switch:")
    b1, b2, b3, b4, b5 = st.columns(5)
    scenarios_list = ["Base Case", "Foot-and-Mouth Outbreak", "JBS Plant Sanitation Issue",
                      "China Beef Ban (Political)", "Brazil Trucker Strike"]

    def set_scen(s):
        st.session_state.current_scenario = s
        st.rerun()

    if b1.button(scenarios_list[0]): set_scen(scenarios_list[0])
    if b2.button(scenarios_list[1]): set_scen(scenarios_list[1])
    if b3.button(scenarios_list[2]): set_scen(scenarios_list[2])
    if b4.button(scenarios_list[3]): set_scen(scenarios_list[3])
    if b5.button(scenarios_list[4]): set_scen(scenarios_list[4])

    # --- RECOMMENDATIONS ---
    st.markdown("### 💡 RECOMMENDATIONS & INSIGHTS")
    scenario = st.session_state.current_scenario

    if scenario == "Foot-and-Mouth Outbreak":
        st.markdown("""<div class="rec-box rec-critical">
            <div class="rec-header">⚠️ CRITICAL: FMD OUTBREAK (MATO GROSSO)</div>
            <ul>
                <li><strong>Observation:</strong> Foot-and-mouth disease detected. Export bans to China, EU, USA.</li>
                <li><strong>Action:</strong> Declare Force Majeure. Pivot to Middle East markets.</li>
            </ul>
        </div>""", unsafe_allow_html=True)
    elif scenario == "China Beef Ban (Political)":
        st.markdown("""<div class="rec-box rec-critical">
            <div class="rec-header">⚠️ CRITICAL: CHINA IMPORT SUSPENSION</div>
            <ul>
                <li><strong>Observation:</strong> Diplomatic tensions halt Brazilian beef imports to China.</li>
                <li><strong>Action:</strong> Emergency pivot to Middle East (UAE) and EU.</li>
            </ul>
        </div>""", unsafe_allow_html=True)
    else:
        if margin < 15:
            st.markdown(f"""<div class="rec-box rec-warning">
                <div class="rec-header">⚠️ WARNING - Low Margin ({margin:.1f}%)</div>
                <ul>
                    <li><strong>Observation:</strong> Margins below 15% target.</li>
                    <li><strong>Action:</strong> Review reefer freight contracts.</li>
                </ul>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="rec-box rec-success">
                <div class="rec-header">✅ OPTIMAL OPERATIONS</div>
                <ul>
                    <li><strong>Observation:</strong> System operating efficiently at {utilization:.1f}% capacity with {margin:.1f}% margins.</li>
                    <li><strong>Strategy:</strong> Strong demand across all markets.</li>
                </ul>
            </div>""", unsafe_allow_html=True)

    # --- EXPORT ---
    st.markdown("### 📥 EXPORT DATA")
    e1, e2, e3 = st.columns(3)
    with e1:
        if not df.empty:
            st.download_button("📊 Export Route Data", df.to_csv(index=False), "bovine_meat_routes.csv", "text/csv")
    with e2:
        if not df.empty:
            st.download_button("💰 Export Financial Summary", df.groupby('Market')[['Revenue', 'Profit']].sum().to_csv(),
                               "bovine_meat_financials.csv", "text/csv")
    with e3:
        if st.button("🔄 Reset to Base Case"):
            st.session_state.current_scenario = "Base Case"
            st.rerun()


if __name__ == "__main__":
    main()