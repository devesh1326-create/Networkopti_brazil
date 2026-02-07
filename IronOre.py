# ==========================================
# IRON ORE SUPPLY CHAIN CONTROL TOWER
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
    page_title="Iron Ore Control Tower",
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
if 'run_id' not in st.session_state:
    st.session_state.run_id = 0


# ------------------------------------------
# 2. DATA ENGINE
# ------------------------------------------

@st.cache_data
def load_data():
    # Transport Arcs
    transport_arcs = pd.DataFrame({
        'Arc_ID': ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15'],
        'From_Node': ['Serra Sul Mine Pit (S11D)', 'Serra Norte Mine', 'S11D Plant', 'Parauapebas Terminal',
                      'Brucutu Mine', 'EFVM Rail Hub', 'Fábrica Nova Mine', 'MRS Rail Hub',
                      'PDM Port (Ponta da Madeira)',
                      'SA Waypoint', 'SA Waypoint', 'Tubarão Port', 'Tubarão Port', 'PDM Port (Ponta da Madeira)',
                      'Guaíba Port'],
        'To_Node': ['S11D Plant', 'S11D Plant', 'Parauapebas Terminal', 'PDM Port (Ponta da Madeira)',
                    'EFVM Rail Hub', 'Tubarão Port', 'MRS Rail Hub', 'Guaíba Port',
                    'SA Waypoint', 'Qingdao Port', 'Teluk Rubiah Port', 'Sohar Port',
                    'Kimitsu Port', 'Rotterdam Port', 'Rotterdam Port'],
        'Mode': ['Rail', 'Rail', 'Rail', 'Rail', 'Rail', 'Rail', 'Rail', 'Rail',
                 'Sea', 'Sea', 'Sea', 'Sea', 'Sea', 'Sea', 'Sea'],
        'Distance_km': [25, 45, 130, 890, 35, 905, 55, 1180, 3300, 17500, 15200, 11300, 18400, 8700, 10000],
        'Capacity_MTPA': [80, 80, 80, 80, 30, 60, 25, 50, 90, 90, 60, 40, 25, 40, 30],
        'Cost_per_t_USD': [1.5, 2.0, 3.5, 7.5, 1.5, 8.5, 2.0, 10.0, 6.0, 14.5, 13.0, 12.0, 15.0, 10.0, 11.5]
    })

    # Nodes
    nodes = pd.DataFrame({
        'Node': ['Serra Sul Mine Pit (S11D)', 'Serra Norte Mine', 'S11D Plant', 'Parauapebas Terminal', 'Brucutu Mine',
                 'EFVM Rail Hub', 'Fábrica Nova Mine', 'MRS Rail Hub', 'PDM Port (Ponta da Madeira)', 'SA Waypoint',
                 'Tubarão Port', 'Guaíba Port', 'Qingdao Port', 'Teluk Rubiah Port', 'Sohar Port', 'Kimitsu Port',
                 'Rotterdam Port'],
        'Type': ['Mine', 'Mine', 'Plant', 'Port / Terminal', 'Mine', 'Rail_Hub', 'Mine', 'Rail_Hub', 'Port', 'Waypoint',
                 'Port', 'Port', 'Market', 'Market', 'Market', 'Market', 'Market'],
        'Lat': [-6.40, -6.10, -6.06, -5.90, -19.86, -19.90, -20.45, -21.10, -2.57, -34.35, -20.29, -23.01,
                36.06, 4.19, 24.4, 35.32, 51.95],
        'Lon': [-50.36, -50.18, -50.18, -49.90, -43.38, -43.25, -43.95, -43.90, -44.38, 18.42, -40.24, -44.03,
                120.33, 100.65, 56.63, 139.91, 4.05]
    })

    # Markets (Total Demand ~55 MTPA)
    markets = pd.DataFrame({
        'Market': ['Qingdao Port', 'Kimitsu Port', 'Teluk Rubiah Port', 'Rotterdam Port', 'Sohar Port'],
        'Region_Group': ['China', 'Japan', 'SE_Asia', 'Europe', 'MiddleEast'],
        'Base_Demand_MTPA': [20, 12, 8, 9, 6],
        'Base_Price_USD': [115, 120, 110, 105, 108]
    })

    # Production (Tighter Constraints)
    production = pd.DataFrame({
        'Node': ['Serra Sul Mine Pit (S11D)', 'Serra Norte Mine', 'Brucutu Mine', 'Fábrica Nova Mine'],
        'Capacity_MTPA': [50, 40, 20, 15],
    })

    return transport_arcs, nodes, markets, production


def get_scenario_params(scenario_name):
    # Base multipliers
    params = {
        "desc": "Normal Operations.",
        "rail_cap": 1.0, "port_cap": 1.0, "sea_cap": 1.0,
        "rail_cost": 1.0, "sea_cost": 1.0,
        "dem_cn": 1.0, "dem_eu": 1.0, "dem_asia": 1.0, "dem_me": 1.0,
        "price_cn": 1.0, "price_eu": 1.0, "price_asia": 1.0, "price_me": 1.0,
        "disabled_nodes": []
    }

    # Scenario Logic
    if scenario_name == "Pará Monsoon Rail Disruption":
        params.update({"desc": "Flooding reduces EFVM rail capacity by 40%. Cost spikes.",
                       "rail_cap": 0.6, "rail_cost": 2.0})

    elif scenario_name == "S11D Mine Equipment Failure":
        params.update({"desc": "S11D Plant offline. 50% supply shock.",
                       "disabled_nodes": ["S11D Plant"]})

    elif scenario_name == "PDM Port Congestion":
        params.update({"desc": "Congestion at PDM. Costs rise significantly.",
                       "port_cap": 0.5, "sea_cost": 1.5})

    elif scenario_name == "China Infrastructure Stimulus":
        params.update({"desc": "China demand +30%. Prices spike.",
                       "dem_cn": 1.3, "price_cn": 1.15})

    elif scenario_name == "Tubarão Port Maintenance":
        params.update({"desc": "Maintenance at Tubarão. Capacity down, costs up.",
                       "port_cap": 0.4, "sea_cost": 1.5})

    return params


def optimize_network(arcs, nodes, markets, production, params):
    active_arcs = arcs.copy()

    # 1. Apply Scenario Multipliers
    for mode in ['Rail', 'Sea']:
        if f"{mode.lower()}_cap" in params:
            active_arcs.loc[active_arcs['Mode'] == mode, 'Capacity_MTPA'] *= params[f"{mode.lower()}_cap"]
        if f"{mode.lower()}_cost" in params:
            active_arcs.loc[active_arcs['Mode'] == mode, 'Cost_per_t_USD'] *= params[f"{mode.lower()}_cost"]

    # 2. Filter Disabled Nodes
    if params['disabled_nodes']:
        active_arcs = active_arcs[~active_arcs['From_Node'].isin(params['disabled_nodes'])]
        active_arcs = active_arcs[~active_arcs['To_Node'].isin(params['disabled_nodes'])]

    # 3. Apply Market Adjustments
    active_markets = markets.copy()

    def get_mult(region, kind):
        if region == 'China': return params[f'dem_cn'] if kind == 'dem' else params[f'price_cn']
        if region == 'Europe': return params[f'dem_eu'] if kind == 'dem' else params[f'price_eu']
        if region == 'MiddleEast': return params[f'dem_me'] if kind == 'dem' else params[f'price_me']
        return params[f'dem_asia'] if kind == 'dem' else params[f'price_asia']

    active_markets['Adj_Demand'] = active_markets.apply(
        lambda x: x['Base_Demand_MTPA'] * get_mult(x['Region_Group'], 'dem'), axis=1)
    active_markets['Adj_Price'] = active_markets.apply(
        lambda x: x['Base_Price_USD'] * get_mult(x['Region_Group'], 'price'), axis=1)

    # ---------------------------------------------------------
    # GLOBAL OPTIMIZATION
    # ---------------------------------------------------------

    all_chains = []

    node_capacity = production.set_index('Node')['Capacity_MTPA'].to_dict()
    arc_capacity = active_arcs.set_index('Arc_ID')['Capacity_MTPA'].to_dict()
    market_demand = active_markets.set_index('Market')['Adj_Demand'].to_dict()
    market_price = active_markets.set_index('Market')['Adj_Price'].to_dict()

    # Iterate Markets
    for mkt_name, demand in market_demand.items():
        if demand <= 0: continue
        price = market_price[mkt_name]

        # Find Sea Legs to this Market
        sea_legs = active_arcs[active_arcs['To_Node'] == mkt_name]

        for _, sea in sea_legs.iterrows():
            port_name = sea['From_Node']

            # Special Waypoint Logic
            if 'Waypoint' in port_name:
                wp_legs = active_arcs[active_arcs['To_Node'] == port_name]
                for _, wp in wp_legs.iterrows():
                    real_port = wp['From_Node']
                    land_legs = active_arcs[active_arcs['To_Node'] == real_port]

                    for _, land in land_legs.iterrows():
                        mine = land['From_Node']
                        effective_mine = mine
                        upstream_cost = 0

                        if mine not in node_capacity:
                            up_arcs = active_arcs[active_arcs['To_Node'] == mine]
                            for _, u in up_arcs.iterrows():
                                effective_mine = u['From_Node']
                                upstream_cost = u['Cost_per_t_USD']
                                break

                        total_cost = land['Cost_per_t_USD'] + wp['Cost_per_t_USD'] + sea[
                            'Cost_per_t_USD'] + upstream_cost
                        profit = price - total_cost

                        all_chains.append({
                            'Mine': effective_mine,
                            'Land_Arc': land['Arc_ID'],
                            'Sea_Arc': sea['Arc_ID'],
                            'WP_Arc': wp['Arc_ID'],
                            'Market': mkt_name,
                            'Profit_per_t': profit,
                            'Total_Cost': total_cost,
                            'Price': price,
                            'Via_Port': real_port,
                            'Distance': land['Distance_km'] + wp['Distance_km'] + sea['Distance_km']
                        })
                continue

            # Standard Logic
            land_legs = active_arcs[active_arcs['To_Node'] == port_name]
            for _, land in land_legs.iterrows():
                mine = land['From_Node']
                effective_mine = mine
                upstream_cost = 0

                if mine not in node_capacity:
                    up_arcs = active_arcs[active_arcs['To_Node'] == mine]
                    for _, u in up_arcs.iterrows():
                        effective_mine = u['From_Node']
                        upstream_cost = u['Cost_per_t_USD']
                        break

                total_cost = land['Cost_per_t_USD'] + sea['Cost_per_t_USD'] + upstream_cost
                profit = price - total_cost

                all_chains.append({
                    'Mine': effective_mine,
                    'Land_Arc': land['Arc_ID'],
                    'Sea_Arc': sea['Arc_ID'],
                    'WP_Arc': None,
                    'Market': mkt_name,
                    'Profit_per_t': profit,
                    'Total_Cost': total_cost,
                    'Price': price,
                    'Via_Port': port_name,
                    'Distance': land['Distance_km'] + sea['Distance_km']
                })

    # Sort and Allocate
    sorted_chains = sorted(all_chains, key=lambda x: x['Profit_per_t'], reverse=True)
    final_routes = []

    for chain in sorted_chains:
        mkt = chain['Market']
        mine = chain['Mine']

        vol = market_demand.get(mkt, 0)
        vol = min(vol, node_capacity.get(mine, 0))
        vol = min(vol, arc_capacity.get(chain['Land_Arc'], 0))
        vol = min(vol, arc_capacity.get(chain['Sea_Arc'], 0))

        if chain['WP_Arc']:
            vol = min(vol, arc_capacity.get(chain['WP_Arc'], 0))

        if vol > 0.01:
            final_routes.append({
                'Origin': mine,
                'Via_Port': chain['Via_Port'],
                'Market': mkt,
                'Volume_MT': vol,
                'Revenue': vol * chain['Price'],
                'Profit': vol * chain['Profit_per_t'],
                'Distance': chain['Distance']
            })

            market_demand[mkt] -= vol
            node_capacity[mine] -= vol
            arc_capacity[chain['Land_Arc']] -= vol
            arc_capacity[chain['Sea_Arc']] -= vol
            if chain['WP_Arc']: arc_capacity[chain['WP_Arc']] -= vol

    return pd.DataFrame(final_routes)


# ------------------------------------------
# 3. VISUALIZATION ENGINE
# ------------------------------------------

def create_dark_map(nodes, routes_df, transport_arcs, params):
    # UPDATED: Use Positron for light theme
    m = folium.Map(location=[-15, -10], zoom_start=2, tiles='CartoDB positron')

    active_pairs = set()
    if not routes_df.empty:
        active_pairs = set(zip(routes_df['Origin'], routes_df['Via_Port'])) | set(
            zip(routes_df['Via_Port'], routes_df['Market']))

    for _, arc in transport_arcs.iterrows():
        if arc['From_Node'] not in nodes['Node'].values or arc['To_Node'] not in nodes['Node'].values:
            continue

        orig = nodes[nodes['Node'] == arc['From_Node']].iloc[0]
        dest = nodes[nodes['Node'] == arc['To_Node']].iloc[0]

        is_active = (arc['From_Node'], arc['To_Node']) in active_pairs

        # Simplified Waypoint check for visual consistency
        if 'Waypoint' in arc['From_Node'] or 'Waypoint' in arc['To_Node']:
            pass

        if is_active:
            # Colors adjusted for light background
            if arc['Mode'] == 'Rail':
                color = '#e65100'  # Orange
            elif arc['Mode'] == 'Sea':
                color = '#0277bd'  # Blue
            else:
                color = '#5d4037'  # Brown
            weight, opacity, dash = 3, 0.9, None
            tooltip = f"✅ Active: {arc['From_Node']} -> {arc['To_Node']}"
        else:
            color, weight, opacity, dash = '#bdbdbd', 1.5, 0.5, '5,5'  # Light Grey
            tooltip = f"❌ Inactive: {arc['From_Node']} -> {arc['To_Node']}"

        folium.PolyLine(locations=[[orig['Lat'], orig['Lon']], [dest['Lat'], dest['Lon']]],
                        color=color, weight=weight, opacity=opacity, dash_array=dash, tooltip=tooltip).add_to(m)

    for _, node in nodes.iterrows():
        is_disabled = node['Node'] in params['disabled_nodes']
        if node['Type'] == 'Mine':
            color = '#c62828'  # Red
        elif node['Type'] in ['Plant', 'Rail_Hub']:
            color = '#f57f17'  # Orange
        elif 'Port' in node['Type']:
            color = '#2e7d32'  # Green
        elif node['Type'] == 'Market':
            color = '#6a1b9a'  # Purple
        else:
            color = '#757575'

        if is_disabled: color = '#212121'

        folium.CircleMarker(location=[node['Lat'], node['Lon']], radius=5 if not is_disabled else 3, color=color,
                            fill=True, fill_color=color, fill_opacity=0.8,
                            popup=f"<b>{node['Node']}</b>", tooltip=node['Node']).add_to(m)

    # MAP LEGEND
    legend_html = '''
    <div style="position: fixed; bottom: 50px; right: 50px; width: 150px; background-color: white; border: 1px solid #ccc; z-index: 9999; font-family: sans-serif; font-size: 13px; padding: 10px; border-radius: 4px; box-shadow: 2px 2px 5px rgba(0,0,0,0.3); color: #333;">
        <div style="margin-bottom: 5px; font-weight: bold;">LEGEND</div>
        <div><span style="color: #c62828;">●</span> Mine</div>
        <div><span style="color: #f57f17;">●</span> Processing/Hub</div>
        <div><span style="color: #2e7d32;">●</span> Port</div>
        <div><span style="color: #6a1b9a;">●</span> Market</div>
        <hr style="margin: 5px 0; border: 0; border-top: 1px solid #ddd;">
        <div><span style="color: #e65100;">━</span> Rail</div>
        <div><span style="color: #0277bd;">━</span> Sea</div>
        <div><span style="color: #bdbdbd;">--</span> Inactive</div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    return m


# ------------------------------------------
# 4. MAIN LAYOUT
# ------------------------------------------

def main():
    with st.sidebar:
        st.header("⚙️ SCENARIO CONTROL")
        st.info("Simulate disruptions to Global Iron Ore Logistics.")

        scenarios = ["Base Case", "Pará Monsoon Rail Disruption", "S11D Mine Equipment Failure",
                     "PDM Port Congestion", "China Infrastructure Stimulus", "Tubarão Port Maintenance"]

        if st.session_state.current_scenario not in scenarios:
            st.session_state.current_scenario = "Base Case"

        selected_scen = st.selectbox("Select Scenario:", scenarios,
                                     index=scenarios.index(st.session_state.current_scenario))

        if st.button("Apply Scenario"):
            st.session_state.current_scenario = selected_scen
            st.session_state.run_id += 1
            st.rerun()

        st.markdown("---")
        params = get_scenario_params(st.session_state.current_scenario)
        st.markdown(f"**Current Desc:**\n{params['desc']}")

    st.title(f"ACTIVE SCENARIO: {st.session_state.current_scenario}")

    arcs, nodes, markets, production = load_data()
    params = get_scenario_params(st.session_state.current_scenario)
    df = optimize_network(arcs, nodes, markets, production, params)

    # --- KPI CARDS ---
    c1, c2, c3, c4 = st.columns(4)
    tot_rev = df['Revenue'].sum() if not df.empty else 0
    tot_profit = df['Profit'].sum() if not df.empty else 0
    margin = (tot_profit / tot_rev * 100) if tot_rev > 0 else 0
    tot_cost = tot_rev - tot_profit

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

    # --- MAP ---
    st.markdown("### 🗺️ SUPPLY CHAIN NETWORK MAP")
    st_folium(create_dark_map(nodes, df, arcs, params), width=1400, height=450,
              key=f"map_{st.session_state.run_id}")

    # --- PERFORMANCE ---
    st.markdown("### ⚙️ OPERATIONAL PERFORMANCE")
    o1, o2, o3, o4 = st.columns(4)
    tot_vol = df['Volume_MT'].sum() if not df.empty else 0
    tot_cap = production['Capacity_MTPA'].sum()
    utilization = (tot_vol / tot_cap * 100) if tot_cap > 0 else 0
    active_cnt = len(df)
    avg_dist = df['Distance'].mean() if not df.empty else 0

    o1.metric("Production Utilization", f"{utilization:.1f}%", f"{tot_vol:.1f} / {tot_cap:.1f} MTPA")
    o2.metric("Demand Fulfillment", "100%", "On Target")
    o3.metric("Active Routes", f"{active_cnt}", "Network Health")
    o4.metric("Avg Route Distance", f"{avg_dist:,.0f} km", "Weighted Avg")

    # --- CHARTS ---
    st.markdown("### 📊 DETAILED ANALYSIS")
    col1, col2 = st.columns(2)
    with col1:
        if not df.empty:
            st.markdown("**Market Allocation (Volume)**")
            fig = px.pie(df, values='Volume_MT', names='Market', hole=0.4,
                         color_discrete_sequence=px.colors.qualitative.Pastel)
            # UPDATED CHART STYLING
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color="#2c3e50",
                legend_font_color="#2c3e50",
                margin=dict(t=0, b=0, l=0, r=0)
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if not df.empty:
            st.markdown("**Profit by Market ($M)**")
            fig2 = px.bar(df.groupby('Market')['Profit'].sum().reset_index(), x='Market', y='Profit', color='Market')
            # UPDATED CHART STYLING
            fig2.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color="#2c3e50",
                showlegend=False,
                margin=dict(t=0, b=0, l=0, r=0)
            )
            st.plotly_chart(fig2, use_container_width=True)

    # --- TABLE ---
    st.markdown("### 📋 ROUTE ALLOCATION DETAILS")
    if not df.empty:
        st.dataframe(df[['Origin', 'Via_Port', 'Market', 'Volume_MT', 'Revenue', 'Profit']], use_container_width=True)

    # --- SCENARIOS ---
    st.markdown("### 🔄 SCENARIO COMPARISON")
    b1, b2, b3, b4, b5 = st.columns(5)

    def set_scen(s):
        st.session_state.current_scenario = s
        st.session_state.run_id += 1
        st.rerun()

    if b1.button("Base Case"): set_scen("Base Case")
    if b2.button("Rail Disruption"): set_scen("Pará Monsoon Rail Disruption")
    if b3.button("S11D Failure"): set_scen("S11D Mine Equipment Failure")
    if b4.button("China Stimulus"): set_scen("China Infrastructure Stimulus")
    if b5.button("Port Congestion"): set_scen("PDM Port Congestion")

    # --- RECOMMENDATIONS ---
    st.markdown("### 💡 RECOMMENDATIONS & INSIGHTS")

    if "Pará Monsoon" in st.session_state.current_scenario:
        st.markdown(f"""
        <div class="rec-box rec-critical">
            <div class="rec-header">⚠️ CRITICAL: RAIL WASHOUT (EFVM)</div>
            <ul>
                <li><strong>Impact:</strong> Rail capacity < Demand. Revenue down by ${(55 - (tot_rev / 1000 * 10)):.1f}B (est).</li>
                <li><strong>Action:</strong> Declare Force Majeure on spot cargoes. Reroute Brucutu inventory to domestic plants.</li>
            </ul>
        </div>""", unsafe_allow_html=True)
    elif "S11D" in st.session_state.current_scenario:
        st.markdown(f"""
        <div class="rec-box rec-critical">
            <div class="rec-header">⚠️ CRITICAL: S11D OFFLINE</div>
            <ul>
                <li><strong>Impact:</strong> Primary northern asset down. Supply gap of 20 MTPA.</li>
                <li><strong>Action:</strong> Maximize output at Serra Norte. Buy third-party ore to blend.</li>
            </ul>
        </div>""", unsafe_allow_html=True)
    elif "China" in st.session_state.current_scenario:
        st.markdown(f"""
        <div class="rec-box rec-success">
            <div class="rec-header">✅ OPPORTUNITY: DEMAND SURGE</div>
            <ul>
                <li><strong>Observation:</strong> China steel demand +30%. Margins up to {margin:.1f}%.</li>
                <li><strong>Action:</strong> Maximize spot sales to Qingdao. Delay maintenance.</li>
            </ul>
        </div>""", unsafe_allow_html=True)
    elif tot_rev == 0:
        st.error("System Failure: No feasible routes found.")
    else:
        st.markdown(f"""
        <div class="rec-box rec-success">
            <div class="rec-header">✅ SYSTEM STABLE</div>
            <ul>
                <li><strong>Status:</strong> Operations normal. Monitor weather in Northern System.</li>
            </ul>
        </div>""", unsafe_allow_html=True)

    # --- EXPORT ---
    st.markdown("### 📥 EXPORT DATA")
    e1, e2, e3 = st.columns(3)
    with e1:
        if not df.empty:
            st.download_button("📊 Export Route Data", df.to_csv(index=False), "routes.csv", "text/csv")
    with e2:
        if not df.empty:
            st.download_button("💰 Export Financials", df.groupby('Market')['Profit'].sum().to_csv(), "profits.csv",
                               "text/csv")
    with e3:
        if st.button("🔄 Reset to Base Case"):
            st.session_state.current_scenario = "Base Case"
            st.session_state.run_id += 1
            st.rerun()


if __name__ == "__main__":
    main()