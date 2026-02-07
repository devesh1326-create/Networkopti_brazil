# ==========================================
# COFFEE SUPPLY CHAIN CONTROL TOWER
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
    page_title="Coffee Supply Chain Control Tower",
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

    /* --- FIX 1: SIDEBAR TEXT VISIBILITY --- */
    /* Force all text elements in sidebar to be dark grey */
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

    /* --- FIX 3: UNIFORM BUTTON STYLING --- */
    /* Apply Green Gradient to BOTH standard buttons and Download buttons */
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
    /* Hover State for both */
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
    transport_arcs = pd.DataFrame({
        'Arc_ID': ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15'],
        'From_Node': ['Sul de Minas Region', 'Cooxupé Cooperative', 'Cocape Cooperative', 'São Paulo Roastery',
                      'Espirito Santo Region', 'Santos Port', 'Santos Port', 'Santos Port', 'Santos Port',
                      'Santos Port',
                      'Vitória Port', 'São Paulo Roastery', 'São Paulo DC', 'Santos Port', 'Santos Port'],
        'To_Node': ['Cooxupé Cooperative', 'São Paulo Roastery', 'Santos Port', 'Santos Port',
                    'Vitória Port', 'Hamburg Port', 'Antwerp Port', 'New Orleans Port', 'Genoa Port', 'Yokohama Port',
                    'Hamburg Port', 'São Paulo DC', 'Rio de Janeiro DC', 'Rotterdam Port', 'Shanghai Port'],
        'Mode': ['Road', 'Rail', 'Road', 'Road', 'Road', 'Sea', 'Sea', 'Sea', 'Sea', 'Sea', 'Sea', 'Road', 'Road',
                 'Sea', 'Sea'],
        'Distance_km': [50, 350, 400, 80, 100, 9500, 9300, 7800, 9200, 18900, 9100, 40, 430, 9400, 18500],
        'Capacity_MT': [1.8, 2.5, 1.2, 1.5, 0.8, 3.0, 1.2, 1.5, 1.0, 0.8, 0.6, 1.0, 0.5, 0.7, 0.4],
        'Cost_per_t_USD': [25, 45, 55, 30, 28, 85, 82, 75, 80, 150, 80, 15, 45, 83, 145]
    })

    nodes = pd.DataFrame({
        'Node': ['Sul de Minas Region', 'Cooxupé Cooperative', 'Cocape Cooperative', 'Minasul Cooperative',
                 'São Paulo Roastery', '3 Corações Factory', 'Santos Port', 'Vitória Port',
                 'São Paulo DC', 'Rio de Janeiro DC', 'Hamburg Port', 'Antwerp Port',
                 'New Orleans Port', 'Genoa Port', 'Yokohama Port', 'Rotterdam Port', 'Shanghai Port',
                 'Espirito Santo Region'],
        'Type': ['Farm', 'Cooperative', 'Cooperative', 'Cooperative', 'Roastery', 'Roastery', 'Port', 'Port',
                 'Distribution', 'Distribution', 'Market', 'Market', 'Market', 'Market', 'Market', 'Market', 'Market',
                 'Farm'],
        'Lat': [-21.2, -21.3, -21.5, -21.2, -23.5, -22.4, -23.96, -20.3, -23.55, -22.9, 53.5, 51.2, 29.9, 44.4, 35.4,
                51.9, 31.2, -19.5],
        'Lon': [-45.0, -46.7, -45.4, -45.0, -46.6, -46.8, -46.33, -40.3, -46.63, -43.1, 9.9, 4.4, -90.0, 8.9, 139.6,
                4.5, 121.5, -40.5]
    })

    markets = pd.DataFrame({
        'Market': ['Hamburg Port', 'New Orleans Port', 'Antwerp Port', 'Genoa Port', 'Yokohama Port', 'Rotterdam Port',
                   'Shanghai Port'],
        'Region_Group': ['EU', 'USA', 'EU', 'EU', 'Asia', 'EU', 'Asia'],
        'Base_Demand_MT': [1.2, 1.0, 0.8, 0.7, 0.5, 0.4, 0.3],
        'Base_Price_USD': [4500, 4600, 4450, 4550, 4800, 4420, 4700]
    })

    production = pd.DataFrame({
        'Node': ['Sul de Minas Region', 'Cooxupé Cooperative', 'Cocape Cooperative', 'São Paulo Roastery',
                 'Espirito Santo Region'],
        'Capacity_MT': [5.0, 3.0, 2.0, 2.5, 3.0]
    })

    return transport_arcs, nodes, markets, production


def get_scenario_params(scenario_name):
    params = {
        "desc": "Normal Operations.",
        "farm_cap": 1.0, "port_cap": 1.0, "sea_cap": 1.0,
        "road_cost": 1.0, "sea_cost": 1.0, "rail_cost": 1.0,
        "dem_eu": 1.0, "dem_usa": 1.0, "dem_asia": 1.0,
        "price_eu": 1.0, "price_usa": 1.0, "price_asia": 1.0,
        "disabled_nodes": []
    }

    if scenario_name == "Minas Gerais Frost Event":
        params.update({"desc": "July frost reduces crop yields by 30%. Farm output drops.",
                       "farm_cap": 0.7, "price_eu": 1.15, "price_usa": 1.12})
    elif scenario_name == "Santos Port Strike":
        params.update({"desc": "Strike blocks 60% of Santos exports. Rerouting to Vitória required.",
                       "port_cap": 0.4, "sea_cost": 1.15})
    elif scenario_name == "Rhine River Low Water":
        params.update({"desc": "Low water disrupts EU inland logistics. Demand reduced.",
                       "dem_eu": 0.6})
    elif scenario_name == "Panama Canal Restrictions":
        params.update({"desc": "Drought reroutes Asia shipments. Sea costs spike.",
                       "sea_cost": 1.5, "price_asia": 1.08})
    elif scenario_name == "Brazil Trucker Strike":
        params.update({"desc": "Protest paralyzes road logistics. Capacity down 60%.",
                       "road_cost": 2.0, "farm_cap": 0.4})
    elif scenario_name == "EU Deforestation Regulation":
        params.update({"desc": "EUDR compliance costs rise. Demand softens.",
                       "dem_eu": 0.7, "price_eu": 0.92})
    elif scenario_name == "Red Sea Shipping Crisis":
        params.update({"desc": "Suez canal avoidance. Sea freight to EU/Asia +60%.",
                       "sea_cost": 1.6, "price_eu": 1.1})

    return params


def optimize_network(arcs, nodes, markets, production, params):
    active_arcs = arcs.copy()

    if params.get('road_cost') != 1.0: active_arcs.loc[active_arcs['Mode'] == 'Road', 'Cost_per_t_USD'] *= params[
        'road_cost']
    if params.get('rail_cost') != 1.0: active_arcs.loc[active_arcs['Mode'] == 'Rail', 'Cost_per_t_USD'] *= params[
        'rail_cost']
    if params.get('sea_cost') != 1.0: active_arcs.loc[active_arcs['Mode'] == 'Sea', 'Cost_per_t_USD'] *= params[
        'sea_cost']

    production_cap = production.set_index('Node')['Capacity_MT'].to_dict()
    for node, cap in production_cap.items():
        if 'Region' in node: production_cap[node] = cap * params['farm_cap']

    if params['port_cap'] != 1.0:
        active_arcs.loc[active_arcs['From_Node'].str.contains('Port'), 'Capacity_MT'] *= params['port_cap']

    active_markets = markets.copy()

    def get_mult(region, kind):
        if region == 'EU': return params['dem_eu'] if kind == 'dem' else params['price_eu']
        if region == 'USA': return params['dem_usa'] if kind == 'dem' else params['price_usa']
        return params['dem_asia'] if kind == 'dem' else params['price_asia']

    active_markets['Adj_Demand'] = active_markets.apply(
        lambda x: x['Base_Demand_MT'] * get_mult(x['Region_Group'], 'dem'), axis=1)
    active_markets['Adj_Price'] = active_markets.apply(
        lambda x: x['Base_Price_USD'] * get_mult(x['Region_Group'], 'price'), axis=1)

    all_chains = []

    def get_arc_details(u, v):
        rows = active_arcs[(active_arcs['From_Node'] == u) & (active_arcs['To_Node'] == v)]
        if rows.empty: return None
        return rows.iloc[0]

    market_demand = active_markets.set_index('Market')['Adj_Demand'].to_dict()
    market_price = active_markets.set_index('Market')['Adj_Price'].to_dict()

    for mkt, demand in market_demand.items():
        if demand <= 0: continue
        price = market_price[mkt]
        sea_legs = active_arcs[active_arcs['To_Node'] == mkt]

        for _, sea in sea_legs.iterrows():
            port = sea['From_Node']

            if port == 'Santos Port':
                a4 = get_arc_details('São Paulo Roastery', 'Santos Port')
                if a4 is not None:
                    a2 = get_arc_details('Cooxupé Cooperative', 'São Paulo Roastery')
                    if a2 is not None:
                        a1 = get_arc_details('Sul de Minas Region', 'Cooxupé Cooperative')
                        if a1 is not None:
                            total_cost = sea['Cost_per_t_USD'] + a4['Cost_per_t_USD'] + a2['Cost_per_t_USD'] + a1[
                                'Cost_per_t_USD']
                            all_chains.append({
                                'Type': 'Complex', 'Origin': 'Sul de Minas Region', 'Via_Port': port, 'Market': mkt,
                                'Arcs': [a1['Arc_ID'], a2['Arc_ID'], a4['Arc_ID'], sea['Arc_ID']],
                                'Profit_per_t': price - total_cost, 'Price': price,
                                'Distance': a1['Distance_km'] + a2['Distance_km'] + a4['Distance_km'] + sea[
                                    'Distance_km']
                            })

            if port == 'Santos Port':
                a3 = get_arc_details('Cocape Cooperative', 'Santos Port')
                if a3 is not None:
                    total_cost = sea['Cost_per_t_USD'] + a3['Cost_per_t_USD']
                    all_chains.append({
                        'Type': 'Direct_Coop', 'Origin': 'Cocape Cooperative', 'Via_Port': port, 'Market': mkt,
                        'Arcs': [a3['Arc_ID'], sea['Arc_ID']],
                        'Profit_per_t': price - total_cost, 'Price': price,
                        'Distance': a3['Distance_km'] + sea['Distance_km']
                    })

            if port == 'Vitória Port':
                a5 = get_arc_details('Espirito Santo Region', 'Vitória Port')
                if a5 is not None:
                    total_cost = sea['Cost_per_t_USD'] + a5['Cost_per_t_USD']
                    all_chains.append({
                        'Type': 'Direct_Farm', 'Origin': 'Espirito Santo Region', 'Via_Port': port, 'Market': mkt,
                        'Arcs': [a5['Arc_ID'], sea['Arc_ID']],
                        'Profit_per_t': price - total_cost, 'Price': price,
                        'Distance': a5['Distance_km'] + sea['Distance_km']
                    })

    sorted_chains = sorted(all_chains, key=lambda x: x['Profit_per_t'], reverse=True)
    final_routes = []
    arc_capacity = active_arcs.set_index('Arc_ID')['Capacity_MT'].to_dict()

    for chain in sorted_chains:
        mkt = chain['Market']
        origin = chain['Origin']
        vol = market_demand.get(mkt, 0)
        vol = min(vol, production_cap.get(origin, 0))
        for arc_id in chain['Arcs']:
            vol = min(vol, arc_capacity.get(arc_id, 0))

        if vol > 0.001:
            final_routes.append({
                'Origin': origin, 'Via_Port': chain['Via_Port'], 'Market': mkt,
                'Volume_MT': vol, 'Revenue': vol * chain['Price'],
                'Profit': vol * chain['Profit_per_t'], 'Distance': chain['Distance']
            })
            market_demand[mkt] -= vol
            production_cap[origin] -= vol
            for arc_id in chain['Arcs']: arc_capacity[arc_id] -= vol

    return pd.DataFrame(final_routes)


# ------------------------------------------
# 3. VISUALIZATION ENGINE
# ------------------------------------------

def create_dark_map(nodes, routes_df, transport_arcs, params):
    m = folium.Map(location=[-10, -30], zoom_start=3, tiles='CartoDB positron')

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

        is_active = False
        if (arc['From_Node'], arc['To_Node']) in active_pairs:
            is_active = True
        elif arc['Mode'] in ['Road', 'Rail']:
            if arc['To_Node'] in routes_df['Via_Port'].values:
                is_active = True

        if is_active:
            if arc['Mode'] == 'Rail':
                color = '#e65100'
            elif arc['Mode'] == 'Sea':
                color = '#0277bd'
            else:
                color = '#5d4037'
            weight, opacity, dash = 3, 0.9, None
            tooltip = f"✅ Active: {arc['From_Node']} -> {arc['To_Node']}"
        else:
            color, weight, opacity, dash = '#bdbdbd', 1, 0.4, '5,5'
            tooltip = f"❌ Inactive: {arc['From_Node']} -> {arc['To_Node']}"

        folium.PolyLine(locations=[[orig['Lat'], orig['Lon']], [dest['Lat'], dest['Lon']]],
                        color=color, weight=weight, opacity=opacity, dash_array=dash, tooltip=tooltip).add_to(m)

    for _, node in nodes.iterrows():
        is_disabled = node['Node'] in params['disabled_nodes']
        if 'Farm' in node['Type']:
            color = '#5d4037'
        elif 'Cooperative' in node['Type']:
            color = '#795548'
        elif 'Roastery' in node['Type']:
            color = '#e65100'
        elif 'Port' in node['Type']:
            color = '#2e7d32'
        elif 'Market' in node['Type']:
            color = '#6a1b9a'
        else:
            color = '#757575'
        if is_disabled: color = '#212121'

        folium.CircleMarker(location=[node['Lat'], node['Lon']], radius=5 if not is_disabled else 3, color=color,
                            fill=True, fill_color=color, fill_opacity=0.9,
                            popup=f"<b>{node['Node']}</b><br>{node['Type']}", tooltip=node['Node']).add_to(m)

    # MAP LEGEND (Styled for Visibility)
    legend_html = '''
    <div style="position: fixed; bottom: 50px; right: 50px; width: 150px; background-color: white; border: 1px solid #ccc; z-index: 9999; font-family: sans-serif; font-size: 13px; padding: 10px; border-radius: 4px; box-shadow: 2px 2px 5px rgba(0,0,0,0.3); color: #333;">
        <div style="margin-bottom: 5px; font-weight: bold;">LEGEND</div>
        <div><span style="color: #5d4037;">●</span> Farm</div>
        <div><span style="color: #795548;">●</span> Cooperative</div>
        <div><span style="color: #e65100;">●</span> Roastery</div>
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
        st.info("Simulate disruptions to Coffee Logistics.")

        scenarios = ["Base Case", "Minas Gerais Frost Event", "Santos Port Strike",
                     "Rhine River Low Water", "Panama Canal Restrictions", "Brazil Trucker Strike",
                     "EU Deforestation Regulation", "Red Sea Shipping Crisis"]

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

    st.title(f"☕ COFFEE SUPPLY CHAIN CONTROL TOWER")
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

    c1.markdown(
        f"""<div class="kpi-card"><div class="kpi-title">REVENUE</div><div class="kpi-value">${tot_rev / 1000:,.1f}B</div></div>""",
        unsafe_allow_html=True)
    c2.markdown(
        f"""<div class="kpi-card"><div class="kpi-title">LOGISTICS COST</div><div class="kpi-value" style="color: #c62828;">${tot_cost / 1000:,.1f}B</div></div>""",
        unsafe_allow_html=True)
    c3.markdown(
        f"""<div class="kpi-card"><div class="kpi-title">NET PROFIT</div><div class="kpi-value" style="color: #2e7d32;">${tot_profit / 1000:,.1f}B</div></div>""",
        unsafe_allow_html=True)
    c4.markdown(
        f"""<div class="kpi-card"><div class="kpi-title">MARGIN</div><div class="kpi-value" style="color: #f57f17;">{margin:.1f}%</div></div>""",
        unsafe_allow_html=True)

    # --- MAP ---
    st.markdown("### 🗺️ GLOBAL COFFEE LOGISTICS MAP")
    st_folium(create_dark_map(nodes, df, arcs, params), width=1400, height=450,
              key=f"map_{st.session_state.run_id}")

    # --- PERFORMANCE ---
    st.markdown("### ⚙️ OPERATIONAL METRICS")
    o1, o2, o3, o4 = st.columns(4)
    tot_vol = df['Volume_MT'].sum() if not df.empty else 0
    active_cnt = len(df)
    avg_dist = df['Distance'].mean() if not df.empty else 0
    total_prod_cap = production['Capacity_MT'].sum()
    util = (tot_vol / total_prod_cap * 100) if total_prod_cap > 0 else 0

    o1.metric("Farm/Coop Utilization", f"{util:.1f}%", f"{tot_vol:.1f} MT Exported")
    o2.metric("Market Demand Met", "Dynamic", "See Chart")
    o3.metric("Active Supply Chains", f"{active_cnt}", "Routes Optimized")
    o4.metric("Avg Distance to Market", f"{avg_dist:,.0f} km", "Weighted Avg")

    # --- CHARTS (FIXED VISIBILITY) ---
    st.markdown("### 📊 MARKET ANALYSIS")
    col1, col2 = st.columns(2)
    with col1:
        if not df.empty:
            st.markdown("**Export Volume by Destination (MT)**")
            fig = px.pie(df, values='Volume_MT', names='Market', hole=0.4,
                         color_discrete_sequence=px.colors.sequential.Greens)
            # --- FIX 2: BLACK TEXT FOR LEGEND ---
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color="#2c3e50",  # BLACK/DARK GREY
                legend_font_color="#2c3e50",
                margin=dict(t=0, b=0, l=0, r=0)
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if not df.empty:
            st.markdown("**Profitability by Route ($M)**")
            fig2 = px.bar(df.groupby('Market')['Profit'].sum().reset_index(), x='Market', y='Profit', color='Market',
                          color_discrete_sequence=px.colors.qualitative.Pastel)
            fig2.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color="#2c3e50",  # BLACK/DARK GREY
                showlegend=False,
                margin=dict(t=0, b=0, l=0, r=0)
            )
            st.plotly_chart(fig2, use_container_width=True)

    # --- TABLE ---
    st.markdown("### 📋 EXPORT MANIFEST")
    if not df.empty:
        st.dataframe(df[['Origin', 'Via_Port', 'Market', 'Volume_MT', 'Revenue', 'Profit']], use_container_width=True)

    # --- SCENARIOS ---
    st.markdown("### 🔄 QUICK SCENARIOS")
    b1, b2, b3, b4 = st.columns(4)

    def set_scen(s):
        st.session_state.current_scenario = s
        st.session_state.run_id += 1
        st.rerun()

    if b1.button("Base Case"): set_scen("Base Case")
    if b2.button("Frost Event"): set_scen("Minas Gerais Frost Event")
    if b3.button("Port Strike"): set_scen("Santos Port Strike")
    if b4.button("EU Regulation"): set_scen("EU Deforestation Regulation")

    # --- RECOMMENDATIONS ---
    st.markdown("### 💡 AI-DRIVEN INSIGHTS")
    scen = st.session_state.current_scenario

    if scen == "Minas Gerais Frost Event":
        st.markdown(f"""
        <div class="rec-box rec-critical">
            <div class="rec-header">⚠️ CRITICAL: CROP FAILURE (FROST)</div>
            <ul>
                <li><strong>Impact:</strong> Supply from Sul de Minas down 30%. Global prices spiking.</li>
                <li><strong>Action:</strong> Release strategic stocks from Cooxupé. Prioritize high-margin contracts to USA/Germany.</li>
            </ul>
        </div>""", unsafe_allow_html=True)

    elif scen == "Santos Port Strike":
        st.markdown(f"""
        <div class="rec-box rec-critical">
            <div class="rec-header">⚠️ CRITICAL: LOGISTICS BLOCKADE (SANTOS)</div>
            <ul>
                <li><strong>Impact:</strong> Santos capacity cut by 60%. Major bottleneck.</li>
                <li><strong>Action:</strong> Reroute road shipments to Vitória Port (Espirito Santo) immediately. Activate Arc A11.</li>
            </ul>
        </div>""", unsafe_allow_html=True)

    elif scen == "EU Deforestation Regulation":
        st.markdown(f"""
        <div class="rec-box rec-warning">
            <div class="rec-header">⚠️ REGULATORY RISK: EUDR COMPLIANCE</div>
            <ul>
                <li><strong>Impact:</strong> EU demand softening due to compliance friction.</li>
                <li><strong>Action:</strong> Shift non-certified volumes to Asian markets (Shanghai/Yokohama).</li>
            </ul>
        </div>""", unsafe_allow_html=True)

    elif scen == "Brazil Trucker Strike":
        st.markdown(f"""
        <div class="rec-box rec-critical">
            <div class="rec-header">⚠️ CRITICAL: ROAD PARALYSIS</div>
            <ul>
                <li><strong>Impact:</strong> Farm-to-Port links broken.</li>
                <li><strong>Action:</strong> Maximize rail usage (Arc A2: Cooxupé -> SP). Declare Force Majeure on FOB terms.</li>
            </ul>
        </div>""", unsafe_allow_html=True)

    else:
        st.markdown(f"""
        <div class="rec-box rec-success">
            <div class="rec-header">✅ NETWORK OPTIMAL</div>
            <ul>
                <li><strong>Status:</strong> Operations normal. Arabica harvest on schedule.</li>
                <li><strong>Strategy:</strong> Monitor Rhine water levels for future EU deliveries.</li>
            </ul>
        </div>""", unsafe_allow_html=True)

    # --- EXPORT ---
    st.markdown("### 📥 DATA EXPORT")
    e1, e2, e3 = st.columns(3)
    with e1:
        if not df.empty:
            st.download_button("📊 Download Routes", df.to_csv(index=False), "coffee_routes.csv", "text/csv")
    with e2:
        if not df.empty:
            st.download_button("💰 Download Financials", df.groupby('Market')['Profit'].sum().to_csv(),
                               "coffee_financials.csv", "text/csv")
    with e3:
        if st.button("🔄 Reset Simulation"):
            set_scen("Base Case")


if __name__ == "__main__":
    main()
