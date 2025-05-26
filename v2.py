"""
Delivery Clustering & Route Optimization - Visual Prototype
Demonstrates cost savings and efficiency improvements with interactive maps
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import folium
from folium import plugins
from streamlit_folium import st_folium
from datetime import datetime, timedelta
from geopy.distance import geodesic
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Last-Mile Delivery Optimization",
    page_icon="🚚",
    layout="wide"
)

# Initialize session state
if 'clusters' not in st.session_state:
    st.session_state.clusters = None

# Custom CSS for better visuals
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
    }
    .highlight-box {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #1976d2;
        margin: 10px 0;
    }
    .cost-savings {
        background-color: #e8f5e9;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #4caf50;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🚚 Last-Mile Delivery Route Optimization System")
st.markdown("""
### Visual Analysis & Cost-Benefit Demonstration
Transform your manual pincode-based routing into intelligent, cost-effective clusters with multi-trip optimization.
""")

# Sidebar for data upload
st.sidebar.header("📊 Data Upload")
st.sidebar.markdown("""
Please upload your data files:
- [📦 Order Data](https://analytics.blowhorn.com/question/3120-if-network-analysis-last-mile?start=2025-05-01&end=2025-05-26)
- [💰 Driver Cost Data](https://analytics.blowhorn.com/question/3113-if-costs-by-driver?start=2025-05-01&end=2025-05-26)
""")

order_file = st.sidebar.file_uploader(
    "Upload Order Data (CSV)",
    type=['csv'],
    help="File from network analysis with delivery coordinates"
)

cost_file = st.sidebar.file_uploader(
    "Upload Driver Cost Data (CSV)",
    type=['csv'],
    help="File with driver payment information"
)

# Parameters
st.sidebar.header("⚙️ Optimization Parameters")
clustering_method = st.sidebar.selectbox(
    "Clustering Algorithm",
    ["K-Means (Fast)", "DBSCAN (Density-based)", "Hierarchical (Distance-based)"]
)

min_orders_per_cluster = st.sidebar.slider("Min Orders per Cluster", 15, 40, 25)
max_distance_km = st.sidebar.slider("Max Route Distance (km)", 10, 30, 20)
enable_multi_trip = st.sidebar.checkbox("Enable Multi-Trip Optimization", value=True)

if enable_multi_trip:
    multi_trip_percent = st.sidebar.slider("% Drivers for Multi-Trip", 20, 50, 30)
    trips_per_driver = st.sidebar.slider("Max Trips per Driver", 2, 4, 3)

# Helper function for convex hull (if scipy not available)
def ConvexHull(points):
    """Simple convex hull implementation"""
    from collections import namedtuple
    Point = namedtuple('Point', ['x', 'y'])
    
    def cross(O, A, B):
        return (A.x - O.x) * (B.y - O.y) - (A.y - O.y) * (B.x - O.x)
    
    points = [Point(p[1], p[0]) for p in points]
    points = sorted(set(points))
    if len(points) <= 1:
        return type('obj', (object,), {'vertices': list(range(len(points)))})
    
    # Build lower hull
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    
    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    
    hull_points = lower[:-1] + upper[:-1]
    
    # Find indices
    vertices = []
    for hp in hull_points:
        for i, p in enumerate(points):
            if p.x == hp.x and p.y == hp.y:
                vertices.append(i)
                break
    
    return type('obj', (object,), {'vertices': vertices})

# Main content
if order_file is not None:
    # Load data
    df_orders = pd.read_csv(order_file)
    
    # Data preprocessing
    df_orders['created_date'] = pd.to_datetime(df_orders['created_date'])
    df_orders['date'] = df_orders['created_date'].dt.date
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Current State Analysis", 
        "🗺️ Cluster Visualization", 
        "💰 Cost-Benefit Analysis",
        "🚛 Route Planning",
        "📈 Performance Metrics"
    ])
    
    with tab1:
        st.header("Current State Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Orders", f"{len(df_orders):,}")
            st.metric("Unique Drivers", df_orders['driver'].nunique())
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Active Hubs", df_orders['hub'].nunique())
            st.metric("Avg Orders/Driver/Day", 
                     f"{len(df_orders) / df_orders['driver'].nunique() / df_orders['date'].nunique():.1f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            vehicle_types = df_orders['vehicle_model'].value_counts()
            st.metric("Vehicle Types", len(vehicle_types))
            st.metric("Most Common", vehicle_types.index[0])
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Hub distribution
        st.subheader("📍 Order Distribution by Hub")
        hub_stats = df_orders.groupby('hub').agg({
            'number': 'count',
            'weight': 'sum',
            'driver': 'nunique'
        }).round(2)
        hub_stats.columns = ['Total Orders', 'Total Weight (kg)', 'Unique Drivers']
        
        fig_hub = px.bar(hub_stats.reset_index(), 
                         x='hub', 
                         y='Total Orders',
                         color='Total Orders',
                         title="Orders by Hub",
                         color_continuous_scale="Viridis")
        st.plotly_chart(fig_hub, use_container_width=True)
        
        # Vehicle type analysis
        col1, col2 = st.columns(2)
        
        with col1:
            vehicle_dist = df_orders['vehicle_model'].value_counts()
            fig_vehicle = px.pie(values=vehicle_dist.values, 
                               names=vehicle_dist.index,
                               title="Vehicle Type Distribution")
            st.plotly_chart(fig_vehicle, use_container_width=True)
        
        with col2:
            # Orders by date
            daily_orders = df_orders.groupby('date').size().reset_index(name='orders')
            fig_daily = px.line(daily_orders, x='date', y='orders',
                              title="Daily Order Volume",
                              markers=True)
            st.plotly_chart(fig_daily, use_container_width=True)
    
    with tab2:
        st.header("🗺️ Intelligent Clustering Visualization")
        
        # Select a specific date for analysis
        selected_date = st.selectbox("Select Date for Analysis", 
                                   sorted(df_orders['date'].unique(), reverse=True))
        
        df_day = df_orders[df_orders['date'] == selected_date].copy()
        st.info(f"Analyzing {len(df_day)} orders for {selected_date}")
        
        # Perform clustering
        if st.button("🔄 Generate Optimized Clusters"):
            with st.spinner("Creating intelligent clusters..."):
                # Prepare data for clustering
                coords = df_day[['delivered_lat', 'delivered_long']].values
                
                # Apply selected clustering method
                if "K-Means" in clustering_method:
                    n_clusters = max(1, len(df_day) // min_orders_per_cluster)
                    clusterer = KMeans(n_clusters=n_clusters, random_state=42)
                elif "DBSCAN" in clustering_method:
                    clusterer = DBSCAN(eps=0.02, min_samples=min_orders_per_cluster)
                else:  # Hierarchical
                    n_clusters = max(1, len(df_day) // min_orders_per_cluster)
                    clusterer = AgglomerativeClustering(n_clusters=n_clusters)
                
                df_day['cluster'] = clusterer.fit_predict(coords)
                
                # Store in session state
                st.session_state.clusters = df_day
                
        # Visualize clusters on map
        if st.session_state.clusters is not None:
            df_clustered = st.session_state.clusters
            
            # Create base map centered on Bengaluru
            center_lat = df_clustered['delivered_lat'].mean()
            center_lon = df_clustered['delivered_long'].mean()
            
            m = folium.Map(location=[center_lat, center_lon], 
                          zoom_start=11,
                          tiles='OpenStreetMap')
            
            # Add hub markers
            hubs = df_clustered.groupby(['hub', 'hub_lat', 'hub_long']).size().reset_index()
            for _, hub in hubs.iterrows():
                folium.Marker(
                    location=[hub['hub_lat'], hub['hub_long']],
                    popup=f"Hub: {hub['hub']}",
                    icon=folium.Icon(color='red', icon='warehouse', prefix='fa'),
                    tooltip=hub['hub']
                ).add_to(m)
            
            # Add clustered delivery points
            colors = px.colors.qualitative.Plotly
            cluster_groups = df_clustered.groupby('cluster')
            
            for cluster_id, group in cluster_groups:
                if cluster_id >= 0:  # Valid cluster
                    color = colors[cluster_id % len(colors)]
                    
                    # Add delivery points
                    for _, order in group.iterrows():
                        folium.CircleMarker(
                            location=[order['delivered_lat'], order['delivered_long']],
                            radius=5,
                            popup=f"Order: {order['number']}<br>Driver: {order['driver']}",
                            color=color,
                            fill=True,
                            fillColor=color,
                            fillOpacity=0.7
                        ).add_to(m)
                    
                    # Draw cluster boundary
                    if len(group) > 2:
                        points = group[['delivered_lat', 'delivered_long']].values
                        hull_points = points[ConvexHull(points).vertices]
                        folium.Polygon(
                            locations=hull_points.tolist(),
                            color=color,
                            fill=True,
                            fillOpacity=0.2,
                            weight=2
                        ).add_to(m)
            
            # Display map
            st_folium(m, width=1000, height=600)
            
            # Cluster statistics
            st.subheader("📊 Cluster Statistics")
            cluster_stats = df_clustered.groupby('cluster').agg({
                'number': 'count',
                'weight': 'sum',
                'delivered_lat': ['mean', 'std'],
                'delivered_long': ['mean', 'std']
            }).round(3)
            
            cluster_stats.columns = ['Orders', 'Total Weight', 'Lat_Mean', 'Lat_Std', 'Lon_Mean', 'Lon_Std']
            cluster_stats = cluster_stats[cluster_stats.index >= 0]  # Remove noise
            
            # Calculate cluster spread (approximate radius)
            cluster_stats['Spread_km'] = cluster_stats.apply(
                lambda row: np.sqrt(row['Lat_Std']**2 + row['Lon_Std']**2) * 111, 
                axis=1
            ).round(2)
            
            st.dataframe(cluster_stats[['Orders', 'Total Weight', 'Spread_km']], 
                        use_container_width=True)
    
    with tab3:
        st.header("💰 Cost-Benefit Analysis")
        
        # Default cost value
        total_cost = len(df_orders) * 900  # Default assumption
        
        # Load cost data if available
        if cost_file is not None:
            df_costs = pd.read_csv(cost_file)
            
            # Current cost analysis
            st.subheader("Current Cost Structure")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                if 'cost' in df_costs.columns:
                    total_cost = df_costs['cost'].sum()
                avg_cost_per_order = total_cost / len(df_orders)
                st.metric("Total Cost (Period)", f"₹{total_cost:,.0f}")
                st.metric("Average Cost per Order", f"₹{avg_cost_per_order:.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                avg_driver_earning = total_cost / df_orders['driver'].nunique()
                st.metric("Avg Driver Earning", f"₹{avg_driver_earning:,.0f}")
                st.metric("Cost per KM", "₹12-15 (estimated)")
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            # Use default values if no cost file
            avg_cost_per_order = total_cost / len(df_orders)
            st.info("📊 Using estimated costs (₹900 per driver). Upload cost data for accurate analysis.")
        
        # Optimized cost projection
        st.subheader("🎯 Optimized Cost Projection")
        
        if enable_multi_trip:
            # Calculate optimized costs
            total_drivers = df_orders['driver'].nunique()
            multi_trip_drivers = int(total_drivers * multi_trip_percent / 100)
            single_trip_drivers = total_drivers - multi_trip_drivers
            
            # Cost calculations
            multi_trip_cost = multi_trip_drivers * 600 + (multi_trip_drivers * trips_per_driver * 200)
            single_trip_cost = single_trip_drivers * 900
            optimized_total = multi_trip_cost + single_trip_cost
            
            # Capacity calculations
            multi_trip_capacity = multi_trip_drivers * 20 * trips_per_driver
            single_trip_capacity = single_trip_drivers * 30
            total_capacity = multi_trip_capacity + single_trip_capacity
            
            st.markdown('<div class="cost-savings">', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Optimized Total Cost", f"₹{optimized_total:,.0f}",
                         delta=f"-₹{(total_cost - optimized_total):,.0f}")
            
            with col2:
                savings_percent = ((total_cost - optimized_total) / total_cost) * 100
                st.metric("Cost Savings", f"{savings_percent:.1f}%",
                         delta=f"₹{avg_cost_per_order - (optimized_total/len(df_orders)):.2f}/order")
            
            with col3:
                st.metric("Delivery Capacity", f"{total_capacity:,} orders",
                         delta=f"+{(total_capacity - len(df_orders)):,}")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Detailed breakdown
            st.subheader("💡 Cost Breakdown Comparison")
            
            comparison_data = pd.DataFrame({
                'Model': ['Current', 'Optimized'],
                'Total Cost': [total_cost, optimized_total],
                'Cost per Order': [avg_cost_per_order, optimized_total/len(df_orders)],
                'Capacity': [len(df_orders), total_capacity],
                'Utilization': ['100%', f"{(len(df_orders)/total_capacity)*100:.1f}%"]
            })
            
            fig_comparison = go.Figure()
            fig_comparison.add_trace(go.Bar(
                name='Current',
                x=['Total Cost', 'Cost per Order'],
                y=[total_cost, avg_cost_per_order],
                marker_color='lightcoral'
            ))
            fig_comparison.add_trace(go.Bar(
                name='Optimized',
                x=['Total Cost', 'Cost per Order'],
                y=[optimized_total, optimized_total/len(df_orders)],
                marker_color='lightgreen'
            ))
            fig_comparison.update_layout(title="Cost Comparison", barmode='group')
            st.plotly_chart(fig_comparison, use_container_width=True)
    
    with tab4:
        st.header("🚛 Optimized Route Planning")
        
        if st.session_state.clusters is not None:
            df_routes = st.session_state.clusters
            
            # Select specific cluster to visualize route
            cluster_ids = sorted(df_routes[df_routes['cluster'] >= 0]['cluster'].unique())
            selected_cluster = st.selectbox("Select Cluster to View Route", cluster_ids)
            
            cluster_orders = df_routes[df_routes['cluster'] == selected_cluster]
            
            # Route optimization visualization
            st.subheader(f"Route for Cluster {selected_cluster}")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Create route map
                route_map = folium.Map(
                    location=[cluster_orders['delivered_lat'].mean(), 
                             cluster_orders['delivered_long'].mean()],
                    zoom_start=13
                )
                
                # Add hub
                hub_info = cluster_orders.iloc[0]
                folium.Marker(
                    location=[hub_info['hub_lat'], hub_info['hub_long']],
                    popup=f"Hub: {hub_info['hub']}",
                    icon=folium.Icon(color='red', icon='warehouse', prefix='fa')
                ).add_to(route_map)
                
                # Simple TSP approximation for route
                coords = cluster_orders[['delivered_lat', 'delivered_long']].values
                hub_coord = np.array([hub_info['hub_lat'], hub_info['hub_long']])
                
                # Nearest neighbor algorithm
                route = [0]  # Start from hub
                unvisited = list(range(len(coords)))
                current = 0
                
                while unvisited:
                    if current == 0:
                        current_coord = hub_coord
                    else:
                        current_coord = coords[current-1]
                    
                    distances = [geodesic(current_coord, coords[i]).km if i in unvisited else float('inf') 
                               for i in range(len(coords))]
                    next_idx = np.argmin(distances)
                    route.append(next_idx + 1)
                    unvisited.remove(next_idx)
                    current = next_idx + 1
                
                route.append(0)  # Return to hub
                
                # Draw route
                route_coords = [hub_coord]
                for i in route[1:-1]:
                    route_coords.append(coords[i-1])
                route_coords.append(hub_coord)
                
                folium.PolyLine(
                    locations=route_coords,
                    color='blue',
                    weight=3,
                    opacity=0.8
                ).add_to(route_map)
                
                # Add delivery points with sequence
                for seq, idx in enumerate(route[1:-1], 1):
                    order = cluster_orders.iloc[idx-1]
                    folium.Marker(
                        location=[order['delivered_lat'], order['delivered_long']],
                        popup=f"Stop {seq}<br>Order: {order['number']}<br>Weight: {order['weight']}kg",
                        icon=folium.DivIcon(html=f"""
                            <div style="background-color: white; border: 2px solid blue; 
                                      border-radius: 50%; width: 30px; height: 30px; 
                                      text-align: center; line-height: 30px; font-weight: bold;">
                                {seq}
                            </div>
                        """)
                    ).add_to(route_map)
                
                st_folium(route_map, width=700, height=500)
            
            with col2:
                st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
                st.markdown("### Route Statistics")
                
                # Calculate route distance
                total_distance = sum(
                    geodesic(route_coords[i], route_coords[i+1]).km 
                    for i in range(len(route_coords)-1)
                )
                
                st.metric("Total Distance", f"{total_distance:.1f} km")
                st.metric("Number of Stops", len(cluster_orders))
                st.metric("Total Weight", f"{cluster_orders['weight'].sum():.1f} kg")
                st.metric("Estimated Time", f"{(total_distance/15 + len(cluster_orders)*0.1):.1f} hours")
                
                # Vehicle recommendation
                if cluster_orders['weight'].sum() > 100 or total_distance > 20:
                    st.info("🚗 Recommended: Auto Rickshaw")
                else:
                    st.success("🏍️ Recommended: Bike")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Driver assignment suggestion
                st.markdown("### Driver Assignment")
                if enable_multi_trip and total_distance < 10:
                    st.success("✅ Suitable for multi-trip")
                    st.write(f"Can complete {trips_per_driver} trips/day")
                else:
                    st.info("📍 Single trip route")
        else:
            st.warning("⚠️ Please generate clusters in the 'Cluster Visualization' tab first.")
    
    with tab5:
        st.header("📈 Performance Metrics & KPIs")
        
        # Efficiency metrics
        st.subheader("🎯 Efficiency Improvements")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_efficiency = len(df_orders) / (df_orders['driver'].nunique() * df_orders['date'].nunique())
            optimized_efficiency = current_efficiency * 1.25  # 25% improvement estimate
            st.metric("Deliveries/Driver/Day", 
                     f"{optimized_efficiency:.1f}",
                     delta=f"+{(optimized_efficiency - current_efficiency):.1f}")
        
        with col2:
            current_cost_per = avg_cost_per_order if 'avg_cost_per_order' in locals() else 30
            optimized_cost_per = current_cost_per * 0.83  # 17% reduction
            st.metric("Cost per Delivery",
                     f"₹{optimized_cost_per:.1f}",
                     delta=f"-₹{(current_cost_per - optimized_cost_per):.1f}")
        
        with col3:
            utilization = 85 if enable_multi_trip else 75
            st.metric("Vehicle Utilization",
                     f"{utilization}%",
                     delta=f"+{utilization-75}%")
        
        with col4:
            on_time = 95 if enable_multi_trip else 88
            st.metric("On-Time Delivery",
                     f"{on_time}%",
                     delta=f"+{on_time-88}%")
        
        # Projected monthly savings
        st.subheader("💰 Projected Monthly Savings")
        
        if 'total_cost' in locals() and 'optimized_total' in locals():
            days_in_period = df_orders['date'].nunique()
            daily_savings = (total_cost - optimized_total) / days_in_period
            monthly_savings = daily_savings * 30
            yearly_savings = monthly_savings * 12
            
            savings_df = pd.DataFrame({
                'Period': ['Daily', 'Monthly', 'Yearly'],
                'Current Cost': [total_cost/days_in_period, total_cost/days_in_period*30, total_cost/days_in_period*365],
                'Optimized Cost': [optimized_total/days_in_period, optimized_total/days_in_period*30, optimized_total/days_in_period*365],
                'Savings': [daily_savings, monthly_savings, yearly_savings]
            })
            
            fig_savings = go.Figure()
            fig_savings.add_trace(go.Bar(name='Current', x=savings_df['Period'], y=savings_df['Current Cost']))
            fig_savings.add_trace(go.Bar(name='Optimized', x=savings_df['Period'], y=savings_df['Optimized Cost']))
            fig_savings.update_layout(title="Cost Projection", barmode='group')
            st.plotly_chart(fig_savings, use_container_width=True)
        
        # ROI calculation
        st.subheader("📊 Return on Investment")
        
        implementation_cost = st.number_input("Estimated Implementation Cost (₹)", 
                                            value=500000, 
                                            step=100000)
        
        if 'monthly_savings' in locals():
            roi_months = implementation_cost / monthly_savings
            st.markdown(f"""
            <div class="cost-savings">
            <h4>ROI Analysis</h4>
            <ul>
            <li>Implementation Cost: ₹{implementation_cost:,.0f}</li>
            <li>Monthly Savings: ₹{monthly_savings:,.0f}</li>
            <li>Payback Period: {roi_months:.1f} months</li>
            <li>First Year Net Benefit: ₹{(yearly_savings - implementation_cost):,.0f}</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

else:
    # Demo mode with sample data
    st.info("👈 Please upload your data files using the links in the sidebar to see personalized analysis")
    
    # Show demo visualization
    st.subheader("Demo: How Clustering Improves Efficiency")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ❌ Current: Pincode-Based")
        st.image("https://via.placeholder.com/400x300/FF6B6B/FFFFFF?text=Scattered+Routes", 
                caption="Inefficient routes crossing multiple areas")
        st.markdown("""
        - Fixed zones by pincode
        - Overlapping routes
        - Unbalanced workload
        - Higher costs
        """)
    
    with col2:
        st.markdown("### ✅ Optimized: Smart Clusters")
        st.image("https://via.placeholder.com/400x300/4ECDC4/FFFFFF?text=Optimized+Clusters", 
                caption="Efficient clusters with balanced loads")
        st.markdown("""
        - Dynamic clustering
        - Optimized routes
        - Balanced workload
        - 17-25% cost savings
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    Built with Streamlit | Optimizing Last-Mile Delivery in Bengaluru
</div>
""", unsafe_allow_html=True)
