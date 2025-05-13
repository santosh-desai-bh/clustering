import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import io
import urllib.request
from urllib.error import URLError
import tempfile
import os
import base64
from datetime import datetime
import random

# Set page configuration
st.set_page_config(
    page_title="Last Mile Delivery Zone Optimizer",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for improved appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        color: #2c3e50;
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: bold;
        margin-top: 2rem;
        color: #34495e;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: bold;
        margin-top: 1.5rem;
        color: #3498db;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #2980b9;
    }
    .metric-label {
        color: #7f8c8d;
        font-size: 0.9rem;
    }
    .insight-box {
        background-color: #e8f4f8;
        border-left: 5px solid #3498db;
        padding: 10px 15px;
        margin: 10px 0;
        border-radius: 0 5px 5px 0;
    }
</style>
""", unsafe_allow_html=True)

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate the Haversine distance between two points in kilometers"""
    # Convert latitude and longitude from degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # Radius of Earth in kilometers
    return c * r

def create_delivery_zones(df, num_zones=15, progress_bar=None):
    """
    Create delivery zones from a DataFrame
    
    Args:
        df: DataFrame containing delivery data
        num_zones: Number of zones to create
        progress_bar: Optional Streamlit progress bar
        
    Returns:
        zones: Dictionary containing zone information
        df: The processed DataFrame with zone assignments
    """    
    # Basic validation
    required_cols = ['delivered_lat', 'delivered_long', 'hub', 'postcode']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.error(f"Missing required columns: {', '.join(missing_cols)}")
        
        # Try to map common column names
        col_mapping = {
            'delivered_lat': ['lat', 'latitude', 'dest_lat', 'customer_lat'],
            'delivered_long': ['long', 'lng', 'longitude', 'dest_long', 'customer_long'],
            'hub': ['warehouse', 'dc', 'distribution_center', 'origin', 'source'],
            'postcode': ['pincode', 'zipcode', 'pin', 'postal_code']
        }
        
        # Try to find matching columns
        for req_col in missing_cols:
            for df_col in df.columns:
                if df_col.lower() in [c.lower() for c in col_mapping.get(req_col, [])]:
                    df = df.rename(columns={df_col: req_col})
                    st.info(f"Mapped column '{df_col}' to '{req_col}'")
        
        # Check again after mapping
        still_missing = [col for col in required_cols if col not in df.columns]
        if still_missing:
            st.error(f"Still missing required columns: {', '.join(still_missing)}")
            return None, None
    
    # Clean the data
    if progress_bar:
        progress_bar.progress(0.1)
        
    # Convert coordinates to numeric values
    for col in ['delivered_lat', 'delivered_long', 'hub_lat', 'hub_long']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with missing coordinates
    orig_len = len(df)
    df = df.dropna(subset=['delivered_lat', 'delivered_long'])
    
    if orig_len > len(df):
        st.warning(f"Removed {orig_len - len(df)} rows with missing coordinates")
    
    # Extract delivery points
    delivery_points = list(zip(df['delivered_lat'], df['delivered_long']))
    
    # Perform K-means clustering
    if progress_bar:
        progress_bar.progress(0.2)
        
    # Select initial centroids randomly
    random.seed(42)  # For reproducibility
    indices = random.sample(range(len(delivery_points)), num_zones)
    centroids = [delivery_points[i] for i in indices]
    
    # Implement K-means algorithm
    MAX_ITERATIONS = 20
    for iteration in range(MAX_ITERATIONS):
        if progress_bar:
            progress_value = 0.2 + (0.6 * (iteration + 1) / MAX_ITERATIONS)
            progress_bar.progress(progress_value)
        
        # Assign each point to the nearest centroid
        clusters = []
        for lat, lon in delivery_points:
            distances = [calculate_distance(lat, lon, c_lat, c_lon) for c_lat, c_lon in centroids]
            closest_cluster = distances.index(min(distances))
            clusters.append(closest_cluster)
        
        # Update centroids
        new_centroids = []
        for i in range(num_zones):
            cluster_points = [delivery_points[j] for j in range(len(delivery_points)) if clusters[j] == i]
            if cluster_points:
                avg_lat = sum(point[0] for point in cluster_points) / len(cluster_points)
                avg_lon = sum(point[1] for point in cluster_points) / len(cluster_points)
                new_centroids.append((avg_lat, avg_lon))
            else:
                # If a cluster is empty, keep the old centroid
                new_centroids.append(centroids[i])
        
        # Check for convergence
        if all(calculate_distance(old[0], old[1], new[0], new[1]) < 0.0001 for old, new in zip(centroids, new_centroids)):
            break
        
        centroids = new_centroids
    
    # Add cluster assignments to the dataframe
    df['zone'] = clusters
    
    # Analyze the zones
    if progress_bar:
        progress_bar.progress(0.9)
        
    zone_info = {}
    
    for zone_id in range(num_zones):
        zone_df = df[df['zone'] == zone_id]
        
        if len(zone_df) == 0:
            continue  # Skip empty zones
            
        # Count deliveries per hub in this zone
        hub_counts = zone_df['hub'].value_counts()
        primary_hub = hub_counts.index[0] if not hub_counts.empty else "Unknown"
        
        # Count postcodes in this zone
        postcode_counts = zone_df['postcode'].value_counts()
        top_postcodes = list(postcode_counts.head(5).index) if not postcode_counts.empty else []
        
        # Calculate average distance from centroid
        centroid = centroids[zone_id]
        distances = []
        for _, row in zone_df.iterrows():
            dist = calculate_distance(row['delivered_lat'], row['delivered_long'], centroid[0], centroid[1])
            distances.append(dist)
        
        avg_distance = sum(distances) / len(distances) if distances else 0
        
        # Store zone information
        zone_info[zone_id] = {
            'centroid': centroid,
            'size': len(zone_df),
            'primary_hub': primary_hub,
            'top_postcodes': top_postcodes,
            'avg_radius_km': avg_distance,
            'hub_distribution': hub_counts.to_dict()
        }
    
    if progress_bar:
        progress_bar.progress(1.0)
        
    return zone_info, df

def analyze_optimization(df, zone_info):
    """Analyze potential optimization from zone-based delivery"""
    
    # Count total deliveries
    total_deliveries = len(df)
    
    # Calculate how many deliveries would change hubs
    reassignments = 0
    for _, row in df.iterrows():
        zone_id = row['zone']
        current_hub = row['hub']
        optimal_hub = zone_info.get(zone_id, {}).get('primary_hub')
        
        if optimal_hub and current_hub != optimal_hub:
            reassignments += 1
    
    optimization_pct = (reassignments / total_deliveries) * 100 if total_deliveries > 0 else 0
    
    return {
        'total_deliveries': total_deliveries,
        'optimal_assignments': total_deliveries - reassignments,
        'potential_reassignments': reassignments,
        'optimization_percentage': optimization_pct
    }

def create_zone_visualization(df, zone_info):
    """Create a visualization of delivery zones"""
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Generate colors for zones
    unique_zones = sorted(zone_info.keys())
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_zones)))
    
    # Plot delivery points by zone
    for i, zone_id in enumerate(unique_zones):
        zone_data = df[df['zone'] == zone_id]
        ax.scatter(
            zone_data['delivered_long'], 
            zone_data['delivered_lat'],
            s=10,
            alpha=0.5,
            color=colors[i],
            label=f'Zone {zone_id+1}'
        )
    
    # Plot zone centroids
    for zone_id, info in zone_info.items():
        centroid = info['centroid']
        ax.scatter(centroid[1], centroid[0], marker='X', color='black', s=100)
        ax.annotate(f'Z{zone_id+1}', (centroid[1], centroid[0]), fontsize=12)
    
    # Plot hubs
    hub_data = {}
    for hub in df['hub'].unique():
        hub_points = df[df['hub'] == hub]
        if 'hub_lat' in df.columns and 'hub_long' in df.columns:
            # Use actual hub coordinates
            hub_coords = hub_points[['hub_lat', 'hub_long']].iloc[0]
            hub_data[hub] = (hub_coords['hub_lat'], hub_coords['hub_long'])
        else:
            # Estimate hub location as average of its delivery points
            avg_lat = hub_points['delivered_lat'].mean()
            avg_long = hub_points['delivered_long'].mean()
            hub_data[hub] = (avg_lat, avg_long)
    
    for hub, (lat, lon) in hub_data.items():
        hub_name = hub.split('[')[0] if '[' in hub else hub
        ax.scatter(lon, lat, marker='*', color='red', s=200)
        ax.annotate(hub_name, (lon, lat), fontsize=10, fontweight='bold')
    
    # Add labels and title
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Delivery Zone Clusters')
    
    # Add legend (but limit the number of items to prevent overcrowding)
    if len(unique_zones) > 10:
        # Just show a subset of zones in the legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:10], labels[:10], loc='upper right', title="Zones (showing 10)")
    else:
        ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    # Convert figure to image
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    plt.close(fig)
    
    return buf

def get_table_download_link(df, filename="data.csv", text="Download CSV"):
    """Generate a download link for a dataframe"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def get_image_download_link(img_buf, filename="image.png", text="Download Image"):
    """Generate a download link for an image"""
    b64 = base64.b64encode(img_buf.getvalue()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">{text}</a>'
    return href

def fetch_from_url(url):
    """Fetch data from a URL"""
    try:
        with st.spinner("Downloading data from URL..."):
            # Create a temporary file
            temp_fd, temp_path = tempfile.mkstemp(suffix='.csv')
            os.close(temp_fd)
            
            # Download the file
            urllib.request.urlretrieve(url, temp_path)
            
            # Read the CSV
            df = pd.read_csv(temp_path)
            
            # Remove the temporary file
            os.unlink(temp_path)
            
            return df
    except Exception as e:
        st.error(f"Error downloading or reading data: {e}")
        return None

def load_demo_data():
    """Create synthetic demo data"""
    # Base coordinates for a city (e.g., Bangalore)
    base_lat, base_long = 12.97, 77.59
    
    # Create hubs
    hubs = {
        "Mahadevapura Hub": (12.97, 77.70),
        "Banashankari Hub": (12.91, 77.60),
        "Chandra Layout Hub": (12.96, 77.52),
        "Hebbal Hub": (13.07, 77.61),
        "Koramangala Hub": (12.93, 77.62),
        "Kudlu Hub": (12.92, 77.67)
    }
    
    # Generate synthetic data
    np.random.seed(42)
    data = []
    
    for _ in range(1000):
        # Randomly select a hub
        hub_name = random.choice(list(hubs.keys()))
        hub_lat, hub_long = hubs[hub_name]
        
        # Generate delivery point near the hub (within ~5km)
        delivered_lat = hub_lat + np.random.normal(0, 0.03)
        delivered_long = hub_long + np.random.normal(0, 0.03)
        
        # Generate random postcode
        postcode = np.random.randint(560001, 560100)
        
        # Random weight
        weight = np.random.exponential(2)
        if weight > 20:
            weight = 20
        
        # Calculate distance
        distance = calculate_distance(hub_lat, hub_long, delivered_lat, delivered_long)
        
        # Add to dataset
        data.append({
            "number": f"ORD-{np.random.randint(10000, 99999)}",
            "created_date": (datetime.now() - pd.Timedelta(days=np.random.randint(1, 30))).strftime("%Y-%m-%d %H:%M:%S"),
            "driver": random.choice(["Amit", "Rajesh", "Suresh", "Prakash", "Vijay", "Ajay"]),
            "vehicle_model": random.choice(["Bike", "Auto Rickshaw", "Van"]),
            "hub": hub_name,
            "customer": random.choice(["Amazon", "Flipkart", "Myntra", "BigBasket", "Swiggy", "Zomato"]),
            "postcode": postcode,
            "hub_lat": hub_lat,
            "hub_long": hub_long,
            "delivered_lat": delivered_lat,
            "delivered_long": delivered_long,
            "weight": round(weight, 2),
            "kms": round(distance, 2)
        })
    
    return pd.DataFrame(data)

# App header
st.markdown('<div class="main-header">Last Mile Delivery Zone Optimizer</div>', unsafe_allow_html=True)
st.markdown("""
This application helps you analyze last mile delivery data and optimize your delivery strategy by 
transitioning from pincode-based to zone-based delivery assignments.
""")

# Sidebar
st.sidebar.header("Configuration")

# Data source selection
data_source = st.sidebar.radio(
    "Select data source:",
    ["Upload CSV", "Blowhorn Analytics URL", "Demo Data"]
)

df = None

if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload delivery data CSV", type=['csv'])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success(f"Loaded {len(df)} records")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")

elif data_source == "Blowhorn Analytics URL":
    url_input = st.sidebar.text_input(
        "Enter Blowhorn Analytics URL",
        "https://analytics.blowhorn.com/question/3120-if-network-analysis-last-mile?start=2025-05-01&end=2025-05-13"
    )
    
    if st.sidebar.button("Fetch Data"):
        df = fetch_from_url(url_input)
        if df is not None:
            st.sidebar.success(f"Loaded {len(df)} records from URL")

else:  # Demo Data
    st.sidebar.info("Using demo data with synthetic delivery points")
    df = load_demo_data()

# Clustering parameters
st.sidebar.header("Zoning Parameters")
num_zones = st.sidebar.slider("Number of delivery zones", 5, 30, 15)

# Only proceed if we have data
if df is not None:
    # Display basic info about the data
    st.markdown('<div class="section-header">Data Summary</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Unique Hubs", df['hub'].nunique() if 'hub' in df.columns else "N/A")
    with col3:
        st.metric("Unique Postcodes", df['postcode'].nunique() if 'postcode' in df.columns else "N/A")
    
    # Run analysis on button click
    if st.button("Run Zone Analysis"):
        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Performing clustering analysis...")
        
        # Create zones
        zone_info, df_with_zones = create_delivery_zones(df, num_zones=num_zones, progress_bar=progress_bar)
        
        if zone_info is None:
            st.error("Zone analysis failed. Please check your data and try again.")
        else:
            status_text.text("Analysis complete!")
            
            # Create tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Zone Map", "Detailed Analysis", "Recommendations"])
            
            # Dashboard tab
            with tab1:
                st.markdown('<div class="sub-header">Delivery Zone Dashboard</div>', unsafe_allow_html=True)
                
                # Calculate optimization metrics
                opt_metrics = analyze_optimization(df_with_zones, zone_info)
                
                # Show key metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-value">{opt_metrics["total_deliveries"]:,}</div>', unsafe_allow_html=True)
                    st.markdown('<div class="metric-label">Total Deliveries</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-value">{num_zones}</div>', unsafe_allow_html=True)
                    st.markdown('<div class="metric-label">Optimized Zones</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col3:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-value">{opt_metrics["optimization_percentage"]:.1f}%</div>', unsafe_allow_html=True)
                    st.markdown('<div class="metric-label">Potential Efficiency Gain</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Hub assignment chart
                st.markdown('<div class="section-header">Current vs. Optimized Hub Assignments</div>', unsafe_allow_html=True)
                
                # Create hub assignment data for pie chart
                hub_assignment_data = pd.DataFrame([
                    {"Status": "Optimal Assignment", "Count": opt_metrics["optimal_assignments"]},
                    {"Status": "Potential Reassignment", "Count": opt_metrics["potential_reassignments"]}
                ])
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig, ax = plt.subplots(figsize=(8, 8))
                    ax.pie(
                        hub_assignment_data["Count"],
                        labels=hub_assignment_data["Status"],
                        autopct='%1.1f%%',
                        colors=['#2ecc71', '#e74c3c'],
                        startangle=90,
                        explode=(0, 0.1)
                    )
                    ax.axis('equal')
                    st.pyplot(fig)
                
                with col2:
                    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
                    st.markdown(f"""
                    ### Key Insights
                    
                    - **{opt_metrics["optimal_assignments"]:,}** deliveries ({100-opt_metrics["optimization_percentage"]:.1f}%) are already assigned to their optimal hub.
                    - **{opt_metrics["potential_reassignments"]:,}** deliveries ({opt_metrics["optimization_percentage"]:.1f}%) could be reassigned to a closer hub.
                    - Transitioning to zone-based delivery could improve efficiency by approximately **{opt_metrics["optimization_percentage"]:.1f}%**.
                    """)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Zone size distribution
                st.markdown('<div class="section-header">Zone Size Distribution</div>', unsafe_allow_html=True)
                
                zone_sizes = pd.DataFrame([
                    {"Zone ID": f"Zone {zone_id+1}", "Deliveries": info["size"]}
                    for zone_id, info in zone_info.items()
                ]).sort_values("Deliveries", ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(zone_sizes["Zone ID"], zone_sizes["Deliveries"], color='#3498db')
                ax.set_ylabel("Number of Deliveries")
                ax.set_title("Deliveries per Zone")
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
                
                # Hub distribution by zone
                st.markdown('<div class="section-header">Hub Distribution by Zone</div>', unsafe_allow_html=True)
                
                # Prepare data for heatmap
                hubs = sorted(df_with_zones['hub'].unique())
                heat_data = np.zeros((len(zone_info), len(hubs)))
                
                for i, zone_id in enumerate(sorted(zone_info.keys())):
                    hub_dist = df_with_zones[df_with_zones['zone'] == zone_id]['hub'].value_counts()
                    for j, hub in enumerate(hubs):
                        heat_data[i, j] = hub_dist.get(hub, 0)
                
                fig, ax = plt.subplots(figsize=(12, 8))
                im = ax.imshow(heat_data, cmap='YlGnBu')
                
                # Add labels
                ax.set_xticks(np.arange(len(hubs)))
                ax.set_yticks(np.arange(len(zone_info)))
                ax.set_xticklabels([h.split('[')[0] if '[' in h else h for h in hubs])
                ax.set_yticklabels([f"Zone {z+1}" for z in sorted(zone_info.keys())])
                
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                
                # Add colorbar
                cbar = ax.figure.colorbar(im, ax=ax)
                cbar.ax.set_ylabel("Number of Deliveries", rotation=-90, va="bottom")
                
                ax.set_title("Hub Distribution by Zone")
                fig.tight_layout()
                
                st.pyplot(fig)
            
            # Zone Map tab
            with tab2:
                st.markdown('<div class="sub-header">Delivery Zone Map</div>', unsafe_allow_html=True)
                
                # Create visualization
                img_buf = create_zone_visualization(df_with_zones, zone_info)
                
                # Display image
                st.image(img_buf, caption="Delivery Zones Map", use_column_width=True)
                
                # Add download link
                st.markdown(get_image_download_link(img_buf, "delivery_zones.png", "Download Zone Map"), unsafe_allow_html=True)
                
                # Add map explanation
                st.markdown("""
                ### Map Legend:
                - **Colored dots**: Delivery points, colored by zone
                - **Black X markers**: Zone centroids
                - **Red stars**: Hub locations
                
                This map shows how delivery points are clustered into zones based on geographic proximity. The zone
                centroids (black X markers) represent the center point of each zone, and can be used as reference points
                for zone boundaries.
                """)
            
            # Detailed Analysis tab
            with tab3:
                st.markdown('<div class="sub-header">Detailed Zone Analysis</div>', unsafe_allow_html=True)
                
                # Create a DataFrame with zone details
                zone_details = []
                for zone_id, info in zone_info.items():
                    zone_details.append({
                        "Zone ID": f"Zone {zone_id+1}",
                        "Size": info["size"],
                        "Primary Hub": info["primary_hub"],
                        "Avg Radius (km)": round(info["avg_radius_km"], 2),
                        "Top Postcodes": ", ".join(str(p) for p in info["top_postcodes"][:3])
                    })
                
                zone_df = pd.DataFrame(zone_details)
                st.dataframe(zone_df)
                
                # Download link for zone details
                st.markdown(get_table_download_link(zone_df, "zone_analysis.csv", "Download Zone Analysis"), unsafe_allow_html=True)
                
                # Postcode distribution within zones
                st.markdown('<div class="section-header">Postcode Distribution within Zones</div>', unsafe_allow_html=True)
                
                # Create a map of postcodes to zones
                postcode_zones = {}
                for _, row in df_with_zones.iterrows():
                    postcode = row['postcode']
                    zone = row['zone']
                    
                    if postcode not in postcode_zones:
                        postcode_zones[postcode] = {}
                    
                    postcode_zones[postcode][zone] = postcode_zones[postcode].get(zone, 0) + 1
                
                # Find postcodes that span multiple zones
                multi_zone_postcodes = []
                for postcode, zones in postcode_zones.items():
                    if len(zones) > 1:
                        total = sum(zones.values())
                        multi_zone_postcodes.append({
                            "Postcode": postcode,
                            "Zones": len(zones),
                            "Total Deliveries": total,
                            "Distribution": ", ".join(f"Zone {z+1}: {c/total*100:.1f}%" for z, c in zones.items())
                        })
                
                if multi_zone_postcodes:
                    multi_zone_df = pd.DataFrame(multi_zone_postcodes)
                    multi_zone_df = multi_zone_df.sort_values("Zones", ascending=False)
                    
                    st.dataframe(multi_zone_df)
                    st.markdown("""
                    ### Insight:
                    Postcodes spanning multiple zones indicate areas where zone-based delivery would be more efficient
                    than the current pincode-based system. These postcodes currently force deliveries to be processed
                    through a single hub due to pincode restrictions, when geographically they would be better served by
                    different hubs.
                    """)
                else:
                    st.info("No postcodes span multiple zones in the current clustering.")
                
                # Hub to Zone Assignment Analysis
                st.markdown('<div class="section-header">Hub to Zone Assignment Analysis</div>', unsafe_allow_html=True)
                
                # Create a map of primary zones for each hub
                hub_primary_zones = {}
                for zone_id, info in zone_info.items():
                    hub = info["primary_hub"]
                    if hub not in hub_primary_zones:
                        hub_primary_zones[hub] = []
                    hub_primary_zones[hub].append(zone_id)
                
                # Display hub-zone allocations
                for hub, zones in hub_primary_zones.items():
                    zones_str = ", ".join(f"Zone {z+1}" for z in zones)
                    st.write(f"**{hub}** is the primary hub for: {zones_str}")
                
                # Calculate workload balance
                current_hub_counts = df_with_zones['hub'].value_counts()
                
                optimal_hub_counts = {}
                for zone_id, info in zone_info.items():
                    hub = info["primary_hub"]
                    optimal_hub_counts[hub] = optimal_hub_counts.get(hub, 0) + info["size"]
                
                optimal_hub_df = pd.DataFrame({
                    "Hub": list(optimal_hub_counts.keys()),
                    "Current Deliveries": [current_hub_counts.get(hub, 0) for hub in optimal_hub_counts.keys()],
                    "Optimal Deliveries": list(optimal_hub_counts.values())
                })
                
                st.dataframe(optimal_hub_df)
                
                # Create visualization of workload balance
                fig, ax = plt.subplots(figsize=(10, 6))
                
                x = np.arange(len(optimal_hub_df))
                width = 0.35
                
                ax.bar(x - width/2, optimal_hub_df["Current Deliveries"], width, label="Current")
                ax.bar(x + width/2, optimal_hub_df["Optimal Deliveries"], width, label="Optimal")
                
                ax.set_xticks(x)
                ax.set_xticklabels([h.split('[')[0] if '[' in h else h for h in optimal_hub_df["Hub"]])
                ax.legend()
                
                plt.xticks(rotation=45, ha='right')
                plt.title("Current vs. Optimal Hub Workload")
                plt.tight_layout()
                
                st.pyplot(fig)
                
                # Download full dataset with zone assignments
                st.markdown('<div class="section-header">Download Full Dataset with Zone Assignments</div>', unsafe_allow_html=True)
                st.markdown(get_table_download_link(df_with_zones, "delivery_data_with_zones.csv", "Download Full Dataset"), unsafe_allow_html=True)
            
            # Recommendations tab
            with tab4:
                st.markdown('<div class="sub-header">Implementation Recommendations</div>', unsafe_allow_html=True)
                
                st.markdown("""
                Based on the analysis of your delivery data, here's a strategic approach to implementing zone-based delivery:
                
                ### Phase 1: Planning & Preparation (1-2 months)
                
                1. **Define Zone Boundaries**
                   - Use the clustering analysis to establish clear zone boundaries
                   - Create a comprehensive zone map for operational use
                   - Develop a zone code system for easy reference
                
                2. **Technology Updates**
                   - Update routing software to support zone-based assignment
                   - Develop zone lookup tools for customer service teams
                   - Create visualization dashboards for monitoring zone performance
                
                3. **Training & Documentation**
                   - Train dispatchers on zone-based assignment principles
                   - Update driver onboarding materials with zone information
                   - Develop quick reference materials for all stakeholders
                
                ### Phase 2: Pilot Implementation (2-3 months)
                
                1. **Select Pilot Zones**
                   - Begin with zones that show highest potential for improvement
                   - Include a mix of high and low density areas
                   - Ensure representation of different hub territories
                
                2. **Operational Changes**
                   - Implement dynamic hub assignment based on delivery zone
                   - Maintain pincode system in parallel during transition
                   - Set up daily reviews of pilot performance
                
                3. **Monitoring & Metrics**
                   - Track key metrics:
                     - Delivery time
                     - Hub workload balance
                     - Driver efficiency
                     - Customer satisfaction
                
                ### Phase 3: Full Rollout (3-6 months)
                
                1. **Expand Zone Implementation**
                   - Gradually convert all areas to zone-based delivery
                   - Prioritize areas with highest expected improvements
                   - Maintain contingency plans for each phase of rollout
                
                2. **Optimization & Adjustment**
                   - Continuously refine zone boundaries based on operational data
                   - Implement seasonal or time-of-day zone variations if needed
                   - Develop zone splitting/merging protocols for demand fluctuations
                
                3. **Customer Communication**
                   - Update customer-facing systems to reflect zone-based delivery
                   - Develop clear messaging around delivery times by zone
                   - Leverage improved delivery precision for customer experience
                """)
                
                st.markdown('<div class="section-header">Potential Challenges & Mitigations</div>', unsafe_allow_html=True)
                
                challenges_df = pd.DataFrame([
                    {
                        "Challenge": "Driver familiarity with new zones",
                        "Impact": "High",
                        "Mitigation": "Provide detailed zone maps, GPS guidance, and gradual transition period for drivers to learn new territories."
                    },
                    {
                        "Challenge": "Technology integration issues",
                        "Impact": "Medium",
                        "Mitigation": "Conduct thorough testing of all systems before implementation, maintain parallel systems temporarily."
                    },
                    {
                        "Challenge": "Hub capacity constraints",
                        "Impact": "High",
                        "Mitigation": "Analyze hub capacity carefully before reassignment, implement gradual transition to avoid sudden shifts."
                    },
                    {
                        "Challenge": "Customer confusion during transition",
                        "Impact": "Low",
                        "Mitigation": "Maintain delivery SLAs, provide proactive communication about improvements in delivery precision."
                    },
                    {
                        "Challenge": "Fluctuating demand by zone",
                        "Impact": "Medium",
                        "Mitigation": "Implement dynamic zone boundaries that can adjust based on seasonal or daily demand patterns."
                    }
                ])
                
                st.dataframe(challenges_df)
                
                st.markdown('<div class="section-header">Final Recommendations</div>', unsafe_allow_html=True)
                
                st.markdown("""
                ### Key Takeaways
                
                1. **Zone-Based Delivery Superiority**: The analysis clearly demonstrates that zone-based delivery offers significant 
                   advantages over pincode-based systems in terms of efficiency, hub workload balance, and delivery distance optimization.
                
                2. **Gradual Implementation**: The transition should be implemented in phases, starting with pilot zones that show the 
                   highest potential for improvement, with continuous monitoring and adjustment.
                
                3. **Technology Integration**: Success depends on proper integration with routing and dispatch systems, with focus on 
                   making zone information easily accessible to all stakeholders.
                
                4. **Driver Experience**: Prioritize driver training and provide tools to ease the transition, as driver familiarity 
                   with territories is crucial for success.
                
                5. **Dynamic Adjustments**: Build flexibility into the zone system to accommodate seasonal variations, special events, 
                   and changing demand patterns.
                """)
    
    # If data is loaded but analysis not yet run
    else:
        st.info("Data loaded successfully. Click 'Run Zone Analysis' to create delivery zones.")

else:
    # Only show this if no data is loaded yet (except for demo data)
    if data_source != "Demo Data":
        st.info("Please upload a CSV file or fetch data from Blowhorn Analytics to get started.")
        
        # Show example of expected data format
        st.markdown('<div class="section-header">Expected Data Format</div>', unsafe_allow_html=True)
        st.markdown("""
        Your CSV file should contain the following columns:
        - `delivered_lat`, `delivered_long`: Coordinates of delivery points
        - `hub`, `hub_lat`, `hub_long`: Hub name and coordinates
        - `postcode`: Delivery postcode/pincode
        
        Additional useful columns:
        - `weight`: Package weight
        - `kms`: Delivery distance
        - `customer`: Customer name
        """)
