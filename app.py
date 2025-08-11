"""
🚚 Delivery Zone Optimization - Pincode vs Geographic Clustering

Clean, focused implementation showing cost savings through intelligent clustering
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial import ConvexHull
import json
import math
import requests
import geopandas as gpd
from shapely.geometry import Point

# Page configuration
st.set_page_config(
    page_title="Delivery Zone Optimization",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

class DeliveryOptimizer:
    def __init__(self):
        """Initialize with Bengaluru warehouse coordinates"""
        self.warehouses = {
            'Mahadevapura': {'lat': 12.99119358634228, 'lng': 77.70770502883568},
            'Hebbal': {'lat': 13.067425838287791, 'lng': 77.60532804961407},
            'Chandra Layout': {'lat': 12.997711927246344, 'lng': 77.51384747974708},
            'Banashankari': {'lat': 12.89201406419532, 'lng': 77.55634971164321},
            'Kudlu': {'lat': 12.880621849247323, 'lng': 77.65504449205629},
            'Domlur': {'lat': 12.961033527003837, 'lng': 77.6360033595211}
        }
        
        # Warehouse-based color coding for visual consistency
        self.warehouse_colors = {
            'Mahadevapura': '#FF4444', 'Hebbal': '#44FF44', 'Chandra Layout': '#4444FF',
            'Banashankari': '#FF8800', 'Kudlu': '#8800FF', 'Domlur': '#00FFFF'
        }
        
        # Initialize data containers
        self.df = None
        self.clustered_df = None
    
    def load_data_from_upload(self, uploaded_file):
        """Load and clean delivery data from uploaded file"""
        try:
            df = pd.read_csv(uploaded_file)
            st.info(f"📄 Raw CSV loaded: {len(df):,} rows")
            
            # Clean and prepare data
            df['delivery_lat'] = pd.to_numeric(df['delivery_lat'], errors='coerce')
            df['delivery_lng'] = pd.to_numeric(df['delivery_lng'], errors='coerce')
            df['created_date'] = pd.to_datetime(df['created_date'])
            df['date'] = df['created_date'].dt.date
            
            # Debug coordinate data thoroughly
            st.info(f"🔍 Coordinate column info:")
            st.info(f"   - delivery_lat type: {df['delivery_lat'].dtype}")
            st.info(f"   - delivery_lng type: {df['delivery_lng'].dtype}")
            st.info(f"   - Non-null lat: {df['delivery_lat'].notna().sum()}")
            st.info(f"   - Non-null lng: {df['delivery_lng'].notna().sum()}")
            
            # Show actual coordinate ranges
            if df['delivery_lat'].notna().sum() > 0:
                lat_range = f"{df['delivery_lat'].min():.6f} to {df['delivery_lat'].max():.6f}"
                lng_range = f"{df['delivery_lng'].min():.6f} to {df['delivery_lng'].max():.6f}"
                st.info(f"📍 Coordinate ranges - Lat: {lat_range}, Lng: {lng_range}")
            else:
                st.error("❌ All coordinates are null!")
            
            # Show a sample of coordinates
            sample_coords = df[['delivery_lat', 'delivery_lng']].head(10)
            st.info(f"📋 Sample coordinates:")
            st.dataframe(sample_coords)
            
            # REMOVE ALL COORDINATE FILTERING FOR NOW - keep everything with non-null coordinates
            valid_coords = (
                df['delivery_lat'].notna() & 
                df['delivery_lng'].notna()
            )
            
            invalid_count = (~valid_coords).sum()
            st.info(f"❌ Only filtering out null coordinates: {invalid_count} rows")
            
            df = df[valid_coords].copy()
            df['hub_clean'] = df['hub'].str.replace(' [ BH Micro warehouse ]', '', regex=False)
            
            st.success(f"✅ After filtering: {len(df):,} valid orders in Bengaluru")
            
            self.df = df
            return df
            
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None
    
    def create_clusters(self, n_clusters: int = 25):
        """Create geographic clusters using KMeans"""
        if self.df is None:
            st.error("No data loaded")
            return None
        
        # Extract coordinates
        coords = self.df[['delivery_lat', 'delivery_lng']].values
        
        # Normalize coordinates for clustering
        scaler = StandardScaler()
        coords_scaled = scaler.fit_transform(coords)
        
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(coords_scaled)
        
        # Get cluster centers in original coordinates
        cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
        
        # Add cluster info to dataframe
        clustered_df = self.df.copy()
        clustered_df['cluster_id'] = cluster_labels
        
        # Calculate distances to cluster centers
        for i, center in enumerate(cluster_centers):
            mask = clustered_df['cluster_id'] == i
            if mask.sum() > 0:
                cluster_coords = coords[cluster_labels == i]
                distances = np.linalg.norm(cluster_coords - center, axis=1) * 111  # Convert to km
                clustered_df.loc[mask, 'distance_to_center_km'] = distances
                clustered_df.loc[mask, 'cluster_center_lat'] = center[0]
                clustered_df.loc[mask, 'cluster_center_lng'] = center[1]
        
        # Assign clusters to nearest warehouses
        clustered_df = self._assign_warehouses(clustered_df, cluster_centers)
        
        self.clustered_df = clustered_df
        return clustered_df
    
    def _assign_warehouses(self, clustered_df, cluster_centers):
        """Assign each cluster to its nearest warehouse"""
        warehouse_names = list(self.warehouses.keys())
        warehouse_coords = np.array([[w['lat'], w['lng']] for w in self.warehouses.values()])
        
        # For each cluster center, find nearest warehouse
        cluster_warehouse_map = {}
        for i, center in enumerate(cluster_centers):
            distances = np.linalg.norm(warehouse_coords - center, axis=1) * 111
            nearest_idx = np.argmin(distances)
            cluster_warehouse_map[i] = warehouse_names[nearest_idx]
        
        # Apply warehouse assignments
        clustered_df['assigned_warehouse'] = clustered_df['cluster_id'].map(cluster_warehouse_map)
        
        return clustered_df
    
    def calculate_distance_traveled(self):
        """Calculate total distance traveled in current vs optimized systems"""
        if self.df is None:
            return None, None
            
        # Current system: Calculate distance from each order to its assigned hub
        current_distances = {}
        for hub, coords in self.warehouses.items():
            hub_orders = self.df[self.df['hub_clean'] == hub].copy()
            if len(hub_orders) > 0:
                # Haversine distance calculation
                hub_orders['hub_lat'] = coords['lat']
                hub_orders['hub_lng'] = coords['lng']
                hub_orders['distance_km'] = hub_orders.apply(
                    lambda row: self._haversine_distance(
                        row['delivery_lat'], row['delivery_lng'],
                        row['hub_lat'], row['hub_lng']
                    ), axis=1
                )
                
                current_distances[hub] = {
                    'total_distance_km': hub_orders['distance_km'].sum(),
                    'avg_distance_km': hub_orders['distance_km'].mean(),
                    'order_count': len(hub_orders),
                    'max_distance_km': hub_orders['distance_km'].max()
                }
        
        # Optimized system: Use cluster assignments
        optimized_distances = {}
        if self.clustered_df is not None:
            for warehouse in self.warehouses.keys():
                warehouse_orders = self.clustered_df[
                    self.clustered_df['assigned_warehouse'] == warehouse
                ].copy()
                
                if len(warehouse_orders) > 0:
                    optimized_distances[warehouse] = {
                        'total_distance_km': warehouse_orders['distance_to_center_km'].sum(),
                        'avg_distance_km': warehouse_orders['distance_to_center_km'].mean(),
                        'order_count': len(warehouse_orders),
                        'max_distance_km': warehouse_orders['distance_to_center_km'].max()
                    }
        
        return current_distances, optimized_distances
    
    def _haversine_distance(self, lat1, lng1, lat2, lng2):
        """Calculate distance between two points using Haversine formula"""
        R = 6371  # Earth's radius in kilometers
        
        lat1_rad = math.radians(lat1)
        lng1_rad = math.radians(lng1)
        lat2_rad = math.radians(lat2)
        lng2_rad = math.radians(lng2)
        
        dlat = lat2_rad - lat1_rad
        dlng = lng2_rad - lng1_rad
        
        a = (math.sin(dlat/2)**2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlng/2)**2)
        c = 2 * math.asin(math.sqrt(a))
        distance = R * c
        
        return distance
    
    def calculate_driver_requirements(self):
        """Calculate driver requirements with rationalized efficiency improvements"""
        if self.df is None:
            return None, None
        
        # Calculate actual capacity based on distance efficiency
        current_avg_distance = 0
        optimized_avg_distance = 0
        
        if self.clustered_df is not None:
            # Calculate average distances for both systems
            current_distances, optimized_distances = self.calculate_distance_traveled()
            if current_distances and optimized_distances:
                current_total_orders = sum(data['order_count'] for data in current_distances.values())
                current_total_dist = sum(data['total_distance_km'] for data in current_distances.values())
                current_avg_distance = current_total_dist / current_total_orders if current_total_orders > 0 else 0
                
                optimized_total_orders = sum(data['order_count'] for data in optimized_distances.values())
                optimized_total_dist = sum(data['total_distance_km'] for data in optimized_distances.values())
                optimized_avg_distance = optimized_total_dist / optimized_total_orders if optimized_total_orders > 0 else 0
        
        # Rationalized capacity calculation based on distance efficiency
        current_capacity = 25  # Base capacity for scattered deliveries
        if current_avg_distance > 0 and optimized_avg_distance > 0:
            # Distance efficiency ratio: shorter distances = more deliveries per day
            distance_efficiency = current_avg_distance / optimized_avg_distance
            # Cap efficiency improvement at 50% (realistic limit)
            distance_efficiency = min(distance_efficiency, 1.5)
            optimized_capacity = int(current_capacity * distance_efficiency)
            # Ensure reasonable bounds (25-40 deliveries per driver)
            optimized_capacity = max(25, min(40, optimized_capacity))
        else:
            # Fallback: conservative 20% improvement for geographic clustering
            optimized_capacity = 30  # 25 * 1.2 = 30 (20% improvement)
        
        # Current system (pincode-based) - Use ACTUAL drivers used
        daily_current = self.df.groupby(['date', 'hub_clean']).agg({
            'order_id': 'count',
            'driver': 'nunique',  # Actual unique drivers used
            'model_name': lambda x: x.nunique() if 'model_name' in self.df.columns else 0,
            'registration_certificate_number': lambda x: x.nunique() if 'registration_certificate_number' in self.df.columns else 0
        }).reset_index()
        daily_current.columns = ['date', 'hub', 'orders', 'actual_drivers_used', 'unique_vehicles', 'unique_registrations']
        
        # Use ACTUAL drivers used instead of theoretical calculation
        daily_current['drivers_theoretical'] = np.ceil(daily_current['orders'] / current_capacity)
        daily_current['drivers_used'] = daily_current['actual_drivers_used']  # Real drivers used
        
        current_summary = daily_current.groupby('hub').agg({
            'orders': ['median', 'max'],
            'drivers_used': ['median', 'max'],  # Actual drivers used
            'drivers_theoretical': ['median', 'max'],  # Theoretical calculation
            'unique_vehicles': ['median', 'max'],
            'unique_registrations': ['median', 'max']
        }).round(1)
        current_summary.columns = ['_'.join(col) for col in current_summary.columns]
        
        # Add efficiency rationale to session state for display
        if not hasattr(st.session_state, 'capacity_rationale'):
            st.session_state.capacity_rationale = {
                'current_capacity': current_capacity,
                'optimized_capacity': optimized_capacity,
                'current_avg_distance': current_avg_distance,
                'optimized_avg_distance': optimized_avg_distance,
                'efficiency_improvement': (optimized_capacity / current_capacity - 1) * 100
            }
        
        # Optimized system (cluster-based)
        if self.clustered_df is not None:
            daily_clustered = self.clustered_df.groupby(['date', 'assigned_warehouse']).agg({
                'order_id': 'count'
            }).reset_index()
            daily_clustered.columns = ['date', 'warehouse', 'orders']
            daily_clustered['drivers_required'] = np.ceil(daily_clustered['orders'] / optimized_capacity)
            
            clustered_summary = daily_clustered.groupby('warehouse').agg({
                'orders': ['median', 'max'],
                'drivers_required': ['median', 'max']
            }).round(1)
            clustered_summary.columns = ['_'.join(col) for col in clustered_summary.columns]
            
            return current_summary, clustered_summary
        
        return current_summary, None
    
    def load_bengaluru_geojson(self):
        """Load Bengaluru pincode boundaries from uploaded or local GeoJSON"""
        try:
            # First check if GeoJSON was uploaded
            if hasattr(st.session_state, 'geojson_data'):
                return st.session_state.geojson_data
            
            # Fallback to local file
            try:
                with open('bengaluru.geojson', 'r') as f:
                    geojson_data = json.load(f)
                return geojson_data
            except:
                # No GeoJSON available - that's okay
                return None
            
        except Exception as e:
            st.warning(f"Could not load Bengaluru GeoJSON: {e}")
            return None
    
    def create_comparison_maps(self):
        """Create same-day delivery analysis using full dataset with peak patterns"""
        if self.df is None:
            return go.Figure()
        
        # Load Bengaluru boundaries
        bengaluru_geojson = self.load_bengaluru_geojson()
        
        # Same-day delivery analysis: Use full dataset but highlight peak patterns
        st.sidebar.markdown("### 📅 Same-Day Delivery Analysis")
        available_dates = sorted(self.df['date'].unique())
        st.sidebar.write(f"**Data Period:** {len(available_dates)} days")
        st.sidebar.write(f"**Date Range:** {available_dates[0]} to {available_dates[-1]}")
        st.sidebar.write(f"**Total Orders:** {len(self.df):,}")
        
        # Calculate daily statistics for context
        daily_totals = self.df.groupby('date').size()
        peak_day = daily_totals.idxmax()
        peak_volume = daily_totals.max()
        avg_volume = daily_totals.mean()
        
        st.sidebar.write(f"**Peak Day:** {peak_day} ({peak_volume:,} orders)")
        st.sidebar.write(f"**Daily Average:** {avg_volume:.0f} orders")
        
        # Use ONLY peak day data for true same-day analysis
        peak_day_data = self.df[self.df['date'] == peak_day].copy()
        
        # Create subplots - use peak day only
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(f"🔴 Current System - Peak Day {peak_day}: {len(peak_day_data):,} orders", 
                          f"🟢 Optimized System - Same Day: {len(peak_day_data):,} orders"),
            specs=[[{"type": "scattermapbox"}, {"type": "scattermapbox"}]],
            horizontal_spacing=0.02
        )
        
        # Use ONLY peak day data for fair comparison
        st.sidebar.success(f"✅ Analyzing peak day only: {len(peak_day_data):,} orders on {peak_day}")
        
        # Get capacity values from session state for tooltips
        current_capacity = getattr(st.session_state, 'capacity_rationale', {}).get('current_capacity', 25)
        optimized_capacity = getattr(st.session_state, 'capacity_rationale', {}).get('optimized_capacity', 30)
        
        # Convert hub names to warehouse assignments for current system
        hub_to_warehouse = {
            'Mahadevapura': 'Mahadevapura', 'Hebbal': 'Hebbal', 'Chandra Layout': 'Chandra Layout',
            'Banashankari': 'Banashankari', 'Kudlu': 'Kudlu', 'Domlur': 'Domlur'
        }
        
        # Initialize variables for consistent bubble sizing
        max_peak_day_orders = 1
        global_max_orders = 1
        
        # LEFT MAP: Current system using ONLY peak day data
        # Show individual order points first, then optionally add boundaries
        sample_size_left = min(2000, len(peak_day_data))
        sample_data_left = peak_day_data.sample(n=sample_size_left, random_state=42) if len(peak_day_data) > sample_size_left else peak_day_data
        
        # Add individual delivery points colored by current hub assignment
        for warehouse in self.warehouse_colors.keys():
            warehouse_orders = sample_data_left[sample_data_left['hub_clean'].map(hub_to_warehouse) == warehouse]
            if len(warehouse_orders) > 0:
                fig.add_trace(go.Scattermapbox(
                    lat=warehouse_orders['delivery_lat'].tolist(),
                    lon=warehouse_orders['delivery_lng'].tolist(),
                    mode='markers',
                    marker=dict(
                        size=6,  # Consistent size for individual points
                        color=self.warehouse_colors[warehouse],
                        opacity=0.8
                    ),
                    name=f'{warehouse} Current Orders',
                    text=[f"📦 Order → {warehouse}<br>📍 Pincode: {pc}<br>📅 Date: {peak_day}<br>🚚 Driver: {driver}<br>🚕 Model: {model}<br>📋 Reg: {reg}<br>🏃 Current System" 
                           for pc, driver, model, reg in zip(
                               warehouse_orders['postcode'], 
                               warehouse_orders['driver'].fillna('Unknown'),
                               warehouse_orders.get('model_name', pd.Series(['Unknown'] * len(warehouse_orders))).fillna('Unknown'),
                               warehouse_orders.get('registration_certificate_number', pd.Series(['Unknown'] * len(warehouse_orders))).fillna('Unknown')
                           )],
                    hovertemplate='%{text}<extra></extra>',
                    showlegend=False
                ), row=1, col=1)
        
        # Add pincode boundaries as light overlay (optional)
        if bengaluru_geojson:
            st.sidebar.info(f"🗺️ Processing {len(bengaluru_geojson.get('features', []))} GeoJSON features")
            
            # Debug: Show sample pincodes from data vs GeoJSON
            sample_data_pincodes = set(peak_day_data['postcode'].dropna().astype(str))
            geojson_pincodes = set()
            for feature in bengaluru_geojson.get('features', []):
                props = feature.get('properties', {})
                pc = props.get('pincode') or props.get('PINCODE') or props.get('pin') or props.get('postal_code')
                if pc:
                    geojson_pincodes.add(str(pc))
            
            st.sidebar.write(f"📍 Data pincodes (sample): {list(sample_data_pincodes)[:10]}")
            st.sidebar.write(f"🗺️ GeoJSON pincodes: {list(geojson_pincodes)[:10]}")
            st.sidebar.write(f"🔗 Matching pincodes: {len(sample_data_pincodes & geojson_pincodes)}")
            
            # Collect pincode data using ONLY peak day data
            pincode_data = []
            
            # Process pincode data for PEAK DAY ONLY
            for feature in bengaluru_geojson.get('features', []):
                if feature['geometry']['type'] == 'Polygon':
                    properties = feature.get('properties', {})
                    pincode = properties.get('pincode') or properties.get('PINCODE') or properties.get('pin') or properties.get('postal_code') or 'Unknown'
                    
                    if str(pincode).isdigit():
                        # Get ONLY peak day orders for this pincode
                        pincode_peak_orders = peak_day_data[peak_day_data['postcode'] == int(pincode)]
                        if len(pincode_peak_orders) > 0:
                            primary_hub = pincode_peak_orders['hub_clean'].mode().iloc[0] if len(pincode_peak_orders['hub_clean'].mode()) > 0 else 'Unknown'
                            warehouse = hub_to_warehouse.get(primary_hub, primary_hub)
                            
                            # Count orders on peak day only
                            peak_day_order_count = len(pincode_peak_orders)
                            max_peak_day_orders = max(max_peak_day_orders, peak_day_order_count)
                            global_max_orders = max(global_max_orders, peak_day_order_count)
                            
                            pincode_data.append({
                                'feature': feature,
                                'pincode': pincode,
                                'primary_hub': primary_hub,
                                'warehouse': warehouse,
                                'peak_day_orders': peak_day_order_count,
                                'peak_day_data': pincode_peak_orders
                            })
            
            # Add light pincode boundaries as background reference (optional)
            for data in pincode_data[:10]:  # Limit to avoid clutter
                feature = data['feature']
                coords = feature['geometry']['coordinates'][0]
                lats, lons = zip(*[(coord[1], coord[0]) for coord in coords])
                
                pincode = data['pincode']
                warehouse = data['warehouse']
                base_color = self.warehouse_colors.get(warehouse, '#CCCCCC')
                
                # Add subtle pincode boundary
                fig.add_trace(go.Scattermapbox(
                    lat=lats,
                    lon=lons,
                    mode='lines',
                    line=dict(color=base_color, width=1),
                    opacity=0.3,  # Very light
                    name=f'Pincode {pincode}',
                    hoverinfo='skip',
                    showlegend=False
                ), row=1, col=1)
        else:
            # If no GeoJSON, create density visualization using delivery points
            st.sidebar.warning("⚠️ No GeoJSON boundaries loaded - showing point density instead")
            
            # Create density grid visualization for current system
            hub_summary = peak_day_data.groupby('hub_clean').agg({
                'order_id': 'count',
                'delivery_lat': 'mean',
                'delivery_lng': 'mean'
            }).reset_index()
            hub_summary.columns = ['hub', 'order_count', 'center_lat', 'center_lng']
            
            # Add hub territory indicators
            for _, hub_data in hub_summary.iterrows():
                hub = hub_data['hub']
                count = hub_data['order_count']
                
                fig.add_trace(go.Scattermapbox(
                    lat=[hub_data['center_lat']],
                    lon=[hub_data['center_lng']],
                    mode='text',
                    text=[f'{count}'],
                    textfont=dict(
                        color=hub_colors.get(hub, '#000000'),
                        size=16,
                        family='Arial Black'
                    ),
                    name=f'Hub Count {hub}',
                    hovertext=f'{hub}: {count} orders in sample',
                    hovertemplate='%{hovertext}<extra></extra>',
                    showlegend=False
                ), row=1, col=1)
        
        # Skip individual delivery points on left map - focus on pincode boundaries
        
        # RIGHT MAP: Optimized clustering using ONLY peak day data
        if self.clustered_df is not None:
            # Filter clustered data to SAME peak day only
            clustered_peak_data = self.clustered_df[self.clustered_df['date'] == peak_day].copy()
            
            if len(clustered_peak_data) == 0:
                st.warning(f"No clustered data available for peak day {peak_day}")
                return fig, peak_day, len(peak_day_data)
            
            # Group by cluster and warehouse using PEAK DAY data only
            cluster_centers = clustered_peak_data.groupby(['cluster_id', 'assigned_warehouse']).agg({
                'cluster_center_lat': 'first',
                'cluster_center_lng': 'first',
                'order_id': 'count',  # Orders on peak day only
                'distance_to_center_km': 'mean'
            }).reset_index()
            
            # Rename for clarity
            cluster_centers['peak_day_orders'] = cluster_centers['order_id']
            max_peak_cluster_orders = cluster_centers['peak_day_orders'].max() if len(cluster_centers) > 0 else 1
            
            # Update global maximum to include cluster data
            global_max_orders = max(global_max_orders, max_peak_cluster_orders)
            
            # Add delivery points from peak day clustered data colored by assigned warehouse
            sample_size_right = min(2000, len(clustered_peak_data))
            clustered_sample = clustered_peak_data.sample(n=sample_size_right, random_state=42) if len(clustered_peak_data) > sample_size_right else clustered_peak_data
            
            for warehouse in self.warehouse_colors.keys():
                warehouse_data = clustered_sample[clustered_sample['assigned_warehouse'] == warehouse]
                if len(warehouse_data) > 0:
                    # Create density effect based on cluster sizes using global scale
                    densities = warehouse_data.groupby('cluster_id').size()
                    warehouse_data_with_density = warehouse_data.merge(
                        densities.to_frame('cluster_density').reset_index(), 
                        on='cluster_id', 
                        how='left'
                    )
                    
                    # Use global maximum for consistent sizing with left map
                    point_sizes = (4 + (warehouse_data_with_density['cluster_density'] / global_max_orders * 6)).tolist()  # Same scale as left
                    
                    fig.add_trace(go.Scattermapbox(
                        lat=warehouse_data['delivery_lat'].tolist(),
                        lon=warehouse_data['delivery_lng'].tolist(),
                        mode='markers',
                        marker=dict(
                            size=point_sizes,
                            color=self.warehouse_colors[warehouse],
                            opacity=0.6,
                            sizemin=2
                        ),
                        name=f'{warehouse} Cluster Orders',
                        text=[f"🏆 Cluster {cid} → {warehouse}<br>📅 Date: {date}<br>📍 Distance: {dist:.1f}km<br>🚚 Driver: {driver}<br>🚕 Model: {model}<br>📋 Reg: {reg}<br>🚀 Optimized System"
                              for cid, date, dist, driver, model, reg in zip(
                                  warehouse_data['cluster_id'],
                                  warehouse_data['date'],
                                  warehouse_data['distance_to_center_km'],
                                  warehouse_data['driver'].fillna('Unknown'),
                                  warehouse_data.get('model_name', pd.Series(['Unknown'] * len(warehouse_data))).fillna('Unknown'),
                                  warehouse_data.get('registration_certificate_number', pd.Series(['Unknown'] * len(warehouse_data))).fillna('Unknown')
                              )],
                        hovertemplate='%{text}<extra></extra>',
                        showlegend=False
                    ), row=1, col=2)
            
            # Add warehouse connection lines with peak day order thickness
            for _, center in cluster_centers.iterrows():
                warehouse = center['assigned_warehouse']
                if warehouse in self.warehouses:
                    warehouse_coords = self.warehouses[warehouse]
                    warehouse_color = self.warehouse_colors.get(warehouse, '#000000')
                    
                    # Line thickness based on peak day orders using global maximum
                    density_ratio = center['peak_day_orders'] / global_max_orders if global_max_orders > 0 else 0
                    line_width = 2 + (density_ratio * 4)  # 2-6px range
                    
                    # Add warehouse-colored connection line
                    fig.add_trace(go.Scattermapbox(
                        lat=[center['cluster_center_lat'], warehouse_coords['lat']],
                        lon=[center['cluster_center_lng'], warehouse_coords['lng']],
                        mode='lines',
                        line=dict(color=warehouse_color, width=line_width),
                        name=f'{warehouse} Connection',
                        text=f'🎆 Connection: Cluster {center["cluster_id"]} → {warehouse}<br>📦 Peak Day ({peak_day}): {int(center["peak_day_orders"])} orders<br>📍 Avg Distance: {center["distance_to_center_km"]:.1f}km<br>🚚 Drivers: {int(np.ceil(center["peak_day_orders"] / optimized_capacity))}',
                        hovertemplate='%{text}<extra></extra>',
                        showlegend=False
                    ), row=1, col=2)
            
            # Add cluster centers with warehouse colors showing peak day orders
            for _, center in cluster_centers.iterrows():
                warehouse = center['assigned_warehouse']
                warehouse_color = self.warehouse_colors.get(warehouse, '#000000')
                
                # Size and appearance based on peak day orders using global maximum
                density_ratio = center['peak_day_orders'] / global_max_orders if global_max_orders > 0 else 0
                marker_size = 16 + (density_ratio * 24)  # 16-40px range - SAME as left map
                font_size = 12 + (density_ratio * 8)  # 12-20px range - consistent with left
                
                # Add cluster center with warehouse color
                fig.add_trace(go.Scattermapbox(
                    lat=[center['cluster_center_lat']],
                    lon=[center['cluster_center_lng']],
                    mode='markers+text',
                    marker=dict(
                        size=int(marker_size),  # Ensure it's an integer
                        color=warehouse_color,
                        opacity=0.9
                    ),
                    text=[f'<b>{int(center["peak_day_orders"])}</b>'],  # Show peak day orders
                    textfont=dict(color='white', size=int(font_size), family='Arial Black'),
                    textposition='middle center',
                    name=f'Cluster {center["cluster_id"]}',
                    hovertext=f'🏆 Cluster {center["cluster_id"]} → {warehouse}<br>📦 Peak Day ({peak_day}): {int(center["peak_day_orders"])} orders<br>📍 Avg Distance: {center["distance_to_center_km"]:.1f}km<br>🚚 Drivers: {int(np.ceil(center["peak_day_orders"] / optimized_capacity))}<br>🚀 Optimized System',
                    hovertemplate='%{hovertext}<extra></extra>',
                    showlegend=False
                ), row=1, col=2)
        
        # Add enhanced warehouse markers to both maps with different prominence
        for col in [1, 2]:
            for warehouse_name, coords in self.warehouses.items():
                warehouse_color = self.warehouse_colors.get(warehouse_name, '#FF0000')
                
                # Make warehouses more prominent on left map for better visibility
                marker_size = 24 if col == 1 else 18  # Larger on left map
                text_size = 16 if col == 1 else 14    # Larger text on left map
                border_width = 3 if col == 1 else 2   # Thicker border on left map
                
                fig.add_trace(go.Scattermapbox(
                    lat=[coords['lat']],
                    lon=[coords['lng']],
                    mode='markers+text',
                    marker=dict(
                        size=marker_size,
                        color=warehouse_color,
                        opacity=1.0
                    ),
                    text=['⌂'],  # House-like icon for warehouses
                    textfont=dict(color='white', size=text_size, family='Arial Black'),
                    textposition='middle center',
                    name=f'Warehouse {warehouse_name}',
                    hovertext=f'⌂ Warehouse: {warehouse_name}<br>📍 Location: {coords["lat"]:.4f}, {coords["lng"]:.4f}<br>🏢 Micro Warehouse Hub',
                    hovertemplate='%{hovertext}<extra></extra>',
                    showlegend=False
                ), row=1, col=col)
        
        # Update layout
        fig.update_layout(
            mapbox1=dict(
                style="open-street-map",
                center=dict(lat=12.97, lon=77.59),
                zoom=10.2
            ),
            mapbox2=dict(
                style="open-street-map",
                center=dict(lat=12.97, lon=77.59),
                zoom=10.2
            ),
            height=700,
            showlegend=False,
            title_text="Enhanced Delivery Zone Analysis"
        )
        
        return fig, peak_day, len(peak_day_data)
    
    def export_cluster_geojson(self):
        """Export cluster boundaries as GeoJSON"""
        if self.clustered_df is None:
            return None
        
        features = []
        
        for cluster_id in self.clustered_df['cluster_id'].unique():
            cluster_data = self.clustered_df[self.clustered_df['cluster_id'] == cluster_id]
            
            if len(cluster_data) >= 3:  # Need at least 3 points for convex hull
                points = cluster_data[['delivery_lng', 'delivery_lat']].values
                try:
                    hull = ConvexHull(points)
                    hull_points = points[hull.vertices]
                    coordinates = [hull_points.tolist() + [hull_points[0].tolist()]]  # Close polygon
                    
                    feature = {
                        "type": "Feature",
                        "properties": {
                            "cluster_id": int(cluster_id),
                            "warehouse": cluster_data['assigned_warehouse'].iloc[0],
                            "order_count": len(cluster_data),
                            "avg_distance_km": round(cluster_data['distance_to_center_km'].mean(), 2)
                        },
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": coordinates
                        }
                    }
                    features.append(feature)
                except:
                    continue  # Skip clusters that can't form valid polygons
        
        return {
            "type": "FeatureCollection",
            "features": features
        }

def main():
    """Main Streamlit application"""
    st.title("🚚 Delivery Zone Optimization")
    st.markdown("### Pincode Assignment vs Geographic Clustering")
    
    # Initialize optimizer
    if 'optimizer' not in st.session_state:
        st.session_state.optimizer = DeliveryOptimizer()
    
    optimizer = st.session_state.optimizer
    
    # Sidebar configuration - minimal
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Show data summary if loaded
        if hasattr(st.session_state, 'data_loaded') and optimizer.df is not None:
            st.markdown("### 📊 Data Summary")
            st.write(f"**Total records:** {len(optimizer.df):,}")
            st.write(f"**Date range:** {optimizer.df['date'].min()} to {optimizer.df['date'].max()}")
            st.write(f"**Unique hubs:** {optimizer.df['hub_clean'].nunique()}")
            st.write(f"**Unique drivers:** {optimizer.df['driver'].nunique()}")
        
        # Clustering parameters (only show after data is loaded)
        if hasattr(st.session_state, 'data_loaded'):
            st.divider()
            st.markdown("### 🎯 Clustering Settings")
            n_clusters = st.slider("Number of clusters", 15, 40, 25)
            
            if st.button("⚡ Create Clusters", type="primary"):
                with st.spinner("Creating geographic clusters..."):
                    clustered_df = optimizer.create_clusters(n_clusters)
                    if clustered_df is not None:
                        st.session_state.clusters_created = True
                        st.success(f"✅ Created {n_clusters} clusters")
        else:
            st.info("Upload data to see clustering options")
    
    # Data upload section in main content
    if not hasattr(st.session_state, 'data_loaded'):
        st.markdown("### 📁 Upload Your Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Step 1: Upload Delivery Data CSV**")
            uploaded_file = st.file_uploader(
                "Choose CSV file",
                type=['csv'],
                help="Upload the delivery data CSV from Blowhorn Analytics",
                key="csv_uploader"
            )
            
            if uploaded_file is not None:
                st.info(f"📄 **File:** {uploaded_file.name}\n📊 **Size:** {uploaded_file.size:,} bytes")
                
                if st.button("🚀 Load Data", type="primary"):
                    with st.spinner("Loading and processing data..."):
                        df = optimizer.load_data_from_upload(uploaded_file)
                        if df is not None:
                            st.session_state.data_loaded = True
                            st.success(f"✅ Loaded {len(df):,} orders")
                            st.rerun()
        
        with col2:
            st.markdown("**Step 2: Upload Pincode Boundaries (Optional)**")
            geojson_file = st.file_uploader(
                "Choose GeoJSON file (optional)",
                type=['geojson', 'json'],
                help="Upload Bengaluru pincode boundaries GeoJSON for enhanced visualization",
                key="geojson_uploader"
            )
            
            if geojson_file is not None:
                try:
                    # Read and clean the GeoJSON content
                    content = geojson_file.getvalue().decode('utf-8').strip()
                    
                    # Remove any trailing content after the closing brace
                    if content.count('}') > 0:
                        # Find the last closing brace and truncate there
                        last_brace_pos = content.rfind('}')
                        if last_brace_pos != -1:
                            content = content[:last_brace_pos + 1]
                    
                    geojson_data = json.loads(content)
                    st.session_state.geojson_data = geojson_data
                    st.success(f"📍 GeoJSON loaded: {geojson_file.name}")
                    st.info(f"Found {len(geojson_data.get('features', []))} pincode boundaries")
                    
                except json.JSONDecodeError as e:
                    st.error(f"Invalid GeoJSON file: {e}")
                    st.info("💡 **Tip:** Make sure your GeoJSON file contains only valid JSON data")
                except Exception as e:
                    st.error(f"Error loading GeoJSON: {e}")
            else:
                st.info("💡 **Tip:** Upload a GeoJSON file to show pincode boundaries on the map")
        
        st.divider()
        
        # Data source information
        st.markdown("### 📊 How to Get Started")
        st.markdown("""
        **Data Source:** Download the dataset from Blowhorn Analytics
        
        🔗 **[Clustering Tech Team Dataset](https://analytics.blowhorn.com/question/3265-clustering-tech-team)**
        
        **Required CSV Format:**
        - `order_id`: Unique order identifier
        - `driver`: Driver name
        - `hub`: Current hub assignment
        - `postcode`: Delivery postcode
        - `delivery_lat`, `delivery_lng`: Delivery coordinates
        - `created_date`: Order creation timestamp
        
        **Optional GeoJSON Format:**
        - Standard GeoJSON with `Polygon` geometries
        - Properties should include pincode field (`pincode`, `PINCODE`, `pin`, or `postal_code`)
        - Used to show pincode boundaries on the left map
        """)
    
    # Main content area
    elif hasattr(st.session_state, 'data_loaded') and optimizer.df is not None:
        
        # Only show maps after clusters are created
        if hasattr(st.session_state, 'clusters_created'):
            # Display comparison maps
            fig, peak_day, peak_day_count = optimizer.create_comparison_maps()
            st.plotly_chart(fig, use_container_width=True)
            
            # Add warehouse color legend
            st.markdown("### 🎨 Warehouse Color Legend")
            legend_cols = st.columns(len(optimizer.warehouse_colors))
            for i, (warehouse, color) in enumerate(optimizer.warehouse_colors.items()):
                with legend_cols[i]:
                    st.markdown(f'<div style="display:flex;align-items:center;"><div style="width:20px;height:20px;background-color:{color};margin-right:8px;border:2px solid white;"></div><small><b>{warehouse}</b></small></div>', unsafe_allow_html=True)
            
            # Add bubble size legend with enhanced visibility explanation
            st.markdown("### 📏 Order Count Legend (Enhanced Bubble Sizes)")
            col1, col2, col3, col4 = st.columns(4)
            
            # Get order count ranges from the data
            if optimizer.df is not None:
                daily_totals = optimizer.df.groupby('date').size()
                peak_day = daily_totals.idxmax()
                peak_day_data = optimizer.df[optimizer.df['date'] == peak_day]
                
                # Calculate order counts by pincode for legend
                pincode_counts = peak_day_data.groupby('postcode').size()
                if len(pincode_counts) > 0:
                    min_orders = int(pincode_counts.min())
                    max_orders = int(pincode_counts.max())
                    q25_orders = int(pincode_counts.quantile(0.25))
                    q75_orders = int(pincode_counts.quantile(0.75))
                    
                    with col1:
                        st.markdown(f'<div style="display:flex;align-items:center;"><div style="width:16px;height:16px;background-color:#666;border-radius:50%;margin-right:8px;border:2px solid white;"></div><small><b>{min_orders} orders</b></small></div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f'<div style="display:flex;align-items:center;"><div style="width:22px;height:22px;background-color:#666;border-radius:50%;margin-right:8px;border:2px solid white;"></div><small><b>{q25_orders} orders</b></small></div>', unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f'<div style="display:flex;align-items:center;"><div style="width:28px;height:28px;background-color:#666;border-radius:50%;margin-right:8px;border:2px solid white;"></div><small><b>{q75_orders} orders</b></small></div>', unsafe_allow_html=True)
                    
                    with col4:
                        st.markdown(f'<div style="display:flex;align-items:center;"><div style="width:40px;height:40px;background-color:#666;border-radius:50%;margin-right:8px;border:2px solid white;"></div><small><b>{max_orders} orders</b></small></div>', unsafe_allow_html=True)
                
                st.markdown("**💡 Tip**: Left map now shows enhanced visibility with larger bubbles, thicker borders, and density-based point sizes for clearer order distribution comparison.")
            
            # Map explanation with peak day analysis
            st.info(f"📊 **Peak Day Analysis**: Both maps show identical {peak_day_count:,} orders from peak day ({peak_day}) | **Left Map**: Current pincode-hub assignments with enhanced visibility | **Right Map**: Optimized geographic clusters | **Bubbles & Numbers** = order count with density-based sizing | **Colored regions** = delivery zones | **Houses** ⌂ = Warehouses")
        else:
            st.info("👆 **Create clusters using the sidebar to see the comparison visualization.**")
        
        # Driver requirements analysis (only if clusters are created)
        if hasattr(st.session_state, 'clusters_created'):
            st.divider()
            st.subheader("📊 Driver Requirements Analysis")
            
            # Show capacity rationale if available
            if hasattr(st.session_state, 'capacity_rationale'):
                rationale = st.session_state.capacity_rationale
                
                with st.expander("🧮 Driver Capacity Rationale - Click to Expand", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**🔴 Current System (Actual)**")
                        actual_efficiency = (current_summary['orders_median'] / current_summary['drivers_used_median']).mean()
                        st.write(f"• **Actual Efficiency:** {actual_efficiency:.1f} orders/driver/day")
                        st.write(f"• **Theoretical:** {rationale['current_capacity']} orders/driver/day")
                        st.write(f"• **Avg Distance:** {rationale['current_avg_distance']:.2f} km/order")
                        st.write(f"• **Issue:** Scattered pincode deliveries")
                    
                    with col2:
                        st.markdown("**🟢 Optimized System**")
                        st.write(f"• **Capacity:** {rationale['optimized_capacity']} orders/driver/day")
                        st.write(f"• **Avg Distance:** {rationale['optimized_avg_distance']:.2f} km/order")
                        st.write(f"• **Benefit:** Geographic clustering")
                    
                    with col3:
                        st.markdown("**📈 Efficiency Gain**")
                        st.write(f"• **Improvement:** {rationale['efficiency_improvement']:.1f}%")
                        distance_reduction = ((rationale['current_avg_distance'] - rationale['optimized_avg_distance']) / rationale['current_avg_distance'] * 100) if rationale['current_avg_distance'] > 0 else 0
                        st.write(f"• **Distance Saved:** {distance_reduction:.1f}%")
                        st.write(f"• **Logic:** Shorter routes = More deliveries")
                    
                    st.info("💡 **Rationale:** Analysis uses ACTUAL driver counts from the current system vs projected requirements for optimized system. Current system shows real driver utilization while optimized system uses distance-based efficiency calculations capped at 50% improvement for realistic projections.")
            
            current_summary, clustered_summary = optimizer.calculate_driver_requirements()
            
            # Initialize variables to avoid UnboundLocalError
            total_current_median = total_current_peak = 0
            total_optimized_median = total_optimized_peak = 0
            total_theoretical_median = total_theoretical_peak = 0
            
            if current_summary is not None and clustered_summary is not None:
                # Calculate totals early
                total_current_median = current_summary['drivers_used_median'].sum()
                total_current_peak = current_summary['drivers_used_max'].sum()
                total_optimized_median = clustered_summary['drivers_required_median'].sum()
                total_optimized_peak = clustered_summary['drivers_required_max'].sum()
                total_theoretical_median = current_summary['drivers_theoretical_median'].sum()
                total_theoretical_peak = current_summary['drivers_theoretical_max'].sum()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🔴 Current System (Actual Driver Usage)**")
                    current_display = pd.DataFrame({
                        'Hub': current_summary.index,
                        'Median Orders/Day': current_summary['orders_median'].values,
                        'Peak Orders/Day': current_summary['orders_max'].values,
                        'Actual Drivers (Median)': current_summary['drivers_used_median'].values.astype(int),
                        'Actual Drivers (Peak)': current_summary['drivers_used_max'].values.astype(int),
                        'Vehicles (Peak)': current_summary['unique_vehicles_max'].values.astype(int)
                    })
                    st.dataframe(current_display, hide_index=True)
                    
                    # Show efficiency metrics
                    actual_efficiency = (current_summary['orders_median'] / current_summary['drivers_used_median']).mean()
                    st.info(f"📊 Current Efficiency: {actual_efficiency:.1f} orders/driver/day (actual data)")
                
                with col2:
                    st.markdown("**🟢 Optimized System (Projected Requirements)**")
                    clustered_display = pd.DataFrame({
                        'Warehouse': clustered_summary.index,
                        'Median Orders/Day': clustered_summary['orders_median'].values,
                        'Peak Orders/Day': clustered_summary['orders_max'].values,
                        'Required Drivers (Median)': clustered_summary['drivers_required_median'].values.astype(int),
                        'Required Drivers (Peak)': clustered_summary['drivers_required_max'].values.astype(int)
                    })
                    st.dataframe(clustered_display, hide_index=True)
                    
                    # Show projected efficiency
                    projected_efficiency = optimized_capacity if 'optimized_capacity' in locals() else 30
                    st.info(f"📊 Projected Efficiency: {projected_efficiency} orders/driver/day (optimized clusters)")
                
                # Actual vs Theoretical Analysis
                st.divider()
                st.markdown("### 📏 Actual vs Projected Driver Usage")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Current (Actual)",
                        f"{int(total_current_peak)} drivers",
                        f"Peak day usage"
                    )
                
                with col2:
                    st.metric(
                        "Current (Theoretical)", 
                        f"{int(total_theoretical_peak)} drivers",
                        f"{int(total_theoretical_peak - total_current_peak):+d} vs actual"
                    )
                
                with col3:
                    actual_efficiency_pct = (total_current_peak / total_theoretical_peak * 100) if total_theoretical_peak > 0 else 0
                    st.metric(
                        "Current Efficiency",
                        f"{actual_efficiency_pct:.0f}%",
                        f"Driver utilization rate"
                    )
                
                # Main comparison metrics
                st.divider()
                col1, col2, col3, col4 = st.columns(4)
                
                # Variables already calculated above
                
                with col1:
                    st.metric(
                        "Optimized (Median)",
                        f"{int(total_optimized_median)} drivers",
                        f"{int(total_optimized_median - total_current_median):+d} vs actual current"
                    )
                
                with col2:
                    st.metric(
                        "Optimized (Peak)",
                        f"{int(total_optimized_peak)} drivers",
                        f"{int(total_optimized_peak - total_current_peak):+d} vs actual current"
                    )
                
                with col3:
                    median_reduction = (total_current_median - total_optimized_median) / total_current_median * 100 if total_current_median > 0 else 0
                    st.metric(
                        "Median Day Savings",
                        f"{median_reduction:.1f}%",
                        f"{int(total_current_median - total_optimized_median)} driver reduction"
                    )
                
                with col4:
                    peak_reduction = (total_current_peak - total_optimized_peak) / total_current_peak * 100 if total_current_peak > 0 else 0
                    st.metric(
                        "Peak Day Savings",
                        f"{peak_reduction:.1f}%",
                        f"{int(total_current_peak - total_optimized_peak)} driver reduction"
                    )
                
                # Distance analysis section
                st.divider()
                st.subheader("🚗 Distance Traveled Analysis")
                
                current_distances, optimized_distances = optimizer.calculate_distance_traveled()
                
                if current_distances and optimized_distances:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**🔴 Current System - Distance per Hub**")
                        current_dist_df = pd.DataFrame([
                            {
                                'Hub': hub,
                                'Total Distance (km)': f"{data['total_distance_km']:.1f}",
                                'Avg Distance (km)': f"{data['avg_distance_km']:.1f}",
                                'Max Distance (km)': f"{data['max_distance_km']:.1f}",
                                'Orders': data['order_count']
                            }
                            for hub, data in current_distances.items()
                        ])
                        st.dataframe(current_dist_df, hide_index=True)
                    
                    with col2:
                        st.markdown("**🟢 Optimized System - Distance per Warehouse**")
                        optimized_dist_df = pd.DataFrame([
                            {
                                'Warehouse': warehouse,
                                'Total Distance (km)': f"{data['total_distance_km']:.1f}",
                                'Avg Distance (km)': f"{data['avg_distance_km']:.1f}",
                                'Max Distance (km)': f"{data['max_distance_km']:.1f}",
                                'Orders': data['order_count']
                            }
                            for warehouse, data in optimized_distances.items()
                        ])
                        st.dataframe(optimized_dist_df, hide_index=True)
                    
                    # Distance comparison metrics
                    st.divider()
                    col1, col2, col3, col4 = st.columns(4)
                    
                    total_current_dist = sum(data['total_distance_km'] for data in current_distances.values())
                    total_optimized_dist = sum(data['total_distance_km'] for data in optimized_distances.values())
                    avg_current_dist = sum(data['avg_distance_km'] * data['order_count'] for data in current_distances.values()) / sum(data['order_count'] for data in current_distances.values())
                    avg_optimized_dist = sum(data['avg_distance_km'] * data['order_count'] for data in optimized_distances.values()) / sum(data['order_count'] for data in optimized_distances.values())
                    
                    with col1:
                        st.metric(
                            "Total Distance Reduction",
                            f"{total_optimized_dist:.0f} km",
                            f"{total_optimized_dist - total_current_dist:.0f} km vs current"
                        )
                    
                    with col2:
                        distance_reduction_pct = (total_current_dist - total_optimized_dist) / total_current_dist * 100 if total_current_dist > 0 else 0
                        st.metric(
                            "Distance Reduction %",
                            f"{distance_reduction_pct:.1f}%",
                            f"{total_current_dist - total_optimized_dist:.0f} km saved"
                        )
                    
                    with col3:
                        st.metric(
                            "Avg Distance/Order",
                            f"{avg_optimized_dist:.1f} km",
                            f"{avg_optimized_dist - avg_current_dist:.1f} km vs current"
                        )
                    
                    with col4:
                        fuel_savings = (total_current_dist - total_optimized_dist) * 0.1  # Assuming 0.1L/km fuel consumption
                        st.metric(
                            "Fuel Savings (Est.)",
                            f"{fuel_savings:.0f} L",
                            f"₹{fuel_savings * 100:.0f} saved" if fuel_savings > 0 else "₹0"
                        )
                
                # Export section
                st.divider()
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📥 Download Cluster Boundaries")
                    if st.button("Generate GeoJSON"):
                        geojson_data = optimizer.export_cluster_geojson()
                        if geojson_data:
                            geojson_str = json.dumps(geojson_data, indent=2)
                            st.download_button(
                                "📥 Download GeoJSON",
                                geojson_str,
                                "optimized_clusters.geojson",
                                "application/json"
                            )
                            st.success("✅ Ready for download!")
                
                with col2:
                    st.subheader("📈 Key Improvements")
                    cluster_summary = optimizer.clustered_df.groupby(['cluster_id', 'assigned_warehouse']).agg({
                        'order_id': 'count',
                        'distance_to_center_km': 'mean'
                    }).reset_index()
                    
                    st.write(f"• **{len(cluster_summary)}** optimized delivery zones")
                    st.write(f"• **{cluster_summary['distance_to_center_km'].mean():.1f}km** average delivery distance")
                    st.write(f"• **{int(total_current_median - total_optimized_median)}** fewer drivers on median days")
                    st.write(f"• **{cluster_summary['order_id'].sum():,}** total orders optimized")
            
            else:
                st.info("Driver analysis available after creating clusters")
        
        else:
            st.info("👆 **Create clusters using the sidebar to see the comparison and driver analysis.**")
    
    else:
        st.info("👆 **Upload your delivery data above to begin analysis.**")

if __name__ == "__main__":
    main()