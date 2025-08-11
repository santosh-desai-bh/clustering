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
from itertools import permutations

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
    
    def _get_direction_from_warehouse(self, warehouse_coords, order_lat, order_lng):
        """Classify order direction from warehouse as NE, NW, SE, SW, or CENTRAL"""
        lat_diff = order_lat - warehouse_coords['lat']
        lng_diff = order_lng - warehouse_coords['lng']
        
        # Calculate distance to determine if it's central
        distance = self._haversine_distance(warehouse_coords['lat'], warehouse_coords['lng'], order_lat, order_lng)
        
        # If very close to warehouse (< 2km), consider it central
        if distance < 2.0:
            return 'CENTRAL'
        
        # Determine direction based on lat/lng differences
        if lat_diff >= 0 and lng_diff >= 0:
            return 'NE'  # North-East
        elif lat_diff >= 0 and lng_diff < 0:
            return 'NW'  # North-West  
        elif lat_diff < 0 and lng_diff >= 0:
            return 'SE'  # South-East
        else:
            return 'SW'  # South-West
    
    def _group_orders_by_direction(self, cluster_data, warehouse_coords):
        """Group orders by geographic direction from warehouse"""
        direction_groups = {'NE': [], 'NW': [], 'SE': [], 'SW': [], 'CENTRAL': []}
        
        for idx, order in cluster_data.iterrows():
            direction = self._get_direction_from_warehouse(
                warehouse_coords, 
                order['delivery_lat'], 
                order['delivery_lng']
            )
            direction_groups[direction].append(order)
        
        # Convert lists to DataFrames
        for direction in direction_groups:
            if direction_groups[direction]:
                direction_groups[direction] = pd.DataFrame(direction_groups[direction])
            else:
                direction_groups[direction] = pd.DataFrame()
        
        return direction_groups
    
    def optimize_cluster_routes(self, max_distance_per_route=40, max_time_hours=4):
        """Optimize delivery routes within each cluster with realistic driver constraints
        
        Args:
            max_distance_per_route: Maximum distance per route in km (default: 40km)
            max_time_hours: Maximum time per route in hours (default: 4 hours)
        
        Driver batching logic:
        - Each driver gets one route starting and ending at warehouse
        - Route optimized for minimum distance using nearest neighbor
        - Constraints: 40km max distance, 4 hours max time
        - Time = travel_time + delivery_time (5 min per delivery)
        """
        if self.clustered_df is None:
            return None
            
        cluster_routes = {}
        delivery_time_per_order = 5  # 5 minutes per delivery
        avg_speed_kmh = 25  # Average speed in city traffic
        
        for cluster_id in self.clustered_df['cluster_id'].unique():
            cluster_data = self.clustered_df[self.clustered_df['cluster_id'] == cluster_id].copy()
            warehouse = cluster_data['assigned_warehouse'].iloc[0]
            warehouse_coords = self.warehouses[warehouse]
            
            # Create driver routes with realistic constraints
            routes = self._create_driver_batches(
                cluster_data, 
                warehouse_coords, 
                max_distance_per_route, 
                max_time_hours, 
                delivery_time_per_order,
                avg_speed_kmh
            )
            
            cluster_routes[cluster_id] = {
                'warehouse': warehouse,
                'warehouse_coords': warehouse_coords,
                'routes': [],
                'drivers_needed': len(routes)
            }
            
            for i, route_orders in enumerate(routes):
                optimized_route = self._optimize_single_route(route_orders, warehouse_coords)
                route_distance = self._calculate_route_distance(optimized_route, warehouse_coords)
                route_time = (route_distance / avg_speed_kmh) + (len(optimized_route) * delivery_time_per_order / 60)
                
                cluster_routes[cluster_id]['routes'].append({
                    'route_id': f"R{cluster_id}-{i+1}",
                    'driver_id': f"Driver-{cluster_id}-{i+1}",
                    'orders': optimized_route,
                    'total_distance': route_distance,
                    'total_time_hours': route_time,
                    'order_count': len(optimized_route),
                    'efficiency_score': len(optimized_route) / route_distance if route_distance > 0 else 0
                })
        
        return cluster_routes
    
    def _create_driver_batches(self, cluster_data, warehouse_coords, max_distance, max_time_hours, delivery_time_per_order, avg_speed):
        """Create driver batches with realistic constraints, balanced loads, and directional routing"""
        total_orders = len(cluster_data)
        if total_orders == 0:
            return []
        
        # Calculate cluster density and spread to determine routing strategy
        cluster_center_lat = cluster_data['delivery_lat'].mean()
        cluster_center_lng = cluster_data['delivery_lng'].mean()
        
        # Calculate max distance from center of cluster
        max_distance_from_center = 0
        for idx, order in cluster_data.iterrows():
            dist = self._haversine_distance(cluster_center_lat, cluster_center_lng, 
                                          order['delivery_lat'], order['delivery_lng'])
            max_distance_from_center = max(max_distance_from_center, dist)
        
        # Determine if cluster is small and dense (< 5km radius) or large and spread
        is_small_dense_cluster = max_distance_from_center < 5.0
        
        # Calculate optimal number of drivers for balanced loads
        min_orders_per_driver = 22
        max_orders_per_driver = 32  # 30 + 2 buffer
        optimal_drivers = max(1, round(total_orders / 28))  # Target ~28 orders per driver to minimize drivers
        
        # Adjust if loads would be too unbalanced or insufficient
        avg_orders_per_driver = total_orders / optimal_drivers
        if avg_orders_per_driver < min_orders_per_driver:
            optimal_drivers = max(1, total_orders // min_orders_per_driver)
        elif avg_orders_per_driver > max_orders_per_driver:
            optimal_drivers = max(1, (total_orders + max_orders_per_driver - 1) // max_orders_per_driver)  # Ceiling division
        
        # Calculate target orders per driver for balanced distribution
        base_orders_per_driver = total_orders // optimal_drivers
        extra_orders = total_orders % optimal_drivers
        
        # Create target sizes for each driver
        target_sizes = []
        for i in range(optimal_drivers):
            target_size = base_orders_per_driver + (1 if i < extra_orders else 0)
            target_sizes.append(target_size)
        
        routes = []
        
        if is_small_dense_cluster:
            # For small dense clusters: use simple balanced batching (all directions mixed)
            routes = self._create_balanced_routes_simple(
                cluster_data, warehouse_coords, target_sizes, 
                max_distance, max_time_hours, delivery_time_per_order, avg_speed, max_orders_per_driver
            )
        else:
            # For large spread clusters: use directional routing for minimum travel
            routes = self._create_directional_routes(
                cluster_data, warehouse_coords, target_sizes,
                max_distance, max_time_hours, delivery_time_per_order, avg_speed, max_orders_per_driver
            )
        
        return routes
    
    def _create_balanced_routes_simple(self, cluster_data, warehouse_coords, target_sizes, 
                                     max_distance, max_time_hours, delivery_time_per_order, avg_speed, max_orders_per_driver):
        """Create balanced routes for small dense clusters - optimize for maximum deliveries per route"""
        routes = []
        remaining_orders = cluster_data.copy()
        
        for driver_idx, target_size in enumerate(target_sizes):
            if len(remaining_orders) == 0:
                break
                
            # Start new route from warehouse - optimize for maximum deliveries with minimum travel
            current_route = []
            
            # Find the most central starting point to minimize overall travel
            if len(remaining_orders) > 0:
                center_lat = remaining_orders['delivery_lat'].mean()
                center_lng = remaining_orders['delivery_lng'].mean()
                
                # Find order closest to center as starting point
                center_distances = remaining_orders.apply(
                    lambda row: self._haversine_distance(center_lat, center_lng, row['delivery_lat'], row['delivery_lng']), axis=1
                )
                start_idx = center_distances.idxmin()
                start_order = remaining_orders.loc[start_idx]
                current_route.append(start_order)
                current_lat, current_lng = start_order['delivery_lat'], start_order['delivery_lng']
                remaining_orders = remaining_orders.drop(start_idx)
            
            # Build route using nearest neighbor but prioritize maximizing deliveries
            while len(current_route) < target_size and len(remaining_orders) > 0:
                # Find nearest unvisited order
                distances = remaining_orders.apply(
                    lambda row: self._haversine_distance(current_lat, current_lng, row['delivery_lat'], row['delivery_lng']), axis=1
                )
                
                # Try multiple nearest orders to find best fit for constraints
                sorted_distances = distances.sort_values()
                added_order = False
                
                for candidate_idx in sorted_distances.head(min(3, len(sorted_distances))).index:
                    candidate_order = remaining_orders.loc[candidate_idx]
                    potential_route = current_route + [candidate_order]
                    
                    potential_distance = self._calculate_route_distance(
                        [order.to_dict() if hasattr(order, 'to_dict') else order for order in potential_route], 
                        warehouse_coords
                    )
                    potential_time = (potential_distance / avg_speed) + (len(potential_route) * delivery_time_per_order / 60)
                    
                    # Check if we can add this order
                    if (potential_distance <= max_distance and 
                        potential_time <= max_time_hours and 
                        len(potential_route) <= max_orders_per_driver):
                        
                        current_route.append(candidate_order)
                        current_lat, current_lng = candidate_order['delivery_lat'], candidate_order['delivery_lng']
                        remaining_orders = remaining_orders.drop(candidate_idx)
                        added_order = True
                        break
                
                if not added_order:
                    break  # No more orders can be added to this route
            
            # Add completed route
            if current_route:
                routes.append(pd.DataFrame(current_route))
        
        # Distribute any remaining orders
        self._distribute_remaining_orders(routes, remaining_orders, max_orders_per_driver)
        return routes
    
    def _create_directional_routes(self, cluster_data, warehouse_coords, target_sizes,
                                 max_distance, max_time_hours, delivery_time_per_order, avg_speed, max_orders_per_driver):
        """Create directional routes for large spread clusters - minimize travel by grouping by direction"""
        # Group orders by direction
        direction_groups = self._group_orders_by_direction(cluster_data, warehouse_coords)
        
        routes = []
        all_remaining = cluster_data.copy()
        
        # Priority order for directions (can be customized)
        direction_priority = ['CENTRAL', 'NE', 'NW', 'SE', 'SW']
        
        for target_size in target_sizes:
            if len(all_remaining) == 0:
                break
                
            current_route = []
            
            # Try to fill route primarily from one direction for efficient travel
            for direction in direction_priority:
                if len(current_route) >= target_size:
                    break
                    
                direction_orders = all_remaining[all_remaining.apply(
                    lambda row: self._get_direction_from_warehouse(warehouse_coords, row['delivery_lat'], row['delivery_lng']) == direction, axis=1
                )]
                
                if len(direction_orders) == 0:
                    continue
                
                # Add orders from this direction efficiently
                remaining_in_direction = direction_orders.copy()
                current_lat, current_lng = warehouse_coords['lat'], warehouse_coords['lng']
                
                while len(current_route) < target_size and len(remaining_in_direction) > 0:
                    # Find nearest order in this direction
                    distances = remaining_in_direction.apply(
                        lambda row: self._haversine_distance(current_lat, current_lng, row['delivery_lat'], row['delivery_lng']), axis=1
                    )
                    
                    nearest_idx = distances.idxmin()
                    nearest_order = remaining_in_direction.loc[nearest_idx]
                    
                    # Check constraints
                    potential_route = current_route + [nearest_order]
                    potential_distance = self._calculate_route_distance(
                        [order.to_dict() if hasattr(order, 'to_dict') else order for order in potential_route], 
                        warehouse_coords
                    )
                    potential_time = (potential_distance / avg_speed) + (len(potential_route) * delivery_time_per_order / 60)
                    
                    if (potential_distance <= max_distance and 
                        potential_time <= max_time_hours and 
                        len(potential_route) <= max_orders_per_driver):
                        
                        current_route.append(nearest_order)
                        current_lat, current_lng = nearest_order['delivery_lat'], nearest_order['delivery_lng']
                        all_remaining = all_remaining.drop(nearest_idx)
                        remaining_in_direction = remaining_in_direction.drop(nearest_idx)
                    else:
                        break  # Can't add more from this direction
            
            # Add completed route
            if current_route:
                routes.append(pd.DataFrame(current_route))
        
        # Distribute any remaining orders
        self._distribute_remaining_orders(routes, all_remaining, max_orders_per_driver)
        return routes
    
    def _distribute_remaining_orders(self, routes, remaining_orders, max_orders_per_driver):
        """Distribute remaining orders to existing routes that have capacity"""
        while len(remaining_orders) > 0:
            order = remaining_orders.iloc[0]
            # Find route with minimum orders that can still fit one more
            best_route_idx = -1
            min_route_size = float('inf')
            
            for i, route in enumerate(routes):
                if len(route) < max_orders_per_driver and len(route) < min_route_size:
                    min_route_size = len(route)
                    best_route_idx = i
            
            if best_route_idx >= 0:
                # Add to the smallest route
                routes[best_route_idx] = pd.concat([routes[best_route_idx], order.to_frame().T], ignore_index=True)
            else:
                # Create new route if all are full
                routes.append(order.to_frame().T)
            
            remaining_orders = remaining_orders.drop(remaining_orders.index[0])
    
    def _optimize_single_route(self, orders, warehouse_coords):
        """Optimize order sequence within a single route using nearest neighbor"""
        if len(orders) <= 1:
            return orders.to_dict('records')
        
        # Start from warehouse
        unvisited = orders.copy()
        route_sequence = []
        current_lat, current_lng = warehouse_coords['lat'], warehouse_coords['lng']
        
        # Nearest neighbor algorithm
        while len(unvisited) > 0:
            distances = unvisited.apply(
                lambda row: self._haversine_distance(
                    current_lat, current_lng,
                    row['delivery_lat'], row['delivery_lng']
                ), axis=1
            )
            
            nearest_idx = distances.idxmin()
            nearest_order = unvisited.loc[nearest_idx]
            
            route_sequence.append(nearest_order.to_dict())
            current_lat, current_lng = nearest_order['delivery_lat'], nearest_order['delivery_lng']
            unvisited = unvisited.drop(nearest_idx)
        
        return route_sequence
    
    def _calculate_route_distance(self, route_orders, warehouse_coords):
        """Calculate total distance for a route including return to warehouse"""
        if not route_orders:
            return 0
        
        total_distance = 0
        current_lat, current_lng = warehouse_coords['lat'], warehouse_coords['lng']
        
        # Distance from warehouse to first order
        first_order = route_orders[0]
        total_distance += self._haversine_distance(
            current_lat, current_lng,
            first_order['delivery_lat'], first_order['delivery_lng']
        )
        
        # Distance between consecutive orders
        for i in range(1, len(route_orders)):
            prev_order = route_orders[i-1]
            curr_order = route_orders[i]
            total_distance += self._haversine_distance(
                prev_order['delivery_lat'], prev_order['delivery_lng'],
                curr_order['delivery_lat'], curr_order['delivery_lng']
            )
        
        # Distance from last order back to warehouse
        last_order = route_orders[-1]
        total_distance += self._haversine_distance(
            last_order['delivery_lat'], last_order['delivery_lng'],
            current_lat, current_lng
        )
        
        return total_distance
    
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
            
            # Skip route generation in map view for performance
            cluster_routes = None
            
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
            
            # Add optimized routes as connected lines
            if cluster_routes:
                route_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3', '#54A0FF']
                route_color_idx = 0
                
                for cluster_id, cluster_info in cluster_routes.items():
                    warehouse_coords = cluster_info['warehouse_coords']
                    
                    for route in cluster_info['routes']:
                        route_orders = route['orders']
                        route_color = route_colors[route_color_idx % len(route_colors)]
                        route_color_idx += 1
                        
                        if len(route_orders) > 0:
                            # Create route path: warehouse -> orders -> warehouse
                            route_lats = [warehouse_coords['lat']]
                            route_lngs = [warehouse_coords['lng']]
                            
                            for order in route_orders:
                                route_lats.append(order['delivery_lat'])
                                route_lngs.append(order['delivery_lng'])
                            
                            route_lats.append(warehouse_coords['lat'])  # Return to warehouse
                            route_lngs.append(warehouse_coords['lng'])
                            
                            # Add route line
                            fig.add_trace(go.Scattermapbox(
                                lat=route_lats,
                                lon=route_lngs,
                                mode='lines',
                                line=dict(
                                    color=route_color,
                                    width=3,
                                    dash='solid'
                                ),
                                name=f'Route {route["route_id"]}',
                                hovertext=f'🚚 Route {route["route_id"]}<br>📦 {route["order_count"]} orders<br>📍 {route["total_distance"]:.1f}km total<br>⏱️ Optimized sequence',
                                hovertemplate='%{hovertext}<extra></extra>',
                                showlegend=False,
                                opacity=0.8
                            ), row=1, col=2)
                            
                            # Add route sequence numbers
                            for i, order in enumerate(route_orders, 1):
                                fig.add_trace(go.Scattermapbox(
                                    lat=[order['delivery_lat']],
                                    lon=[order['delivery_lng']],
                                    mode='text',
                                    text=[f'<b>{i}</b>'],
                                    textfont=dict(
                                        color='white',
                                        size=10,
                                        family='Arial Black'
                                    ),
                                    textposition='middle center',
                                    name=f'Sequence {i}',
                                    hoverinfo='skip',
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
    
    def _create_warehouse_route_map(self, warehouse, warehouse_routes):
        """Create a map visualization showing routes for a specific warehouse"""
        import plotly.graph_objects as go
        
        if not warehouse_routes:
            return None
        
        fig = go.Figure()
        
        # Get warehouse coordinates
        warehouse_coords = self.warehouses[warehouse]
        warehouse_color = self.warehouse_colors.get(warehouse, '#FF0000')
        
        # 15 highly contrasting colors for better visibility
        route_colors = [
            '#FF0000',  # Red
            '#00FF00',  # Lime
            '#0000FF',  # Blue
            '#FFFF00',  # Yellow
            '#FF00FF',  # Magenta
            '#00FFFF',  # Cyan
            '#FFA500',  # Orange
            '#800080',  # Purple
            '#008000',  # Green
            '#FFC0CB',  # Pink
            '#A52A2A',  # Brown
            '#808080',  # Gray
            '#000080',  # Navy
            '#008080',  # Teal
            '#DC143C'   # Crimson
        ]
        route_idx = 1  # Start from Route 1
        
        # Add warehouse marker
        fig.add_trace(go.Scattermapbox(
            lat=[warehouse_coords['lat']],
            lon=[warehouse_coords['lng']],
            mode='markers+text',
            marker=dict(size=20, color=warehouse_color, opacity=1.0),
            text=['⌂'],
            textfont=dict(color='white', size=16, family='Arial Black'),
            textposition='middle center',
            name=f'{warehouse} Warehouse',
            hovertext=f'⌂ {warehouse} Warehouse',
            hovertemplate='%{hovertext}<extra></extra>'
        ))
        
        # Add routes for each cluster with consistent color mapping
        # First, sort clusters and assign route numbers consistently
        sorted_clusters = sorted(warehouse_routes.items())
        
        for cluster_id, cluster_info in sorted_clusters:
            # Sort routes by their geographic center to minimize overlaps
            routes_with_centers = []
            for i, route in enumerate(cluster_info['routes']):
                route_orders = route['orders']
                if len(route_orders) > 0:
                    # Calculate route center
                    center_lat = sum(order['delivery_lat'] for order in route_orders) / len(route_orders)
                    center_lng = sum(order['delivery_lng'] for order in route_orders) / len(route_orders)
                    routes_with_centers.append((route, center_lat, center_lng, i))
            
            # Sort by geographic position to reduce visual overlap
            routes_with_centers.sort(key=lambda x: (x[1], x[2]))  # Sort by lat, then lng
            
            for route, center_lat, center_lng, original_idx in routes_with_centers:
                route_orders = route['orders']
                route_color = route_colors[(route_idx - 1) % len(route_colors)]
                
                if len(route_orders) > 0:
                    # Create optimized route path: warehouse -> orders -> warehouse
                    route_lats = [warehouse_coords['lat']]
                    route_lngs = [warehouse_coords['lng']]
                    
                    for order in route_orders:
                        route_lats.append(order['delivery_lat'])
                        route_lngs.append(order['delivery_lng'])
                    
                    route_lats.append(warehouse_coords['lat'])  # Return to warehouse
                    route_lngs.append(warehouse_coords['lng'])
                    
                    # Add route line with cluster context in name
                    fig.add_trace(go.Scattermapbox(
                        lat=route_lats,
                        lon=route_lngs,
                        mode='lines',
                        line=dict(color=route_color, width=4),  # Slightly thicker for better visibility
                        name=f'Route {route_idx} (C{cluster_id})',
                        hovertext=f'🚚 Route {route_idx}<br>📍 Cluster {cluster_id}<br>📦 {route["order_count"]} orders<br>👤 {route["driver_id"]}',
                        hovertemplate='%{hovertext}<extra></extra>',
                        opacity=0.9
                    ))
                    
                    # Add delivery point markers with consistent colors
                    delivery_lats = [order['delivery_lat'] for order in route_orders]
                    delivery_lngs = [order['delivery_lng'] for order in route_orders]
                    delivery_texts = [f'{j+1}' for j in range(len(route_orders))]
                    
                    fig.add_trace(go.Scattermapbox(
                        lat=delivery_lats,
                        lon=delivery_lngs,
                        mode='markers+text',
                        marker=dict(size=14, color=route_color, opacity=0.9),  # Removed line property
                        text=delivery_texts,
                        textfont=dict(color='white', size=10, family='Arial Black'),
                        textposition='middle center',
                        name=f'R{route_idx} Stops',
                        hovertext=[f'📦 Stop {j+1}<br>🚚 Route {route_idx} (Cluster {cluster_id})<br>👤 {route["driver_id"]}<br>📍 Order: {order.get("order_id", "N/A")}' 
                                  for j, order in enumerate(route_orders)],
                        hovertemplate='%{hovertext}<extra></extra>',
                        showlegend=False
                    ))
                    
                    route_idx += 1  # Increment route counter
        
        # Update map layout
        fig.update_layout(
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=warehouse_coords['lat'], lon=warehouse_coords['lng']),
                zoom=11
            ),
            height=600,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            title=f"Route Visualization for {warehouse}",
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        return fig

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
                        # Clear route optimization cache when new clusters are created
                        if 'route_optimization_cache' in st.session_state:
                            st.session_state.route_optimization_cache = {}
                        st.success(f"✅ Created {n_clusters} clusters")
            
            # Route analysis will be available as separate tab after clustering
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
        
        # Only show content after clusters are created
        if hasattr(st.session_state, 'clusters_created'):
            # Create main tabs
            tab1, tab2 = st.tabs(["🎯 Clusters", "🚚 Routes"])
            
            with tab1:
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
                
                # Driver requirements analysis (only if clusters are created)
                st.divider()
                st.subheader("📊 Driver Requirements Analysis")
                
                # Get driver requirements data first before showing rationale
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
                
                # Show capacity rationale if available
                if hasattr(st.session_state, 'capacity_rationale') and current_summary is not None:
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
                        
                        # Show efficiency metrics for optimized system
                        optimized_efficiency = (clustered_summary['orders_median'] / clustered_summary['drivers_required_median']).mean()
                        st.info(f"📊 Optimized Efficiency: {optimized_efficiency:.1f} orders/driver/day (projected)")
                    
                    # Summary comparison metrics
                    st.divider()
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("**Current Drivers (Median)**", total_current_median)
                    with col2:
                        st.metric("**Optimized Drivers (Median)**", total_optimized_median, delta=int(total_optimized_median - total_current_median))
                    with col3:
                        reduction_pct = ((total_current_median - total_optimized_median) / total_current_median * 100) if total_current_median > 0 else 0
                        st.metric("**Reduction**", f"{reduction_pct:.1f}%")
                    with col4:
                        monthly_savings = (total_current_median - total_optimized_median) * 30 * 900
                        st.metric("**Monthly Savings**", f"₹{monthly_savings:,.0f}")
                    
                    # Peak day comparison
                    st.markdown("### 📈 Peak Day Comparison")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("**Current (Peak)**", total_current_peak)
                    with col2:
                        st.metric("**Optimized (Peak)**", total_optimized_peak, delta=int(total_optimized_peak - total_current_peak))
                    with col3:
                        peak_reduction_pct = ((total_current_peak - total_optimized_peak) / total_current_peak * 100) if total_current_peak > 0 else 0
                        st.metric("**Peak Reduction**", f"{peak_reduction_pct:.1f}%")
                    with col4:
                        peak_savings = (total_current_peak - total_optimized_peak) * 30 * 900
                        st.metric("**Peak Savings/Month**", f"₹{peak_savings:,.0f}")
                
                else:
                    st.info("No driver data available")
        
            
            with tab2:
                st.header("🗺️ Detailed Route Analysis by Hub & Zone")
                st.markdown("**Driver Batching Logic:** Minimize drivers with 22-30 orders per route (4 hours max, 40km max, directional routing: NE/NW/SE/SW/Central)")
                
                if optimizer.clustered_df is not None:
                    # Use SAME peak day data as Map Analysis tab for consistency
                    daily_totals = optimizer.clustered_df.groupby('date').size()
                    peak_day = daily_totals.idxmax()
                    peak_day_clustered = optimizer.clustered_df[optimizer.clustered_df['date'] == peak_day].copy()
                    
                    st.info(f"📊 **Route Analysis for Peak Day**: {peak_day} ({len(peak_day_clustered):,} orders) - Same data as Map Analysis")
                    
                    # Get available warehouses from PEAK DAY clustered data
                    available_warehouses = sorted(peak_day_clustered['assigned_warehouse'].unique())
                    
                    # Initialize session state for route optimization
                    if 'route_optimization_cache' not in st.session_state:
                        st.session_state.route_optimization_cache = {}
                    
                    # Show warehouse selection for lazy loading
                    st.subheader("📍 Select Warehouse to Analyze Routes")
                    
                    # Warehouse selection tabs
                    warehouse_tabs = st.tabs([f"🏪 {warehouse}" for warehouse in available_warehouses])
                    
                    for i, warehouse in enumerate(available_warehouses):
                        with warehouse_tabs[i]:
                            st.subheader(f"Route Analysis for {warehouse} - Peak Day {peak_day}")
                            
                            # Get clusters for this warehouse from PEAK DAY data only
                            warehouse_clusters = peak_day_clustered[
                                peak_day_clustered['assigned_warehouse'] == warehouse
                            ]['cluster_id'].unique()
                            
                            # Show warehouse summary first using PEAK DAY data
                            warehouse_orders = peak_day_clustered[
                                peak_day_clustered['assigned_warehouse'] == warehouse
                            ]
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.info(f"**Total Orders:** {len(warehouse_orders):,}")
                            with col2:
                                st.info(f"**Clusters:** {len(warehouse_clusters)}")
                            
                            # Button to optimize routes for this warehouse only (peak day)
                            # Include cluster count in cache key to invalidate when clusters change
                            n_clusters_current = len(peak_day_clustered['cluster_id'].unique())
                            cache_key = f"routes_{warehouse}_{peak_day}_{n_clusters_current}"
                            
                            if st.button(f"🚀 Optimize Routes for {warehouse}", key=f"optimize_{warehouse}"):
                                with st.spinner(f"Optimizing routes for {warehouse}... This may take 30-60 seconds"):
                                    # Optimize routes for this warehouse's clusters only
                                    warehouse_cluster_routes = {}
                                    
                                    for cluster_id in warehouse_clusters:
                                        cluster_data = peak_day_clustered[peak_day_clustered['cluster_id'] == cluster_id].copy()
                                        warehouse_coords = optimizer.warehouses[warehouse]
                                        
                                        # Create driver routes with realistic constraints
                                        routes = optimizer._create_driver_batches(
                                            cluster_data, 
                                            warehouse_coords, 
                                            40,  # max_distance_per_route
                                            4,   # max_time_hours
                                            5,   # delivery_time_per_order
                                            25   # avg_speed_kmh
                                        )
                                        
                                        warehouse_cluster_routes[cluster_id] = {
                                            'warehouse': warehouse,
                                            'warehouse_coords': warehouse_coords,
                                            'routes': [],
                                            'drivers_needed': len(routes)
                                        }
                                        
                                        for j, route_orders in enumerate(routes):
                                            optimized_route = optimizer._optimize_single_route(route_orders, warehouse_coords)
                                            route_distance = optimizer._calculate_route_distance(optimized_route, warehouse_coords)
                                            route_time = (route_distance / 25) + (len(optimized_route) * 5 / 60)
                                            
                                            warehouse_cluster_routes[cluster_id]['routes'].append({
                                                'route_id': f"R{cluster_id}-{j+1}",
                                                'driver_id': f"Driver-{cluster_id}-{j+1}",
                                                'orders': optimized_route,
                                                'total_distance': route_distance,
                                                'total_time_hours': route_time,
                                                'order_count': len(optimized_route),
                                                'efficiency_score': len(optimized_route) / route_distance if route_distance > 0 else 0
                                            })
                                    
                                    # Cache the results
                                    st.session_state.route_optimization_cache[cache_key] = warehouse_cluster_routes
                                    st.success(f"✅ Routes optimized for {warehouse}")
                            
                            # Display cached results if available
                            if cache_key in st.session_state.route_optimization_cache:
                                warehouse_routes = st.session_state.route_optimization_cache[cache_key]
                                
                                if warehouse_routes:
                                    # Calculate warehouse totals
                                    total_orders = sum(
                                        route['order_count'] 
                                        for cluster_info in warehouse_routes.values() 
                                        for route in cluster_info['routes']
                                    )
                                    total_routes = sum(len(cluster_info['routes']) for cluster_info in warehouse_routes.values())
                                    total_distance = sum(
                                        route['total_distance'] 
                                        for cluster_info in warehouse_routes.values() 
                                        for route in cluster_info['routes']
                                    )
                                    
                                    # Show warehouse summary
                                    st.markdown("### 📊 Route Summary")
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.metric("**Orders**", f"{total_orders:,}")
                                    with col2:
                                        st.metric("**Routes**", f"{total_routes}")
                                    with col3:
                                        total_clusters = len(warehouse_routes)
                                        st.metric("**Clusters**", f"{total_clusters}")
                                    with col4:
                                        avg_orders_per_route = total_orders / total_routes if total_routes > 0 else 0
                                        st.metric("**Avg Orders/Route**", f"{avg_orders_per_route:.1f}")
                                    
                                    # Show route visualization map
                                    st.markdown("### 🗺️ Route Visualization")
                                    route_map = optimizer._create_warehouse_route_map(warehouse, warehouse_routes)
                                    if route_map:
                                        st.plotly_chart(route_map, use_container_width=True)
                                    else:
                                        st.info("Route map will appear here after optimization")
                                    
                                    # Show detailed cluster-to-route mapping
                                    st.markdown("### 🎯 Cluster-to-Route Mapping")
                                    st.markdown(f"**Hub Coverage:** {warehouse} serves {len(warehouse_routes)} clusters with detailed route breakdown:")
                                    
                                    # Create a comprehensive mapping table first
                                    cluster_mapping_data = []
                                    global_route_counter = 1
                                    
                                    # Define 15 highly contrasting colors for route visualization
                                    route_colors = [
                                        '#FF0000',  # Red
                                        '#00FF00',  # Lime
                                        '#0000FF',  # Blue
                                        '#FFFF00',  # Yellow
                                        '#FF00FF',  # Magenta
                                        '#00FFFF',  # Cyan
                                        '#FFA500',  # Orange
                                        '#800080',  # Purple
                                        '#008000',  # Green
                                        '#FFC0CB',  # Pink
                                        '#A52A2A',  # Brown
                                        '#808080',  # Gray
                                        '#000080',  # Navy
                                        '#008080',  # Teal
                                        '#DC143C'   # Crimson
                                    ]
                                    
                                    for cluster_id, cluster_info in sorted(warehouse_routes.items()):
                                        cluster_orders = sum(route['order_count'] for route in cluster_info['routes'])
                                        cluster_routes_count = len(cluster_info['routes'])
                                        
                                        # Add main cluster row
                                        cluster_mapping_data.append({
                                            'Type': '📍 Cluster',
                                            'ID': f'Cluster {cluster_id}',
                                            'Orders': cluster_orders,
                                            'Routes': cluster_routes_count,
                                            'Details': f'{cluster_routes_count} routes, {cluster_orders} orders'
                                        })
                                        
                                        # Add individual routes for this cluster
                                        for j, route in enumerate(cluster_info['routes'], 1):
                                            cluster_mapping_data.append({
                                                'Type': '  └── 🚚 Route',
                                                'ID': f'Route {global_route_counter}',
                                                'Orders': route['order_count'],
                                                'Routes': '',
                                                'Details': f"Driver: {route['driver_id']}"
                                            })
                                            global_route_counter += 1
                                    
                                    # Display the mapping table
                                    if cluster_mapping_data:
                                        mapping_df = pd.DataFrame(cluster_mapping_data)
                                        st.dataframe(mapping_df, hide_index=True, use_container_width=True)
                                    
                                    # Show expandable detailed view for each cluster
                                    st.markdown("### 📊 Detailed Cluster Analysis")
                                    for cluster_id, cluster_info in sorted(warehouse_routes.items()):
                                        cluster_orders = sum(route['order_count'] for route in cluster_info['routes'])
                                        cluster_routes_count = len(cluster_info['routes'])
                                        
                                        with st.expander(f"🎯 Cluster {cluster_id} - {cluster_orders} orders, {cluster_routes_count} routes", expanded=False):
                                            
                                            # Show routes within this specific cluster
                                            routes_data = []
                                            for j, route in enumerate(cluster_info['routes'], 1):
                                                # Find the global route number for this route
                                                route_global_num = "N/A"
                                                temp_counter = 1
                                                for temp_cluster_id, temp_cluster_info in sorted(warehouse_routes.items()):
                                                    if temp_cluster_id == cluster_id:
                                                        route_global_num = temp_counter + j - 1
                                                        break
                                                    temp_counter += len(temp_cluster_info['routes'])
                                                
                                                routes_data.append({
                                                    'Global Route': f"Route {route_global_num}",
                                                    'Local Route': f"Route {j}",
                                                    'Orders': route['order_count'],
                                                    'Driver': route['driver_id'],
                                                    'Color': route_colors[(route_global_num - 1) % len(route_colors)] if isinstance(route_global_num, int) else '#000000'
                                                })
                                            
                                            if routes_data:
                                                routes_df = pd.DataFrame(routes_data)
                                                st.dataframe(routes_df, hide_index=True)
                                                
                                                # Show cluster-specific insights
                                                st.markdown(f"**Cluster {cluster_id} Insights:**")
                                                col1, col2, col3 = st.columns(3)
                                                with col1:
                                                    st.metric("Total Orders", cluster_orders)
                                                with col2:
                                                    st.metric("Routes Needed", cluster_routes_count)
                                                with col3:
                                                    avg_orders_per_route = cluster_orders / cluster_routes_count if cluster_routes_count > 0 else 0
                                                    st.metric("Avg Orders/Route", f"{avg_orders_per_route:.1f}")
                                
                                else:
                                    st.info("No routes generated for this warehouse")
                            else:
                                st.info(f"👆 Click the button above to optimize routes for {warehouse}")
                
                else:
                    st.error("❌ No clustered data available. Please create clusters first in the Map Analysis tab.")
        
        else:
            st.info("👆 **Create clusters using the sidebar to see the comparison visualization.**")
    
    else:
        st.info("👆 **Upload your delivery data above to begin analysis.**")

if __name__ == "__main__":
    main()
