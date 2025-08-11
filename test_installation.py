#!/usr/bin/env python3
"""
Test script to verify all dependencies are installed correctly
for the Delivery Zone Optimization application.
"""

def test_imports():
    """Test all required imports"""
    print("🧪 Testing imports...")
    
    try:
        import streamlit as st
        print("✅ streamlit")
        
        import pandas as pd
        import numpy as np
        print("✅ pandas, numpy")
        
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        print("✅ plotly")
        
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        print("✅ scikit-learn")
        
        from scipy.spatial import ConvexHull
        print("✅ scipy")
        
        import geopandas as gpd
        from shapely.geometry import Point
        print("✅ geopandas, shapely")
        
        import requests
        print("✅ requests")
        
        import json
        import math
        print("✅ json, math (built-in)")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality of key components"""
    print("\n🔧 Testing basic functionality...")
    
    try:
        # Test pandas
        import pandas as pd
        import numpy as np
        df = pd.DataFrame({'a': [1,2,3], 'b': [4,5,6]})
        print("✅ Pandas DataFrame creation")
        
        # Test sklearn
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        data = np.array([[1,2], [3,4], [5,6], [7,8]])
        kmeans.fit(data)
        print("✅ KMeans clustering")
        
        # Test plotly
        import plotly.graph_objects as go
        fig = go.Figure()
        print("✅ Plotly figure creation")
        
        # Test geopandas basic functionality
        import geopandas as gpd
        from shapely.geometry import Point
        point = Point(1, 1)
        print("✅ Geospatial processing")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality error: {e}")
        return False

def main():
    print("🚚 DELIVERY ZONE OPTIMIZATION - INSTALLATION TEST")
    print("=" * 55)
    
    imports_ok = test_imports()
    functionality_ok = test_basic_functionality()
    
    print("\n" + "=" * 55)
    if imports_ok and functionality_ok:
        print("🎯 ALL TESTS PASSED")
        print("✅ Ready to run: streamlit run app.py")
        print("🌐 Access at: http://localhost:8501")
    else:
        print("❌ TESTS FAILED")
        print("💡 Try: pip install -r requirements.txt")
    
    print("=" * 55)

if __name__ == "__main__":
    main()