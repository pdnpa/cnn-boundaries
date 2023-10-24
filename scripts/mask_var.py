import os
from osdatahub import Extent
import geopandas as gpd
import rasterio

def bb(lyr): # creates a BBox of input layers
    if isinstance (lyr, rasterio.io.DatasetReader):
        t = Extent.from_bbox(lyr.bounds, crs="EPSG:27700")
        return t
    else:
        g = round(lyr.bounds)
        return g # note this only works with pd
    
def urban_mask(gdf): # creates an urban mask of villages from os api
    m = gpd.GeoDataFrame.from_features(gdf, crs="EPSG:27700")
    # subset based on made surface and garden for proper urban
    m = (m.loc[(m['description']== 'Made Surface') | (m['description']== 'Residential Garden')])
    m.unary_union
    m['pdnp'] = 'pdnp'
    m = m.dissolve(by = 'pdnp', as_index=False)
    m = m.buffer(10)
    return m

def wall_subset(gdf): # takes the subset based on built obstructions
    w = gpd.GeoDataFrame.from_features(gdf, crs="EPSG:27700")
    #subset based on 'Built Obstruction'
    w = (w.loc[w['description'] == 'Built Obstruction'])
    return w
