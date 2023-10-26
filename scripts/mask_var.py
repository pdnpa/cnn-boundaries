import os
from osdatahub import Extent
import geopandas as gpd
import pandas as pd
import rasterio

def bb(lyr): # creates a BBox of input layers
    if isinstance (lyr, rasterio.io.DatasetReader):
        t = Extent.from_bbox(lyr.bounds, crs="EPSG:27700")
        return t
    else:
        g = round(lyr.bounds)
        return g # note this only works with pd
    
def sub(d): # subset the land df with made surfaces or gardens
    db = (d.loc[(d['theme'] == 'Land')]) # subset land
    db = (db.loc[(db['description'] == 'Made Surface') | (db['description'] == 'Residential Garden')]) # subset the subset of land
    d = (d.loc[(d['theme'] != 'Land')])   # drop the land from the origianl df
    d = pd.concat([d, db])
    return d
    
def urban_mask(gdf): # creates an urban mask of villages from os api
    m = gpd.GeoDataFrame.from_features(gdf, crs="EPSG:27700")
    m = sub(m)
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
