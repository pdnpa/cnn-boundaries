import os
from osdatahub import Extent
from osdatahub import NGD
import geopandas as gpd
import rasterio

def bb(lyr):
    if isinstance (lyr, rasterio.io.DatasetReader):
        t = Extent.from_bbox(lyr.bounds, crs="EPSG:27700")
        return t
    else:
        g = round(lyr.bounds)
        return g # note this only works with pd

def urban_mask(gdf):
    m = gpd.GeoDataFrame.from_features(gdf, crs="EPSG:27700")
    # subset based on made surface and garden for proper urban
    m = (m.loc[(m['description']== 'Made Surface') | (m['description']== 'Residential Garden')])
    m.unary_union
    m['pdnp'] = 'pdnp'
    m = m.dissolve(by = 'pdnp', as_index=False)
    m = m.buffer(10)
    return m

def wall_mask(gdf):
    walls = gpd.GeoDataFrame.from_features(gdf, crs="EPSG:27700")
    #subset based on 'Built Obstruction'
    walls = walls[walls['description'] == 'Built Obstruction']
    return walls
