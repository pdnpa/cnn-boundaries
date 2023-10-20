import rasterio
from osdatahub import Extent
from osdatahub import NGD
from osdatahub import Extent
import geopandas as gpd

def bb(tif):
    t = rasterio.open(tif)
    b = Extent.from_bbox(t.bounds, crs="EPSG:27700")
    return b

def urban_mask(gdf):
    m = gpd.GeoDataFrame.from_features(gdf, crs="EPSG:27700")
    m = (m.loc[(m['description']== 'Made Surface') | (m['description']== 'Residential Garden')])
    m.unary_union
    m['pdnp'] = 'pdnp'
    m = m.dissolve(by = 'pdnp', as_index=False)
    m = m.buffer(10)
    return m
