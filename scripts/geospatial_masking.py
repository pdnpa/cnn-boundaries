'''
Run this script to mask a geotiff file using;
1. osdatahub (variables for data API call can be changed)
2. Text mask using keras-ocr API
This will output a geotiff file with the blank mask printed over the 
original geotiff and shapefile of the mask used for analysis.

Assumes you need to enable GPU memory growth. 
'''
import os
import sys
import copy
import tensorflow as tf
from typing import Collection
from osdatahub import Extent, NGD
from dotenv import load_dotenv
from shapely.geometry import Polygon, MultiPolygon
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from rasterio.enums import Resampling
from geocube.api.core import make_geocube
import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
import keras_ocr

# Enable GPU memory growth
def enable_gpu_memory_growth():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Enabled GPU memory growth")
        except RuntimeError as e:
            # Memory growth must be set at program startup
            print(f"Failed to set memory growth: {e}")

# Load environment variables and API key
load_dotenv()
key_os = os.getenv('KEY')
assert key_os, "API key not found in environment variables."

class Mask:
    def __init__(self, lyr, collections, subsets=None):
        self.lyr = lyr
        self.collections = collections
        self.subsets = subsets

    def os_mask(self):  # API call
        bbox = self.bb()
        dict_os_layers = {}
        gdf = gpd.GeoDataFrame()

        def urban_mask(gdf):  # Creates an urban mask of villages from OS API
            gdf = gpd.GeoDataFrame.from_features(gdf, crs="EPSG:27700")
            gdf['pdnp'] = 'pdnp'
            gdf = gdf.dissolve(by='pdnp', as_index=False)
            gdf = gdf.buffer(10)  # Set pixel value
            gdf = gpd.GeoDataFrame.from_features(gdf, crs="EPSG:27700")
            return gdf
        
        for key, value in self.collections.items():
            ngd = NGD(key_os, value)
            ngd_ = "ngd_" + key
            result = ngd.query(extent=bbox, crs="EPSG:27700", max_results=1000000000)

            if result['numberReturned'] > 0:
                dict_os_layers[ngd_] = result
                d = gpd.GeoDataFrame.from_features(result, crs="EPSG:27700")
                gdf = pd.concat([gdf, d])
                print(ngd_ + " query contains the following number of features:" + str(result['numberReturned']))
            else:
                print(key + " query is empty!")
        
        print("-- Total OS Collections Data Set Contains: " + str(gdf.shape[0]) + " values")

        for i in self.subsets.values():
            x = list(i)
            
        sub_a = gdf.loc[gdf['description'].isin(x)]  # Extract the subsets
        sub_b = gdf.loc[~gdf['theme'].isin(self.subsets.keys())]  # Drop the land from the original df
        gdf = pd.concat([sub_a, sub_b])

        print("-- Final Subsetted Data Set Contains: " + str(gdf.shape[0]))
        
        gdf = urban_mask(gdf)
        gdf = gpd.clip(gdf, bbox.polygon)

        # Ensure the 'pdnp' column exists
        if 'pdnp' not in gdf.columns:
            gdf['pdnp'] = 'pdnp'

        return gdf

    def bb(self):  # Creates a BBOX of input layers
        with rasterio.open(self.lyr) as src:
            extent_crs = str(src.crs)
            extent = Extent.from_bbox(src.bounds, crs=extent_crs)
        return extent

class Text_Mask:
    def __init__(self, lyr):
        self.lyr = lyr

    def k_mask(self):
        pipeline = keras_ocr.pipeline.Pipeline()
        img = keras_ocr.tools.read(self.lyr)
        prediction_groups = pipeline.recognize([img])
        keras_ocr.tools.drawAnnotations(image=img, predictions=prediction_groups[0])
        
        polygons = []

        with rasterio.open(self.lyr) as src:
            transform = src.transform
            for word, vertices_array in prediction_groups[0]:
                vertices = vertices_array.tolist()
                coords = [transform * (vertex[0], vertex[1]) for vertex in vertices]
                polygon = Polygon(coords)
                polygons.append(polygon)

        print("-- Detected " + str(len(polygons)) + " words or letters")

        gdf = gpd.GeoDataFrame(geometry=[MultiPolygon(polygons)], crs=src.crs)
        gdf['pdnp'] = 'pdnp'
        gdf = gdf.dissolve(by='pdnp', as_index=False) 

        if isinstance(gdf, MultiPolygon):
            gdf = gdf.convex_hull

        return gdf
    
class RasterPlotter:
    def __init__(self, lyr):
        self.lyr = lyr

    def plot_over_raster(self, gdf):
        with rasterio.open(self.lyr) as src:
            fig, ax = plt.subplots(figsize=(10, 10))
            show(src, ax=ax, cmap='gray')
            gdf.plot(ax=ax, facecolor='white', edgecolor='white', linewidth=1)
            plt.show()

        if gdf is None or gdf.empty:
            print("Merged mask is empty.")
        return

class CombinedMask:
    def __init__(self, lyr, collections, subsets=None, input_image_path=None):
        self.lyr = lyr
        self.collections = collections
        self.subsets = subsets
        self.input_image_path = input_image_path

    def create_combined_mask(self):
        mask_obj = Mask(self.lyr, self.collections, self.subsets)
        mask_gdf = mask_obj.os_mask()

        text_mask_obj = Text_Mask(self.lyr)
        text_mask_gdf = text_mask_obj.k_mask()

        text_mask_gdf = text_mask_gdf.to_crs(mask_gdf.crs)

        combined_gdf = gpd.GeoDataFrame(pd.concat([mask_gdf, text_mask_gdf], ignore_index=True), crs=mask_gdf.crs)
        combined_gdf = combined_gdf.dissolve(by='pdnp', as_index=False)
        combined_gdf['pdnp'] = 1

        return combined_gdf
    
    def plot_combined_mask(self):
        combined_gdf = self.create_combined_mask()
        raster_plotter_obj = RasterPlotter(self.lyr)
        raster_plotter_obj.plot_over_raster(combined_gdf)

    def export_combined_mask(self, raster_output_path, shp_output_path):
        combined_gdf = self.create_combined_mask()
        shp_filename = os.path.join(shp_output_path, os.path.splitext(os.path.basename(self.lyr))[0] + '.shp')
        combined_gdf.to_file(shp_filename)

        with rasterio.open(self.lyr) as src:
            assert src.count == 3, f'The raster must have 3 bands (but currently has {src.count})'
            assert src.meta['dtype'] == 'uint8', f'The raster must have dtype uint8 (but currently has {src.meta["dtype"]})'
            assert len(src.shape) == 2, f'Expected 2 dimensions, got {len(src.shape)}'
            full_shape_src = (src.count, src.shape[0], src.shape[1])

            cube = make_geocube(combined_gdf, measurements=['pdnp'], resolution=src.res, fill=0)
            combined_raster = cube['pdnp'].astype('uint8')

            new_meta = src.meta.copy()

            geometry_union = combined_gdf.unary_union
            masked, transform = rasterio.mask.mask(src, [geometry_union], crop=True, nodata=0)
            masked = masked.squeeze()

            assert len(masked.shape) == 3 and len(full_shape_src) == 3, f'Expected 3 dimensions, got {len(masked.shape)} and {len(full_shape_src)}'

            blank_array = np.full_like(masked, fill_value=255)
            min_height = min(masked.shape[1], full_shape_src[1])
            min_width = min(masked.shape[2], full_shape_src[2])

            blank_array[:, :min_height, :min_width][masked[:, :min_height, :min_width] == 0] = src.read([1, 2, 3])[:, :min_height, :min_width][masked[:, :min_height, :min_width] == 0]

            raster_filename = os.path.join(raster_output_path, os.path.splitext(os.path.basename(self.lyr))[0] + '_combined.tif')
            with rasterio.open(raster_filename, 'w', **new_meta) as dst:
                dst.write(blank_array)

        return shp_filename, raster_filename

def main():   
    enable_gpu_memory_growth()

    collections = {
        'buildings': 'bld-fts-buildingpart-1',
        'sites': 'lus-fts-site-1', 
        'railways': 'trn-fts-rail-1', 
        'land': 'lnd-fts-land-1',
        'water': 'wtr-fts-waterpoint-1',
        'road': 'trn-fts-roadline-1',
        'track': 'trn-fts-roadtrackorpath-1',
        'waterlink': 'wtr-ntwk-waterlink-1',
        'waterlinkset': 'wtr-ntwk-waterlinkset-1'
    }

    subsets = {'Land': ['Made Surface', 'Residential Garden', 'Non-Coniferous Trees', 'Coniferous Trees', 'Mixed Trees']}
    
    tiles = ["SK1070.tif", "SK1468.tif", "SK1469.tif", "SK1474.tif", "SK1476.tif",
             "SK1567.tif", "SK1570.tif", "SK1668.tif", "SK1678.tif", "SK1767.tif",
             "SK1768.tif", "SK1867.tif", "SK1868.tif"]

    shp_folder = "../QGIS/masks/"
    raster_output_folder = "../content/tifs/masked/"

    for tile in tiles:
        tile_path = os.path.join("../content/tifs/1k_tifs/", tile)
        combined_mask = CombinedMask(tile_path, collections, subsets)
        shp_filename, raster_filename = combined_mask.export_combined_mask(raster_output_folder, shp_folder)

        print(f"Exported for {tile}:")
        print("Shapefile exported to:", shp_filename)
        print("Raster file exported to:", raster_filename)
        print("=" * 50)

if __name__ == "__main__":
    main()