'''
Run this script to mask a geotiff file using;
1. osdatahub (variables for data API call can be changed)
2. Text mask using keras-ocr API
This will output a geotiff file with the blank mask printed over the 
original geotiff and shapefile of the mask used for analysis
'''
import os, sys, copy
import tensorflow as tf

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

# Call the function 
enable_gpu_memory_growth()

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


# Set API key, details from https://github.com/OrdnanceSurvey/osdatahub
load_dotenv()
key_os = os.environ['KEY']
assert key_os, "API key not found in environment variables."

class Mask:
    def __init__(self, lyr, collections, subsets = None):
        self.lyr = lyr
        self.collections = collections
        self.subsets = subsets

    def os_mask(self):  # api call
        bbox = self.bb()
        dict_os_layers = {}
        gdf = gpd.GeoDataFrame()

        def urban_mask(gdf):  # creates an urban mask of villages from os api
            gdf = gpd.GeoDataFrame.from_features(gdf, crs="EPSG:27700")
            gdf['pdnp'] = 'pdnp'
            gdf = gdf.dissolve(by='pdnp', as_index=False)
            gdf = gdf.buffer(10)  # set pixel value
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
            
        sub_a = gdf.loc[gdf['description'].isin(x)]  # extract the subsets
        sub_b = gdf.loc[~gdf['theme'].isin(self.subsets.keys())]  # drop the land from the original df
        gdf = pd.concat([sub_a, sub_b])

        print("-- Final Subsetted Data Set Contains: " + str(gdf.shape[0]))
        
        gdf = urban_mask(gdf)
        gdf = gpd.clip(gdf, bbox.polygon)

        # Ensure the 'pdnp' column exists
        if 'pdnp' not in gdf.columns:
            gdf['pdnp'] = 'pdnp'

        return gdf

    def bb(self):  # creates a BBOX of input layers
        # Open the raster file using the stored path
        with rasterio.open(self.lyr) as src:
            # Convert the CRS to a string before passing it to Extent.from_bbox
            extent_crs = str(src.crs)
            extent = Extent.from_bbox(src.bounds, crs=extent_crs)
        return extent

    def wall_subset(gdf):  # takes the subset based on built obstructions
        gdf = gpd.GeoDataFrame.from_features(gdf, crs="EPSG:27700")
        # subset based on 'Built Obstruction'
        gdf = gdf.loc[gdf['description'] == 'Built Obstruction']
        return gdf

class Text_Mask:
    def __init__(self, lyr):
        self.lyr = lyr

    def k_mask(self):
        pipeline = keras_ocr.pipeline.Pipeline()
        img = keras_ocr.tools.read(self.lyr)
        # Prediction_groups is a list of (word, box) tuples
        prediction_groups = pipeline.recognize([img]) 
        keras_ocr.tools.drawAnnotations(image=img, predictions=prediction_groups[0])
        
        polygons = []
        #constant_column_values = []  # Add this list to store constant values

        # Read the raster file using rasterio to get georeferencing information
        with rasterio.open(self.lyr) as src:
            transform = src.transform  # Get the affine transformation
            for word, vertices_array in prediction_groups[0]:
                # Convert the vertices array to a list
                vertices = vertices_array.tolist()

                # Transform image pixel coordinates to geographical coordinates
                coords = [transform * (vertex[0], vertex[1]) for vertex in vertices]

                # Create a Shapely Polygon
                polygon = Polygon(coords)
                polygons.append(polygon)

        print("-- Detected " + str(len(polygons)) + " words or letters")

        # Create a GeoDataFrame with the correct CRS
        gdf = gpd.GeoDataFrame(geometry=[MultiPolygon(polygons)], crs=src.crs)

        # Create a constant pdnp column for all polygons in the text mask
        gdf['pdnp'] = 'pdnp'

        # dissolve the result in to one polygon
        gdf = gdf.dissolve(by='pdnp', as_index=False) 

        # Convert the MultiPolygon to a Polygon
        if isinstance(gdf, MultiPolygon):
            gdf = gdf.convex_hull

        return gdf
    
class RasterPlotter:
    def __init__(self, lyr):
        self.lyr = lyr

    def plot_over_raster(self, gdf):
        # Read the raster file using rasterio
        with rasterio.open(self.lyr) as src:
            # Plot the raster
            fig, ax = plt.subplots(figsize=(10, 10))
            show(src, ax=ax, cmap='gray')

            # Plot the GeoDataFrame over the raster
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
        self.input_image_path = input_image_path  # Store input image path

    def create_combined_mask(self):
        # Create Mask object
        mask_obj = Mask(self.lyr, self.collections, self.subsets)
        mask_gdf = mask_obj.os_mask()

        # Create Text_Mask object
        text_mask_obj = Text_Mask(self.lyr)
        text_mask_gdf = text_mask_obj.k_mask()

        # Ensure both GeoDataFrames have the same CRS
        text_mask_gdf = text_mask_gdf.to_crs(mask_gdf.crs)

        # Merge GeoDataFrames based on 'pdnp' column
        combined_gdf = gpd.GeoDataFrame(pd.concat([mask_gdf, text_mask_gdf], ignore_index=True), crs=mask_gdf.crs)
        
        # dissolve the result in to one polygon
        combined_gdf = combined_gdf.dissolve(by='pdnp', as_index=False)

        combined_gdf['pdnp'] = 1

        return combined_gdf
    
    def plot_combined_mask(self):
        combined_gdf = self.create_combined_mask()

        # Create RasterPlotter object
        raster_plotter_obj = RasterPlotter(self.lyr)
        raster_plotter_obj.plot_over_raster(combined_gdf)

    # if you want to export as a shp for testing call: combined_mask.export_combined_mask('filepath')
    def export_combined_mask(self, raster_output_path, shp_output_path):
        # Create the combined mask GeoDataFrame
        combined_gdf = self.create_combined_mask()

        # Export combined GeoDataFrame to shapefile
        shp_filename = os.path.join(shp_output_path, os.path.splitext(os.path.basename(self.lyr))[0] + '.shp')
        combined_gdf.to_file(shp_filename)

        # Read the original raster file using rasterio
        with rasterio.open(self.lyr) as src:
            ## Some asserts to ensure that src is as expected:
            assert src.count == 3, f'The raster must have 3 bands (but currently has {src.count})'
            assert src.meta['dtype'] == 'uint8', f'The raster must have dtype uint8 (but currently has {src.meta["dtype"]})'
            assert len(src.shape), f'Expected 2 dimensions, got {len(src.shape)}'
            full_shape_src = (src.count, src.shape[0], src.shape[1])

            # Convert the combined GeoDataFrame to a raster
            cube = make_geocube(combined_gdf, measurements=['pdnp'], resolution=src.res, fill=0)
            combined_raster = cube['pdnp'].astype('uint8')

            # Update metadata for the new raster
            new_meta = src.meta.copy()
            # new_meta.update({'count': 1, 'dtype': 'uint8'})  ## don't need this anymore

            # Get the union of all geometries in the combined GeoDataFrame
            geometry_union = combined_gdf.unary_union

            masked, transform = rasterio.mask.mask(src, [geometry_union], crop=True, nodata=0)
            # masked = masked[0]  # Select the first band

            # Reshape the masked array to have only one band
            masked = masked.squeeze()  # Remove singleton dimensions
            print("Squeezed masked array shape:", masked.shape)

            ## Ensure that they are the same dimension: (not the same shape exactly because they might be 1-2 pixels different)
            assert len(masked.shape) == 3 and len(full_shape_src) == 3, f'Expected 3 dimensions, got {len(masked.shape)} and {len(full_shape_src)}'

            # Create a new array filled with 255 (white)
            blank_array = np.full_like(masked, fill_value=255)

            # Ensure dimensions match before copying values (not of [0] because that is the number of bands)
            min_height = min(masked.shape[1], full_shape_src[1])
            min_width = min(masked.shape[2], full_shape_src[2])

            # Print dimensions for debugging
            print("Minimum height:", min_height)
            print("Minimum width:", min_width)

            # Copy original values to the blank array where the mask is zero | added extra : for first dimension (Bands)
            blank_array[:, :min_height, :min_width][masked[:, :min_height, :min_width] == 0] = src.read([1, 2, 3])[:, :min_height, :min_width][masked[:, :min_height, :min_width] == 0]
            # Write the modified raster to a new file
            raster_filename = os.path.join(raster_output_path, os.path.splitext(os.path.basename(self.lyr))[0] + '_combined.tif')
            with rasterio.open(raster_filename, 'w', **new_meta) as dst:
                dst.write(blank_array)

        return shp_filename, raster_filename
    

if __name__ == "__main__":
    load_dotenv()
    key_os = os.getenv('KEY')
    assert key_os, "API key not found in environment variables."

    # define os collections for urban mask aggreation
    collections = {'buildings' : 'bld-fts-buildingpart-1',
                'sites' : 'lus-fts-site-1', 
                'railways' : 'trn-fts-rail-1', 
                'land' : 'lnd-fts-land-1',
                'water' : 'wtr-fts-waterpoint-1',
                'road' : 'trn-fts-roadline-1',
                'track' : 'trn-fts-roadtrackorpath-1',
                'waterlink' : 'wtr-ntwk-waterlink-1',
                'waterlinkset' : 'wtr-ntwk-waterlinkset-1',
                'road-track-path' : 'trn-ntwk-pathlink-1'}

    # if specifics are required from 'collection' define which to keep from 'description' column
    subsets = {'Land' : ['Made Surface', 
                        'Residential Garden', 
                        'Non-Coniferous Trees', 
                        'Coniferous Trees', 
                        'Mixed Trees']}
    
    # replace with all 500 1 km2 (eventually)
    tiles = ["SK1070.tif", "SK1468.tif", "SK1469.tif", "SK1474.tif", "SK1476.tif",
            "SK1567.tif", "SK1570.tif", "SK1668.tif", "SK1678.tif", "SK1767.tif",
            "SK1768.tif", "SK1867.tif", "SK1868.tif"]

    shp_folder = "../QGIS/masks/"  # output for shp mask
    raster_output_folder = "../content/tifs/masked/"  # output for raster mask

    # Iterate over each tile and export combined mask
    for tile in tiles:
        tile_path = os.path.join("../content/tifs/1k_tifs/", tile)  # Adjust the path to match your directory structure

        # Create CombinedMask object
        combined_mask = CombinedMask(tile_path, collections, subsets)

        # Export combined mask for the current tile
        shp_filename, raster_filename = combined_mask.export_combined_mask(raster_output_folder, shp_folder)

        print(f"Exported for {tile}:")
        print("Shapefile exported to:", shp_filename)
        print("Raster file exported to:", raster_filename)
        print("=" * 50)