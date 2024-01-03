import os, sys, copy
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
import tensorflow as tf

# Set API key, details from https://github.com/OrdnanceSurvey/osdatahub
load_dotenv()
key_os = os.environ['KEY']

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

                # Add a constant value to the list
                #constant_column_values.append(1)  # You can use any constant value

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
    def __init__(self, lyr, collections, subsets=None):
        self.lyr = lyr
        self.collections = collections
        self.subsets = subsets

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
    def export_combined_raster(self, output_file):
        # Create the combined mask GeoDataFrame
        combined_gdf = self.create_combined_mask()

        # Read the original raster file using rasterio
        with rasterio.open(self.lyr) as src:
            # Convert the combined GeoDataFrame to a raster
            cube = make_geocube(combined_gdf, measurements=['pdnp'], resolution=src.res, fill=0)
            combined_raster = cube['pdnp'].astype('uint8')

            # Update metadata for the new raster
            new_meta = src.meta.copy()
            new_meta.update({'count': 1, 'dtype': 'uint8'})

            # Get the union of all geometries in the combined GeoDataFrame
            geometry_union = combined_gdf.unary_union

            # Apply the mask on the original raster
            masked, transform = rasterio.mask.mask(src, [geometry_union], crop=True, nodata=0)

            # Reshape the masked array to have only one band
            masked = masked[0]  # Select the first band

            # Create a new array filled with 255 (white)
            blank_array = np.full_like(masked, fill_value=255)

            # Copy original values to the blank array where the mask is zero
            blank_array[masked == 0] = src.read(1)[masked == 0]

            # Write the modified raster to a new file
            with rasterio.open(output_file, 'w', **new_meta) as dst:
                dst.write(blank_array, 1)