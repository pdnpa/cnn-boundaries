import os, sys, copy
from typing import Collection

from osdatahub import Extent, NGD
from dotenv import load_dotenv
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show
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
        return gdf

    def bb(self):  # creates a BBOX of input layers
        self.lyr = rasterio.open(self.lyr)
        self.lyr = Extent.from_bbox(self.lyr.bounds, crs="EPSG:27700")
        return self.lyr  # note this only works with pd

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

        # Create a GeoDataFrame with the correct CRS
        gdf = gpd.GeoDataFrame(geometry=polygons, crs=src.crs)

        print("Detected " + str(gdf.shape[0]) + " words or letters")
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
            gdf.plot(ax=ax, facecolor='white', edgecolor='white', linewidth=0)
            
            plt.show()

class CombinedMask:
    def __init__(self, lyr, collections, subsets=None):
        self.lyr = lyr
        self.collections = collections
        self.subsets = subsets

    def merge_masks(self):
        # Create instances of Mask and Text_Mask
        mask_instance = Mask(self.lyr, self.collections, self.subsets)
        text_mask_instance = Text_Mask(self.lyr)

        # Obtain the individual masks
        os_mask = mask_instance.os_mask()
        text_mask = text_mask_instance.k_mask()

        # Merge the masks using overlay
        merged_mask = gpd.overlay(os_mask, text_mask, how="union")

        return merged_mask
    

    




  


