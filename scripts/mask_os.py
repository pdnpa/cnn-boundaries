import os, sys, copy
from typing import Collection

from osdatahub import Extent, NGD
from dotenv import load_dotenv
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from rasterio.enums import Resampling
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
        constant_column_values = []  # Add this list to store constant values

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
                constant_column_values.append(1)  # You can use any constant value

        # Create a GeoDataFrame with the correct CRS
        gdf = gpd.GeoDataFrame(geometry=polygons, crs=src.crs)

        # Add a constant column to the GeoDataFrame
        gdf['pdnp'] = constant_column_values

        print("-- Detected " + str(gdf.shape[0]) + " words or letters")
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
        self.merged_mask = None  # Initialize merged_mask as an instance variable

    def merge_masks(self):
        # Create instances of Mask and Text_Mask
        mask_instance = Mask(self.lyr, self.collections, self.subsets)
        text_mask_instance = Text_Mask(self.lyr)

        # Obtain the individual masks
        os_mask = mask_instance.os_mask()
        text_mask = text_mask_instance.k_mask()

        # Create a constant pdnp column for all polygons in the text mask
        text_mask['pdnp'] = 1
        # Merge the os_mask polygons using unary_union, preserving the 'pdnp' column
        merged_os_mask = os_mask.dissolve(by='pdnp', aggfunc='first').unary_union
        # Create a GeoDataFrame from the merged_os_mask geometry
        merged_os_mask_gdf = gpd.GeoDataFrame(geometry=[merged_os_mask], crs=os_mask.crs)

        # Reproject merged_os_mask_gdf to a common CRS
        common_crs = "EPSG:27700"  # Choose a common CRS, you may change this
        merged_os_mask_gdf = merged_os_mask_gdf.to_crs(common_crs)

        # Reproject os_mask_gdf to the common CRS
        os_mask_gdf = os_mask.to_crs(common_crs)
        os_mask_gdf['pdnp'] = 1

        # Reproject text_mask to the common CRS
        text_mask = text_mask.to_crs(common_crs)

        # Concatenate the GeoDataFrames
        merged_mask = gpd.GeoDataFrame(pd.concat([merged_os_mask_gdf, os_mask_gdf, text_mask], ignore_index=True))
        # Dissolve the merged mask
        dissolved_merged_mask = merged_mask.dissolve(by='pdnp', as_index=False)

        # Check if the dissolved_merged_mask is not empty
        if dissolved_merged_mask.empty:
            print("Merged mask is empty!")
        else:
            print("Merged mask successfully created.")
            # Get the bounding box polygon
            bbox = mask_instance.bb().polygon
            # clip the final output
            dissolved_merged_mask = gpd.clip(dissolved_merged_mask, bbox)

            # Set self.merged_mask
            self.merged_mask = dissolved_merged_mask

        return self.merged_mask
    
    def export_and_combine(self, output_path):
        # Get metadata from the original raster
        with rasterio.open(self.lyr) as src:
            lyr_meta = {
                'dtype': src.dtypes[0],
                'crs': src.crs,
                'transform': src.transform,
                'height': src.height,
                'width': src.width
            }

        # Create an instance of GeoDataFrameRasterizer with lyr_meta
        rasterizer = GeoDataFrameRasterizer(lyr_meta)

        # Export the merged mask to a raster
        self.merge_masks()  # Call merge_masks to populate self.merged_mask

        # Check if self.merged_mask is not None before calling info()
        if self.merged_mask is not None:
            print("Merged Mask GeoDataFrame Info:")
            print(self.merged_mask.info())
            print("Merged Mask Geometry Column:")
            print(self.merged_mask['geometry'])
        else:
            print("Merged mask is None. Export and combine process aborted.")
            return

        rasterizer.export_to_raster(
            self.merged_mask,
            output_path,
            attribute=None,
            fill_value=0,
            resampling=Resampling.nearest
        )

        unique_values = np.unique(self.merged_mask['pdnp'].dropna())
        print("Mask Values before combining:", unique_values)

        # Combine the exported mask with the underlying raster
        rasterizer.combine_with_underlying(output_path, output_path)
    
class GeoDataFrameRasterizer:
    def __init__(self, lyr_meta):
        self.lyr_meta = lyr_meta
        self.shape = lyr_meta['height'], lyr_meta['width']

    def export_to_raster(self, gdf, output_path, attribute=None, fill_value=0, resampling=Resampling.nearest):
        # Check if the GeoDataFrame is empty or does not have a valid geometry
        if gdf is None or gdf.empty or 'geometry' not in gdf.columns:
            raise ValueError("GeoDataFrame is empty or does not have a valid geometry.")

        # Create a blank raster
        raster = np.full(self.shape, fill_value, dtype=self.lyr_meta['dtype'])

        # Check if the GeoDataFrame has a valid geometry column
        if gdf['geometry'].is_empty.any():
            raise ValueError("GeoDataFrame has empty or invalid geometries.")
        
        # Add debugging print statements
        print("GeoDataFrame Info:")
        print(gdf.info())
        print("GeoDataFrame Geometry Column:")
        print(gdf['geometry'])

        # Rasterize the GeoDataFrame onto the blank raster
        mask = geometry_mask(gdf.geometry, out_shape=self.shape, transform=self.lyr_meta['transform'], invert=True)

        # Ensure that the mask has integer values
        mask = mask.astype(self.lyr_meta['dtype'])

        print("Rasterized Mask Values before combining:", np.unique(mask))

        if attribute is not None:
            rasterize_values = gdf[attribute].values
            raster[mask] = rasterize_values

        # Write the raster to a GeoTIFF file
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=self.shape[0],
            width=self.shape[1],
            count=1,
            dtype=self.lyr_meta['dtype'],
            crs=self.lyr_meta['crs'],
            transform=self.lyr_meta['transform']
        ) as dst:
            dst.write(raster, 1)

        # Add debugging print statement
        print("Rasterized Mask Values:", np.unique(raster))

    def combine_with_underlying(self, mask_path, output_path):
        # Open the underlying raster
        with rasterio.open(mask_path) as src:
            # Read the underlying raster data
            underlying_data = src.read(1)
            underlying_transform = src.transform
            underlying_crs = src.crs

        # Open the exported mask raster
        with rasterio.open(output_path) as src:
            # Read the exported mask raster data
            mask_data = src.read(1)
            mask_transform = src.transform
            mask_crs = src.crs

        # Ensure the CRS and transform match
        if underlying_crs != mask_crs or underlying_transform != mask_transform:
            raise ValueError("CRS or transform mismatch between the underlying and mask rasters.")

        # Convert mask to integer before combining
        mask_data = mask_data.astype(int)

        # Create a combined raster with the same properties as the underlying raster
        combined_data = np.where((underlying_data > 0) | (mask_data > 0), 1, 0)

        # Write the combined raster to a new GeoTIFF file
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=combined_data.shape[0],
            width=combined_data.shape[1],
            count=1,
            dtype=combined_data.dtype,
            crs=underlying_crs,
            transform=underlying_transform
        ) as dst:
            dst.write(combined_data, 1)

        print("Mask Values:", np.unique(mask_data))
        print("Underlying Values:", np.unique(underlying_data))