import os, sys, copy
from typing import Collection
from osdatahub import Extent
from dotenv import load_dotenv
from osdatahub import NGD
from shapely.geometry import Polygon
import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio

# Set API key, details from https://github.com/OrdnanceSurvey/osdatahub
load_dotenv()
key_os = os.environ['KEY']

class Mask:

    def __init__(self, lyr, collections, subsets):
        self.lyr = lyr
        self.collections = collections
        self.subsets = subsets

    def os_mask(self): # api call
        bbox = self.lyr
        dict_os_layers = {}
        gdf = gpd.GeoDataFrame()
        for key, value in self.collections.items():
            ngd = NGD(key_os, value)
            ngd_ = "ngd_" + key
            dict_os_layers[ngd_] = ngd.query(extent=bbox, crs="EPSG:27700", max_results=1000000000)
            if dict_os_layers[ngd_]['numberReturned'] > 0:
                dict_os_layers.update(dict_os_layers[ngd_]) # add to empty dict
                d =  gpd.GeoDataFrame.from_features((dict_os_layers[ngd_]), crs="EPSG:27700")
                gdf = pd.concat([gdf, d])
                print(ngd_ + " query contains the following number of features:" + str(dict_os_layers[ngd_]['numberReturned']))
            else:
                print(key + " query is empty!")
        print("-- Total OS Collections Data Set Contains: " + str(gdf.shape[0]) + " values" )
        for i in self.subsets.values():
            x = list(i)
        sub_a = gdf.loc[gdf['description'].isin(x)] # extract the subsets
        sub_b = gdf.loc[~gdf['theme'].isin([i for i in self.subsets.keys()])] # drop the land from the origianl df
        gdf = pd.concat([sub_a, sub_b])
        print("-- Final Subsetted Data Set Contains: " + str(gdf.shape[0]))
        gdf = urban_mask(gdf)
        gdf = gpd.clip(gdf, bbox.polygon)
        return(gdf)

    def bb(self): # creates a BBOX of input layers
        if isinstance (self.lyr, rasterio.io.DatasetReader):
            self.lyr = Extent.from_bbox(self.lyr.bounds, crs="EPSG:27700")
            return self.lyr
        else:
            self.lyr = round(self.lyr.bounds)
            return self.lyr # note this only works with pd

def urban_mask(gdf): # creates an urban mask of villages from os api
    gdf = gpd.GeoDataFrame.from_features(gdf, crs="EPSG:27700")
    gdf.unary_union
    gdf['pdnp'] = 'pdnp'
    gdf = gdf.dissolve(by = 'pdnp', as_index=False)
    gdf = gdf.buffer(10) # set pixel value
    gdf = gpd.GeoDataFrame.from_features(gdf, crs ="EPSG:27700")
    return gdf

def wall_subset(gdf): # takes the subset based on built obstructions
    w = gpd.GeoDataFrame.from_features(gdf, crs="EPSG:27700")
    #subset based on 'Built Obstruction'
    w = (w.loc[w['description'] == 'Built Obstruction'])
    return w