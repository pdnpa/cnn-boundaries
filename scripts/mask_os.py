import os, sys, copy
from typing import Collection
from osdatahub import Extent
from dotenv import load_dotenv
from osdatahub import NGD
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
        for i in self.subsets.values():
            x = list(i)
        sub_a = gdf.loc[gdf['description'].isin(x)] # extract the subsets
        sub_b = gdf.loc[~gdf['theme'].isin([i for i in self.subsets.keys()])] # drop the land from the origianl df
        gdf = pd.concat([sub_a, sub_b])
        gdf = urban_mask(gdf)
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

'''
Turn gdf of shape file polygon into a raster file. Possibly store & plot.
Assumes col_name is a numeric column with class labels.

interpolation:
    - None: nothing done with missing data (turned into 0)
    - 'nearest': using label of nearest pixels (takes bit of extra time)
'''

def convert_shp_mask_to_raster(df_shp, col_name='theme',
                                resolution=( -0.6713209989263084, 0.6713215494092012),
                                interpolation=None, 
                                save_raster=False, filename='mask.tif',
                                maskdir=None, plot_raster=False,
                                verbose=0):

    assert len(resolution) == 2 and resolution[0] < 0 and resolution[1] > 0, 'resolution has unexpected size/values'

    ## Convert shape to raster:
    assert len(df_shp) > 0, 'df_shp is empty'

    cube = make_geocube(df_shp, measurements=[col_name],
                        interpolate_na_method=interpolation,
                        resolution=resolution,
                        fill=0)
    
    if col_name in df_shp.columns and col_name not in cube.data_vars:
        print(cube)
    shape_cube = cube[col_name].shape  # somehow sometimes an extra row or of NO CLASS is added... 
    if shape_cube[0]  == 1491:
        if len(np.unique(cube[col_name][0, :])) > 1:
            print(f'WARNING: {filename} has shape {shape_cube} but first y-row contains following classes: {np.unique(cube[col_name][:, 0])}. Still proceeding..')    
        cube = cube.isel(y=np.arange(1, 1491))  #discard first one that is just no classes 
    if shape_cube[1] == 1490:
        if len(np.unique(cube[col_name][:, 0])) > 1:
            print(f'WARNING: {filename} has shape {shape_cube} but first x-col contains following classes: {np.unique(cube[col_name][:, 0])}. Still proceeding..')    
        cube = cube.isel(x=np.arange(1, 1490))  #discard first one that is just no classes 

    ## Decrease data size:
    if verbose > 0:
        print(f'Current data size cube is {cube.nbytes / 1e6} MB')
    unique_labels = copy.deepcopy(np.unique(cube[col_name]))  # want to ensure these are not messed up 
    assert np.nanmin(unique_labels) >=0 and np.nanmax(unique_labels) < 256, f'unexpectedly high number of labels. conversion to int8 wont work. Labels: {unique_labels}'
    low_size_raster = cube[col_name].astype('uint8')  # goes from 0 to & incl 255
    cube[col_name] = low_size_raster
    new_unique_labels = np.unique(cube[col_name])
    assert (unique_labels == new_unique_labels).all(), f'Old labels: {unique_labels}, new labels: {new_unique_labels}, comaprison: {(unique_labels == new_unique_labels)}'  # unique labels are sorted by default so this works as sanity check
    if verbose > 0:
        print(f'New cube data size is {cube.nbytes / 1e6} MB')

    if save_raster:
        assert type(filename) == str, 'filename must be string'
        if filename[-4:] != '.tif':
            filename = filename + '.tif'
        if maskdir is None:  # use default path for mask files 
            maskdir = "../content/tifs/"
        # print(maskdir, filename)
        filepath = os.path.join(maskdir, filename)
        cube[col_name].rio.to_raster(filepath)
        if verbose > 0:
            print(f'Saved to {filepath}')

    if plot_raster:
        cube[col_name].plot()

    return cube 