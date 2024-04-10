## Functions for wall extraction from historic maps 
import os
import numpy as np
import cv2
import PIL.Image
from skimage.transform import probabilistic_hough_line
from skimage.feature import canny
from skimage.filters import threshold_otsu
from osgeo import gdal, osr
import geopandas as gpd
from shapely.geometry import LineString
import rasterio
from rasterio.transform import Affine
import skimage.measure
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
if tuple([int(x) for x in matplotlib.__version__.split('.')]) < (3, 5, 0):
    print(f'MPL version {matplotlib.__version__} is too old to support plt.axline(), so do not use this function or update to 3.5.0 or higher')
import skimage.transform
import pandas as pd 

def binarise_array(arr, binarise_threshold=190):
    assert arr.max() <= 255 and arr.min() >= 0
    binarise_threshold = 250
    assert binarise_threshold >= 0 and binarise_threshold <= 255
    arr_bin = arr < binarise_threshold  # (values go from 0 to 255)
    assert arr_bin.sum() / arr_bin.size < 0.5, 'Binarised image is not as sparse as expected, perhaps increase threshold?' 
    # assert (np.unique(arr_bin) == np.array([0, 1])).all(), 'Binarised image is not binary, so something went wrong'
    return arr_bin

def hough_transform_array(arr_bin):
    ## Follow this example for Hough transform: https://scikit-image.org/docs/stable/auto_examples/edges/plot_line_hough_transform.html
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
    h, theta, d = skimage.transform.hough_line(arr_bin, theta=tested_angles)

    angle_step = 0.5 * np.diff(theta).mean()
    d_step = 0.5 * np.diff(d).mean()
    bounds = [np.rad2deg(theta[0] - angle_step),
            np.rad2deg(theta[-1] + angle_step),
            d[-1] + d_step, d[0] - d_step]
    return h, theta, d, bounds

def plot_hough_transform(h, bounds, ax=None):
    '''Plot Hough transform output. h and bounds are outputs of skimage.transform.hough_line() (or hough_transform_array() above))'''
    if ax is None:
        ax = plt.subplot(111)

    ax.imshow(np.log(1 + h), extent=bounds, cmap=matplotlib.cm.gray, aspect=1 / 1.5)
    ax.set_title('Hough transform')
    ax.set_xlabel('Angles (degrees)')
    ax.set_ylabel('Distance (pixels)')
    ax.axis('image')

def find_start_and_end_of_inferred_lines(h, theta, d, arr_bin, min_duration=11):
    '''Hough transform has identified lines in the image. 
   But these are without start and end points (i.e., just the slope and start point).
   This function then finds the start and end of these lines, by looking at the image
   and seeing if there is a black pixel in the vicinity of the line to determine
   the start and end of the line.

   ------
   Parameters:
    h, theta, d: outputs of skimage.transform.hough_line()
    arr_bin: binarised image (0 = black, 1 = white)
    min_duration: minimum duration of a line (in pixels) to be considered a line.
    '''
    assert h.ndim == 2 and theta.ndim == 1 and d.ndim == 1, 'h, theta, d should be 2D, 1D, 1D respectively'
    assert h.shape[0] == len(d) and h.shape[1] == len(theta), f'h.shape = {h.shape}, theta.shape = {theta.shape}, d.shape = {d.shape}'

    list_inferred_lines = []  # store inferred lines in list of tuples: ((x_start, x_stop), (y_start, y_stop))

    for _, angle, dist in zip(*skimage.transform.hough_line_peaks(h, theta, d)):
        ## Compute linear line from hough transform:
        (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])  # point on the (edge) line
        slope_line = np.tan(angle + np.pi / 2)  # slope of line 
        # offset = y0 - slope_line * x0 

        ## Find out what parts of the linear line contain pixels:
        x_line_present_arr = np.zeros(arr_bin.shape[1])
        for xx in range(arr_bin.shape[1]):  # loop through x coordinates; (x/y bit messy here but works out)
            y_middle = int(np.round(y0 + (xx - x0) * slope_line))  # get centre y coord in integers

            ## Now check if there is a black pixel in the vicinity of the line at this x-coordinate. (Check the vicinity along y-axis):
            ## (Maybe replace by the sum of the vicinity: for lines it should be 1-2, while for letters it'll be more.) 
            for yy in range(y_middle - 2, y_middle + 1):  # look in vicinity 
                if yy >= 0 and yy < arr_bin.shape[0]:  # if in range of image
                    if arr_bin[yy, xx] == 1:  # if a black pixel exists; accept
                        x_line_present_arr[xx] = 1  # store that there is a black pixel at this x-coordinate
                        break  # only 1 pixel needed, so break out of loop

        ## Find continuous segments (of present pixels along this line):
        diff_pres = np.diff(x_line_present_arr)  # difference between nth and n+1th element
        inds_change = np.where(diff_pres != 0)[0]  # where is an up/down change
        if len(inds_change) == 0:  # in this case; the line exist across all pixels (in this image) 
            x_start = 0
            x_stop = arr_bin.shape[1]
            list_inferred_lines.append(((x_start, x_stop), ( y0 + (x_start - x0) * slope_line, y0 + (x_stop - x0) * slope_line)))
        elif len(inds_change) == 1:  # only 1 line segment:
            if diff_pres[inds_change[0]] == -1:  # only up till down (i.e., a line exist until it doesnt)
                x_start = 0
                x_stop = inds_change[0]
            elif diff_pres[inds_change[0]] == 1:  # only down till up  (i.e., no line exist until it does)
                x_start = inds_change[0]
                x_stop = arr_bin.shape[1]
            list_inferred_lines.append(((x_start, x_stop), ( y0 + (x_start - x0) * slope_line, y0 + (x_stop - x0) * slope_line)))
        else:
            # (x_start, x_stop) = inds_change[np.argmax(np.diff(inds_change))], inds_change[np.argmax(np.diff(inds_change)) + 1]  # get longest continouous segmenet 

            ## Find each segment with minimum duration        
            up_inds = np.where(diff_pres == 1)[0]  # (where a line exists)
            down_inds = np.where(diff_pres == - 1)[0]  # (where a line doesnt exist)
            if down_inds[0] < up_inds[0]:  # has started on up
                up_inds = np.concatenate((np.array([0]), up_inds))
            if down_inds[-1] < up_inds[-1]: # ended on up 
                down_inds = np.concatenate((down_inds, np.array([arr_bin.shape[0]]))) 
            assert len(up_inds) == len(down_inds)
            duration_up = [down_inds[ii] - up_inds[ii] for ii in range(len(up_inds))]
            
            for i_seg, dur in enumerate(duration_up):
                if dur >= min_duration:
                    x_start = up_inds[i_seg]
                    x_stop = down_inds[i_seg]
                    list_inferred_lines.append(((x_start, x_stop), ( y0 + (x_start - x0) * slope_line, y0 + (x_stop - x0) * slope_line)))


    ## Put all inferred lines in pandas dataframe:
    df_inferred_lines = pd.DataFrame(list_inferred_lines, columns=['x', 'y'])

    return list_inferred_lines, df_inferred_lines

def save_image(image_rgb, source_image, output_file):
    """
    Save the processed image to a file.
    """
    driver = gdal.GetDriverByName('GTiff')
    output_dataset = driver.Create(output_file, image_rgb.shape[1], image_rgb.shape[0], 3, gdal.GDT_Byte)
    if output_dataset is None:
        raise ValueError(f"Could not create output file: {output_file}")
    for i in range(3):
        output_dataset.GetRasterBand(i + 1).WriteArray(image_rgb[:, :, i])
    output_dataset.SetGeoTransform(source_image.GetGeoTransform())
    output_dataset.SetProjection(source_image.GetProjection())
    output_dataset = None

def apply_line_mask(input_image_path, output_folder):
    """
    Apply a mask to filter out a specific color range and save the result.
    """
    image = gdal.Open(input_image_path)
    image_data = image.ReadAsArray()
    image_rgb = np.dstack((image_data[0], image_data[1], image_data[2]))

    lower_orange = np.array([160, 60, 30])
    upper_orange = np.array([255, 255, 255])
    mask = cv2.inRange(image_rgb, lower_orange, upper_orange)
    image_rgb[mask > 0] = (255, 255, 255)

    # Adjusted line: Directly replace "_combined.tif" with "_line_mask.tif" in the filename
    output_filename = os.path.basename(input_image_path).replace("_combined.tif", "_line_mask.tif")
    output_file = os.path.join(output_folder, output_filename)
    save_image(image_rgb, image, output_file)
    
def apply_hough_transform(input_image_path, output_folder):
    """
    Apply Hough Transform to detect lines, transform them to geographic coordinates, and save to a shapefile.
    """
    #im = PIL.Image.open(input_image_path).convert("L")
    #ima = np.array(im).reshape(im.size[::-1])
    #ima = ima[7:-7, 7:-7]

    # Apply binarization with the predefined threshold
    #ima_bin = binarise_array(ima)

    # Canny edge detection and Hough Transform
    #edges = canny(ima_bin, sigma=0.9, low_threshold=0.1, high_threshold=0.9)
    #lines = probabilistic_hough_line(edges, threshold=1, line_length=40, line_gap=10)

    # Transform lines for shapefile
    #with rasterio.open(input_image_path) as src:
    #    srs = src.crs
    #    transformed_lines = [LineString([(src.xy(y, x)[0], src.xy(y, x)[1]) for x, y in line]) for line in lines]
    #    gdf = gpd.GeoDataFrame(geometry=transformed_lines, crs=srs)
    #    output_shapefile = os.path.join(output_folder, os.path.basename(input_image_path).replace("_line_mask.tif", "_hough_lines.shp"))
    #    gdf.to_file(output_shapefile)
    
    #return lines  # Return the raw line coordinates for visualization
def apply_hough_transform(input_image_path, output_folder):
    with rasterio.open(input_image_path) as src:
        original_transform = src.transform
        ima = src.read(1)  # Read the first band into an array

        # Crop the image by removing 7 pixels from each edge
        cropped_ima = ima[7:-7, 7:-7]

        # Calculate the new geotransform after cropping
        new_transform = Affine.translation(7, -7) * original_transform

        # Apply binarization with the predefined threshold
        ima_bin = binarise_array(cropped_ima)  # Ensure you have this function defined

        # Canny edge detection and Hough Transform
        edges = canny(ima_bin, sigma=0.9, low_threshold=0.1, high_threshold=0.9)
        lines = probabilistic_hough_line(edges, threshold=1, line_length=40, line_gap=10)

        # Transform lines for shapefile using the NEW geotransform
        transformed_lines = [
            LineString([
                (new_transform * (x, y)) for x, y in line
            ]) for line in lines
        ]
        
        gdf = gpd.GeoDataFrame(geometry=transformed_lines, crs=src.crs)
        output_shapefile = os.path.join(
            output_folder, 
            os.path.basename(input_image_path).replace("_line_mask.tif", "_hough_lines.shp")
        )
        gdf.to_file(output_shapefile)

    return lines  # Return the raw line coordinates for visualization


def visualize_process(input_image_path, lines):
    im = PIL.Image.open(input_image_path).convert("L")
    ima = np.array(im).reshape(im.size[::-1])
    # Assuming binarise_array is a function that you have defined elsewhere
    ima_binarized = binarise_array(ima)  
    edges = canny(ima, sigma=0.9, low_threshold=0.1, high_threshold=0.9)

    # Adjust to 4 rows and 1 column for vertical layout
    fig, axes = plt.subplots(4, 1, figsize=(6, 24))  
    ax = axes.ravel()

    ax[0].imshow(im, cmap=cm.gray)
    ax[0].set_title('Input image')

    # Assuming binarise_array returns an appropriately binarized image
    ax[1].imshow(ima_binarized, cmap='Greys')  
    ax[1].set_title('Binarized image')

    ax[2].imshow(edges, cmap=cm.gray)
    ax[2].set_title('Canny edges')

    ax[3].imshow(edges * 0)
    for line in lines:
        p0, p1 = line
        ax[3].plot((p0[0], p1[0]), (p0[1], p1[1]), color='red')
    ax[3].set_title('Probabilistic Hough')

    for a in ax:
        a.axis('off')

    plt.tight_layout()
    plt.show()

def visualize_process_separately(input_image_path, output_base_path):
    # Load the original image
    im = PIL.Image.open(input_image_path).convert("L")
    ima = np.array(im)
    ima = np.array(im).reshape(im.size[::-1])
    ima = ima[3:-3, 3:-3]

    
    # Binarize the image
    threshold = 128
    ima_binarized = (ima > threshold) * 255
    
    # Canny edge detection
    edges = canny(ima, sigma=0.9, low_threshold=0.1, high_threshold=0.9)
    
    # Probabilistic Hough Transform
    lines = probabilistic_hough_line(edges, threshold=1, line_length=40, line_gap=10)
    
    # Step 1: Original Image
    plt.figure(figsize=(10, 10))
    plt.imshow(im, cmap='gray')
    #plt.title('Line Mask')
    plt.axis('off')
    plt.savefig(f'{output_base_path}_line_mask.png', dpi=150, bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close()
    
    # Step 2: Binarized Image
    plt.figure(figsize=(10, 10))
    plt.imshow(ima_binarized, cmap='gray')
    #plt.title('Binarized Image')
    plt.axis('off')
    plt.savefig(f'{output_base_path}_binarized.png', dpi=150, bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close()
    
    # Step 3: Canny Edges
    plt.figure(figsize=(10, 10))
    plt.imshow(edges, cmap='gray')
    #plt.title('Canny Edges')
    plt.axis('off')
    plt.savefig(f'{output_base_path}_canny_edges.png', dpi=150, bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close()
    
    # Step 4: Probabilistic Hough
    plt.figure(figsize=(10, 10))
    plt.imshow(im, cmap='gray')  # Show the original image as the background
    for line in lines:
        p0, p1 = line
        plt.plot((p0[0], p1[0]), (p0[1], p1[1]), 'red')  # Plot lines in red
    #plt.title('Probabilistic Hough Lines')
    plt.axis('off')
    plt.savefig(f'{output_base_path}_probabilistic_hough_lines.png', dpi=150, bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close()