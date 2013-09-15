# Create your views here.
import sys
import os
from django.db.models import Q
from django.utils import timezone
from django.utils import simplejson as json
from django.http import HttpResponse, HttpResponseRedirect
from django.views.decorators.csrf import csrf_exempt
from s.models import User, Picture, Featured_Picture

import time as time
import numpy as np
import scipy as sp
import pylab as pl
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import Ward
from skimage.filter import denoise_tv_chambolle
import scipy.ndimage as ndI
from PIL import Image
import matplotlib.pyplot as plt
from skimage.segmentation import felzenszwalb, slic, quickshift
from skimage.segmentation import mark_boundaries
import matplotlib.cm as cm
from rectangle import Rectangle

#from skimage import filter

def calculate_final_image_matrix(shape_image, first_image_cluster_segments_std_array, second_image_cluster_segments_std_array):
    final_image_matrix = np.zeros(shape=(len(shape_image),len(shape_image[0])))

    for index_1 in range(len(shape_image)):
        for index_2 in range(len(shape_image[index_1])):

            first_image_delta = first_image_cluster_segments_std_array[index_1][index_2]
            second_image_delta = second_image_cluster_segments_std_array[index_1][index_2]

            #Compare modules
            if first_image_delta > second_image_delta:
                final_image_matrix[index_1][index_2] = 0
            else:
                final_image_matrix[index_1][index_2] = 1

    return final_image_matrix

def calculate_ambiguous_clusters(first_image_cluster_segments_std, second_image_cluster_segments_std, image_std_mean):
    ambiguous_low_std_clusters = []
    ambiguous_high_std_clusters = []

    for key, value in first_image_cluster_segments_std.iteritems():

        first_std = first_image_cluster_segments_std[key]
        second_std = second_image_cluster_segments_std[key]

        max_std = max([first_std, second_std])

        std_diff = np.absolute(first_std - second_std)

        if std_diff < 2.5 :

            if max_std < image_std_mean:
                ambiguous_low_std_clusters.append(key)
            else:
                ambiguous_high_std_clusters.append(key)

    low_ambiguous_cluster = np.zeros(shape=(len(transitional_matrix),len(transitional_matrix[0])))
    high_ambiguous_cluster = np.zeros(shape=(len(transitional_matrix),len(transitional_matrix[0])))

    for index_1 in range(len(transitional_matrix)):
        for index_2 in range(len(transitional_matrix[index_1])):

            cluster_tuple = transitional_matrix[index_1][index_2]

            if cluster_tuple in ambiguous_low_std_clusters:
                low_ambiguous_cluster[index_1][index_2] = 1

            elif cluster_tuple in ambiguous_high_std_clusters:
                high_ambiguous_cluster[index_1][index_2] = 1


    return [low_ambiguous_cluster, high_ambiguous_cluster, ambiguous_low_std_clusters, ambiguous_high_std_clusters]

def get_bounds(image_result_matrix, index_1, index_2):

    bounds = []

    if index_1 is not 0:
        bounds.append(image_result_matrix[index_1 - 1][index_2])

    if index_2 is not 0:
        bounds.append(image_result_matrix[index_1][index_2 - 1])

    return bounds


def create_segmented_cluster_std_estimate(first_std_matrix, second_std_matrix, transitional_matrix):
    first_image_cluster_segments_std = {}
    first_image_cluster_ocurrence = {}

    for index_1 in range(len(first_std_matrix)):
        for index_2 in range(len(first_std_matrix[index_1])):
            deviation_value = first_std_matrix[index_1][index_2]

            cluster_tuple = transitional_matrix[index_1][index_2]

            if cluster_tuple in first_image_cluster_segments_std:
                first_image_cluster_segments_std[cluster_tuple] = first_image_cluster_segments_std[cluster_tuple] + deviation_value
                first_image_cluster_ocurrence[cluster_tuple] = first_image_cluster_ocurrence[cluster_tuple] + 1
            else:
                first_image_cluster_segments_std[cluster_tuple] = deviation_value
                first_image_cluster_ocurrence[cluster_tuple] = 1

    for key in first_image_cluster_segments_std:
        first_image_cluster_segments_std[key] = first_image_cluster_segments_std[key] / first_image_cluster_ocurrence[key]

    second_image_cluster_segments_std = {}
    second_image_cluster_ocurrence = {}

    for index_1 in range(len(second_std_matrix)):
        for index_2 in range(len(second_std_matrix[index_1])):
            deviation_value = second_std_matrix[index_1][index_2]

            cluster_tuple = transitional_matrix[index_1][index_2]

            if cluster_tuple in second_image_cluster_segments_std:
                second_image_cluster_segments_std[cluster_tuple] = second_image_cluster_segments_std[cluster_tuple] + deviation_value
                second_image_cluster_ocurrence[cluster_tuple] = second_image_cluster_ocurrence[cluster_tuple] + 1
            else:
                second_image_cluster_segments_std[cluster_tuple] = deviation_value
                second_image_cluster_ocurrence[cluster_tuple] = 1

    for key in second_image_cluster_segments_std:
        second_image_cluster_segments_std[key] = second_image_cluster_segments_std[key] / second_image_cluster_ocurrence[key]

    return [first_image_cluster_segments_std, second_image_cluster_segments_std, first_image_cluster_ocurrence, second_image_cluster_ocurrence]


def merge_matrixes(first_label_matrix, second_label_matrix):
    final_matrix = np.empty((len(first_label_matrix), len(first_label_matrix[0])), dtype=object)

    for index_1 in range(len(first_label_matrix)):
        for index_2 in range(len(first_label_matrix[index_1])):
            final_matrix[index_1][index_2] = str(str(first_label_matrix[index_1][index_2]) + "-"+ str(second_label_matrix[index_1][index_2]))

    return final_matrix


def image_std(image):
    image_width, image_height = pil_image.size
    first_image = np.asarray(pil_image)
    image = pl.mean(first_image, 2)

    box_height = box_width = 3

    new_image = np.zeros(shape=(len(image),len(image[0])))

    for index_y in range(int(image_height/box_height)):
        for index_x in range(int(image_width/box_width)):

            points_array = [image[index_y * box_height, index_x * box_width], image[index_y * box_height,index_x * box_width + 1] , image[index_y * box_height + 1, index_x * box_width] , image[index_y * box_height + 1, index_x * box_width + 1], image[index_y * box_height + 2, index_x * box_width + 1], image[index_y * box_height + 1, index_x * box_width + 2], image[index_y * box_height, index_x * box_width + 2], image[index_y * box_height + 2, index_x * box_width], image[index_y * box_height + 2, index_x * box_width + 2]]
            points_std = np.std(points_array)

            new_image[index_y * box_height, index_x * box_width] = points_std
            new_image[index_y * box_height, index_x * box_width + 1] = points_std
            new_image[index_y * box_height + 1, index_x * box_width] = points_std
            new_image[index_y * box_height + 1, index_x * box_width + 1] = points_std
            new_image[index_y * box_height + 1, index_x * box_width + 2] = points_std
            new_image[index_y * box_height + 2, index_x * box_width + 1] = points_std
            new_image[index_y * box_height + 2, index_x * box_width + 2] = points_std
            new_image[index_y * box_height + 2, index_x * box_width + 0] = points_std
            new_image[index_y * box_height + 0, index_x * box_width + 2] = points_std
            # points_array1 = []
            # for box_height_index in range(box_height):
            #    for box_width_index in range(box_width):
            #        points_array1.append(image[index_y * box_height + box_height_index, index_x * box_width_index])
            # 
            # points_std = np.std(points_array1)
            # 
            # for box_height_index in range(box_height):
            #    for box_width_index in range(box_width):
            #        new_image1[index_y * box_height + box_height_index, index_x * box_width + box_width_index] = points_std


    return new_image


@csrf_exempt
def test(request):

    ##############################################################################################################################################################
    ## INITIAL PARAMETERS
    ##############################################################################################################################################################
    st = time.time()
    
    scale = 0.2
    first_slic_ratio = 1
    first_n_segments = 1
    first_sigma = 1
    second_slic_ratio = 2
    second_n_segments = 8
    second_sigma =3
    first_max_iter = 10
    second_max_iter = 10
    
    ###############################################################################
    ## REQUEST POST VALUES
    ###############################################################################
    user_id = request.POST['user_id']
    first_image = Image.open(cStringIO.StringIO(request.FILES['first_image'].read())).convert('RGB')
    second_image = Image.open(cStringIO.StringIO(request.FILES['second_image'].read())).convert('RGB')
    
    ###############################################################################
    ## SCALE AND STD FIRST IMAGE
    ###############################################################################
    width, height = first_image.size
    if scale != 1:
        first_image = first_image.resize((int(scale*width), int(scale*height)), Image.ANTIALIAS)
    first_image = np.asarray(first_image)
    first_image_final = np.copy(first_image)
    first_std_image = image_std(first_image)
    
    ###############################################################################
    ## SCALE AND STD SECOND IMAGE
    ###############################################################################
    if scale != 1:
        second_image = second_image.resize((int(scale*width), int(scale*height)), Image.ANTIALIAS)
    second_image = np.asarray(second_image)
    second_image_final = np.copy(second_image)
    second_std_image = image_std(second_image)
    
    ###############################################################################
    ## ITERATIVE CLUSTERING
    ###############################################################################
    clustering_index = 0:
    is_clustering_done = False
    while not is_clustering_done:
        
        ###############################################################################
        #FIRST IMAGE
        ###############################################################################
        first_label = np.zeros(shape=(len(first_image),len(first_image[0])))
        first_image = pl.mean(first_image,2)
        first_label = np.reshape(first_label, first_image.shape)
        
        ###############################################################################
        #SECOND IMAGE
        ###############################################################################
        second_label = slic(second_image, ratio=second_slic_ratio, n_segments=second_n_segments, sigma=second_sigma, max_iter=second_max_iter)
        second_image = pl.mean(second_image,2)
        second_label = np.reshape(second_label, second_image.shape)
        
        
        ###############################################################################
        # CALCULATE SEGMENTS MEANS
        ###############################################################################
        transitional_matrix = second_label
        
        segmented_stds = create_segmented_cluster_std_estimate(first_std_image, second_std_image, transitional_matrix)
        first_image_cluster_segments_std = segmented_stds[0]
        second_image_cluster_segments_std = segmented_stds[1]
        first_image_cluster_segments_ocurrence = segmented_stds[2]
        second_image_cluster_segments_ocurrence = segmented_stds[3]
        
        first_image_cluster_std_array = []
        second_image_cluster_std_array = []
        
        for key, value in first_image_cluster_segments_std.iteritems():
            first_image_cluster_std_array.append(value)
        
        for key, value in second_image_cluster_segments_std.iteritems():
            second_image_cluster_std_array.append(value)
        
        first_image_std_mean = np.mean(first_image_cluster_std_array)
        second_image_std_mean = np.mean(second_image_cluster_std_array)
        
        image_std_mean = min([second_image_std_mean, first_image_std_mean])
        
        ###############################################################################
        # CALCULATE AMBIGUOUS CLUSTERS
        ###############################################################################
        abiguous_clusters = calculate_ambiguous_clusters(first_image_cluster_segments_std, second_image_cluster_segments_std, image_std_mean)
        low_ambiguous_cluster = abiguous_clusters[0]
        high_ambiguous_cluster = abiguous_clusters[1]
        
        low_ambiguous_array = abiguous_clusters[2]
        high_ambiguous_array = abiguous_clusters[3]
        
        ###############################################################################
        # CALCULATE AREA OF EACH AMBIGUITY
        ###############################################################################
        low_ambiguous_area = 0
        for low_index in range(len(low_ambiguous_array)):
            low_ambiguous_area += first_image_cluster_segments_ocurrence[low_ambiguous_array[low_index]]
        
        high_ambiguous_area = 0
        for high_index in range(len(high_ambiguous_array)):
            high_ambiguous_area += first_image_cluster_segments_ocurrence[high_ambiguous_array[high_index]]
            
        low_ambiguous_clusters_row = []
        low_ambiguous_clusters_col = []
        
        high_ambiguous_clusters_row = []
        high_ambiguous_clusters_col = []
        
        for index_row in range(len(low_ambiguous_cluster)):
            for index_col in range(len(low_ambiguous_cluster[index_row])):
                if low_ambiguous_cluster[index_row][index_col] == 1:
                    low_ambiguous_clusters_row.append(index_row)
                    low_ambiguous_clusters_col.append(index_col)
        low_row_std = np.std(low_ambiguous_clusters_row)
        low_col_std = np.std(low_ambiguous_clusters_col)
        
        for index_row in range(len(high_ambiguous_cluster)):
            for index_col in range(len(high_ambiguous_cluster[index_row])):
                if high_ambiguous_cluster[index_row][index_col] == 1:
                    high_ambiguous_clusters_row.append(index_row)
                    high_ambiguous_clusters_col.append(index_col)
        high_row_std = np.std(high_ambiguous_clusters_row)
        high_col_std = np.std(high_ambiguous_clusters_col)
        
        
        ###############################################################################
        # CALCULATE DIFFERENCE OF AMBIGUITIES AREAS
        ###############################################################################
        ambiguous_dif = low_ambiguous_area - high_ambiguous_area
        
        ###############################################################################
        # CALCULATE NEW CLUSTERING PARAMETERS
        ###############################################################################
        if np.absolute(ambiguous_dif) < 200 or clustering_index == 3:
            is_clustering_done = True
            print "ITERATION FINISHED AT INDEX ", clustering_index
        else:
            if clustering_index == 0:
                if ambiguous_dif > 0:
                    second_n_segments = second_n_segments - 4
                else:
                    second_n_segments = second_n_segments + 2
                    
            elif clustering_index == 1:
                if ambiguous_dif > 0:
                    second_n_segments = second_n_segments - 1
                else:
                    second_n_segments = second_n_segments + 1
                    
            elif clustering_index == 2:
                if ambiguous_dif > 0:
                    second_sigma = second_sigma + 2
                else:
                    second_sigma = second_sigma - 2
                    
            clustering_index = clustering_index + 1


    ###############################################################################
    # CALCULATE THE 2D MEAN STD() OF EACH CLUSTER
    ###############################################################################
    first_image_cluster_segments_std_array = np.zeros(shape=(len(transitional_matrix),len(transitional_matrix[0])))
    second_image_cluster_segments_std_array = np.zeros(shape=(len(transitional_matrix),len(transitional_matrix[0])))
    for index_1 in range(len(transitional_matrix)):
        for index_2 in range(len(transitional_matrix[index_1])):
            transitional_tuple = transitional_matrix[index_1][index_2]
            first_image_cluster_segments_std_array[index_1][index_2] = first_image_cluster_segments_std[transitional_tuple]
            second_image_cluster_segments_std_array[index_1][index_2] = second_image_cluster_segments_std[transitional_tuple]



    ###############################################################################
    # CALCULATE FINAL IMAGE MATRIX
    ###############################################################################
    final_image_matrix = calculate_final_image_matrix(first_image_final, first_image_cluster_segments_std_array, second_image_cluster_segments_std_array)

    ###############################################################################
    # CREATE GEOMETRY ANALISYS
    ###############################################################################
    first_image_result_matrix = np.zeros(shape=(len(first_image_final),len(first_image_final[0])), dtype=np.int)
    second_image_result_matrix = np.zeros(shape=(len(first_image_final),len(first_image_final[0])), dtype=np.int)

    geometry_image_cluster_index = int(1)

    geometries_first_row_index_dict = {}

    for index_1 in range(len(final_image_matrix)):
        for index_2 in  range(len(final_image_matrix[index_1])):

            if int(final_image_matrix[index_1][index_2]) is 0:
                is_first = True
                bounds = get_bounds(first_image_result_matrix, index_1, index_2)
            else:
                is_first = False
                bounds = get_bounds(second_image_result_matrix, index_1, index_2)

            if len(bounds) is 0:
                if is_first:
                    first_image_result_matrix[index_1][index_2] = geometry_image_cluster_index
                else:
                    second_image_result_matrix[index_1][index_2] = geometry_image_cluster_index
                geometry_image_cluster_index = int(geometry_image_cluster_index + 1)
                geometries_first_row_index_dict[geometry_image_cluster_index] = index_1

            elif len(bounds) is 1:

                if int(bounds[0]) is 0:
                    if is_first:
                        first_image_result_matrix[index_1][index_2] = geometry_image_cluster_index
                    else:
                        second_image_result_matrix[index_1][index_2] = geometry_image_cluster_index
                    geometry_image_cluster_index = int(geometry_image_cluster_index + 1)
                    geometries_first_row_index_dict[geometry_image_cluster_index] = index_1

                else:
                    if is_first:
                        first_image_result_matrix[index_1][index_2] = bounds[0]
                    else:
                        second_image_result_matrix[index_1][index_2] = bounds[0]

            elif len(bounds) is 2:

                if int(bounds[0]) is 0 and int(bounds[1]) is 0:

                    if is_first:
                        first_image_result_matrix[index_1][index_2] = geometry_image_cluster_index
                    else:
                        second_image_result_matrix[index_1][index_2] = geometry_image_cluster_index
                    geometry_image_cluster_index = int(geometry_image_cluster_index + 1)
                    geometries_first_row_index_dict[geometry_image_cluster_index] = index_1

                elif int(bounds[0]) == int(bounds[1]) or int(bounds[0]) == 0 or int(bounds[1]) == 0:
                    # ASSIGN
                    if is_first:
                        if int(bounds[0]) is 0:
                            first_image_result_matrix[index_1][index_2] = bounds[1]
                        else:
                            first_image_result_matrix[index_1][index_2] = bounds[0]
                    else:
                        if int(bounds[0]) is 0:
                            second_image_result_matrix[index_1][index_2] = bounds[1]
                        else:
                            second_image_result_matrix[index_1][index_2] = bounds[0]
                else:
                    #MERGE AND ASSIGN
                    should_break = False
                    lower_geometry = int(max(bounds))
                    upper_geometry = int(min(bounds))

                    if is_first:
                        geometry_row_translation = geometries_first_row_index_dict[lower_geometry] - 1

                        for index_final_1 in range(len(first_image_result_matrix)):
                            index_final_1_moved = index_final_1 + geometry_row_translation

                            if should_break or index_final_1_moved == len(first_image_result_matrix):
                                break

                            for index_final_2 in range(len(first_image_result_matrix[index_final_1_moved])):

                                if index_final_1_moved == index_1 and index_final_2 == index_2:
                                    should_break = True
                                    break

                                elif int(first_image_result_matrix[index_final_1_moved][index_final_2]) is lower_geometry:
                                    first_image_result_matrix[index_final_1_moved][index_final_2] = upper_geometry


                        first_image_result_matrix[index_1][index_2] = upper_geometry
                    else:

                        for index_final_1 in range(len(second_image_result_matrix)):
                            if should_break:
                                break

                            for index_final_2 in range(len(second_image_result_matrix[index_final_1])):
                                if index_final_1 == index_1 and index_final_2 == index_2:
                                    should_break = True
                                    break
                                elif int(second_image_result_matrix[index_final_1][index_final_2]) is lower_geometry:
                                    second_image_result_matrix[index_final_1][index_final_2] = upper_geometry

                        second_image_result_matrix[index_1][index_2] = upper_geometry

    ###############################################################################
    # CALCULATE AREA OF EACH GEOMETRY
    ###############################################################################
    first_image_unique_geometries = np.unique(first_image_result_matrix)
    second_image_unique_geometries = np.unique(second_image_result_matrix)

    first_image_geometry_count = {}
    second_image_geometry_count = {}

    for geometry_index in first_image_unique_geometries:
        if int(geometry_index) is not 0:
            geometry_count = 0
            for index_1 in range(len(first_image_result_matrix)):
                for index_2 in range(len(first_image_result_matrix[index_1])):        
                    if int(first_image_result_matrix[index_1][index_2]) is int(geometry_index):
                        geometry_count = geometry_count + 1

            first_image_geometry_count[geometry_index] = geometry_count

    for geometry_index in second_image_unique_geometries:
        if int(geometry_index) is not 0:
            geometry_count = 0
            for index_1 in range(len(second_image_result_matrix)):
                for index_2 in range(len(second_image_result_matrix[index_1])):        
                    if int(second_image_result_matrix[index_1][index_2]) is int(geometry_index):
                        geometry_count = geometry_count + 1

            second_image_geometry_count[geometry_index] = geometry_count


    ###############################################################################
    # REMOVE GEOMETRIES WITH SMALL AREAS
    ###############################################################################
    final_image = np.copy(first_image_final)

    final_image_cluster = np.copy(final_image_matrix)

    for index_1 in range(len(final_image_matrix)):
        for index_2 in range(len(final_image_matrix[index_1])):

            if int(final_image_matrix[index_1][index_2]) is 0:
                #first_image
                if first_image_geometry_count[first_image_result_matrix[index_1][index_2]] < 3000:
                    final_image_cluster[index_1][index_2] = 1
            else:
                #second_image
                if second_image_geometry_count[second_image_result_matrix[index_1][index_2]] < 3000:
                    final_image_cluster[index_1][index_2] = 0

    for index_1 in range(len(final_image_cluster)):
        for index_2 in range(len(final_image_cluster[index_1])):
            if int(final_image_cluster[index_1][index_2]) is 0:
                final_image[index_1][index_2] = first_image_final[index_1][index_2]
            else:
                final_image[index_1][index_2] = second_image_final[index_1][index_2]

    ###############################################################################
    # APPLY BLURR TO BORDERS
    ###############################################################################
    final_blurred_image = np.copy(final_image)

    for index_row in range(len(final_image_cluster)):
        for index_col in range(len(final_image_cluster[index_row])):

            iteration_image_value = final_image_cluster[index_row][index_col]

            pixels_array = []

            ### TOP PIXELS
            if index_row > 0:
                up_image_value = final_image_cluster[index_row-1][index_col]
                if up_image_value != iteration_image_value:
                    pixels_array.append(final_blurred_image[index_row-1][index_col])

                if index_row > 1:
                    top_up_image_value = final_image_cluster[index_row-2][index_col]
                    if top_up_image_value != iteration_image_value:
                        pixels_array.append(final_blurred_image[index_row-2][index_col])

            ## BOTTOM PIXELS
            if index_row < (len(final_image_cluster) - 1):
                down_image_value = final_image_cluster[index_row+1][index_col]
                if down_image_value != iteration_image_value:
                    pixels_array.append(final_blurred_image[index_row+1][index_col])

                if index_row < (len(final_image_cluster) - 2):
                    top_down_image_value = final_image_cluster[index_row+2][index_col]
                    if top_down_image_value != iteration_image_value:
                        pixels_array.append(final_blurred_image[index_row+2][index_col])

            ## LEFT PIXELS 
            if index_col > 0:
                left_image_value = final_image_cluster[index_row][index_col-1]
                if left_image_value != iteration_image_value:
                    pixels_array.append(final_blurred_image[index_row][index_col-1])

                if index_col > 1:
                    top_left_image_value = final_image_cluster[index_row][index_col-2]
                    if top_left_image_value != iteration_image_value:
                        pixels_array.append(final_blurred_image[index_row][index_col-2])

            ## RIGHT PIXELS
            if index_col < (len(final_image_cluster[index_row]) - 1):
                right_image_value = final_image_cluster[index_row][index_col+1]
                if right_image_value != iteration_image_value:
                    pixels_array.append(final_blurred_image[index_row][index_col+1])

                if index_col < (len(final_image_cluster[index_row]) - 2):

                    top_right_image_value = final_image_cluster[index_row][index_col+2]
                    if top_right_image_value != iteration_image_value:
                        pixels_array.append(final_blurred_image[index_row][index_col+2])

            if len(pixels_array) > 0:

                pixels_array.append(final_image[index_row][index_col])

                red_pixels = []
                green_pixels = []
                blue_pixels = []

                for pixel_index in range(len(pixels_array)):
                   rgb_array = pixels_array[pixel_index]

                   red_pixels.append(rgb_array[0])
                   green_pixels.append(rgb_array[1])
                   blue_pixels.append(rgb_array[2])

                red_mean = pl.mean(red_pixels)
                green_mean = pl.mean(green_pixels)
                blue_mean = pl.mean(blue_pixels)

                final_blurred_image[index_row][index_col] = [red_mean, green_mean, blue_mean]


    
    ###############################################################################
    # SEND IMAGE TO BOTO  -> TODO
    ###############################################################################
    #final_blurred_image    
    
    ###############################################################################
    # RETURN HTTP RESPONSE WITH IMAGE URL -> TODO
    ###############################################################################
    
    
    response_data = {"result": "OK"}

    return HttpResponse(json.dumps(response_data), mimetype="aplication/json")
    