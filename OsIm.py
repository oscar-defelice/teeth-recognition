###
### OsIm.py
###
### Created by Oscar de Felice on 10/11/2019.
### Copyright © 2019 Oscar de Felice.
###
### This program is free software: you can redistribute it and/or modify
### it under the terms of the GNU General Public License as published by
### the Free Software Foundation, either version 3 of the License, or
### (at your option) any later version.
###
### This program is distributed in the hope that it will be useful,
### but WITHOUT ANY WARRANTY; without even the implied warranty of
### MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
### GNU General Public License for more details.
###
### You should have received a copy of the GNU General Public License
### along with this program. If not, see <http://www.gnu.org/licenses/>.
###
########################################################################
###
### OsIm.py
### This is a module to collect classes definitions useful for the script.
###
### 08/11/2019 - Oscar: creation of repository and first commit.
### 10/11/2019 - Oscar: creation of modules and first version of the script.
### 13/11/2019 - Oscar: image class - defined all basic attributes and methods.
### 14/11/2019 - Oscar: image class - added the plot methods.
### 15/11/2019 - Oscar: image comparator class - defined all basic attributes and methods.
### 17/11/2019 - Oscar: image comparator class - match method added.
###

### import Libraries ###
import numpy as np
import cv2
import matplotlib.pyplot as plt


### constants definition ###
LOWE_THRS = 0.7
DEFAULT_FIGSIZE = (10,15) # Default size for image plots.
FLANN_INDEX_KDTREE = 1
N_FLANN_TREES = 5
N_FLANN_CHECKS = 50
EDGE_THRS = 20 # SIFT edgeThreshold parameter

### models definitions ###
sift = cv2.xfeatures2d.SIFT_create(edgeThreshold = EDGE_THRS)
surf = cv2.xfeatures2d.SURF_create(extended = True)

class Image:
    """
        Class for image loading and processing.

        Parameters
        ----------
        color:  boolean, optional, default=True
                It specifies to load a color image.
                Any transparency of image will be neglected.
                It is the default flag.
                Alternatively, we can pass integer value 1 for this flag.

        grayscale:  boolean, optional, default=False
                    It specifies to load an image in grayscale mode.
                    Alternatively, we can pass integer value 0 for this flag.

        unchanged:  boolean, optional, default=False
                    It specifies to load an image as such including alpha channel.
                    Alternatively, we can pass integer value -1 for this flag.

        Attributes
        ----------
        size_ : tuple (n_pixels_row, n_pixels_columns, n_channels)
                Image dimension in pixels.

        path_ : str
                Path location of the file.

        img_  : obj
                Image object.


        Notes
        -----
        From the implementation point of view, this is just a collection of methods to import and preprocess an image,
        by using several openCV method.

        TODO: include menpo project methods.
    """

    def __init__(self, path_to_file, flag = 1):
        """
            Constructor method for the image.
            One parameter is the path to the image file.
            The second (optional) parameter is the image read mode from openCV.
            Another attribute comes from the imread() method from openCV.
        """
        self.path_ = path_to_file
        self.img_ = cv2.imread(self.path_, flag)
        self.size_ = self.img_.shape

    def __toGray(self):
        """
            private method for the class.
            It makes use of the cvtColor method of openCV to convert the color image to a gray scale.
        """
        return cv2.cvtColor(self.img_, cv2.COLOR_BGR2GRAY)

    def __model_selection(self, model_name):
        """
            Private method to define which feature detecting algorithm to use.

            model_name is a string the name of the model.
            It returns the model object.
        """
        if model_name == 'sift':
            model = sift
        elif model_name == 'surf':
            model = surf
        else:
            raise ValueError('The only implemented models are sift and surf.')

        return model


    def keypoints(self, model_name):
        """
            Method to calculate keypoints and descriptors.

            The argument model_name is a string the name of the model.

            returns a tuple of lists.
            keypoints is a list of keypoint objects.
            descriptors is a list of arrays encoding the features vector.
        """
        model = self.__model_selection(model_name)
        keypoints, descriptors = model.detectAndCompute(self.img_, None)
        return keypoints, descriptors


    def plotKeypoints(self, model_name, figsize = DEFAULT_FIGSIZE):
        """
            Method to plot the image with keypoints.

            model_name is a string indicating which algorithm to use.
            figsize is a tuple tuning the plot size.
        """

        img_to_plot = cv2.drawKeypoints(self.__toGray(), self.keypoints(model_name)[0], self.img_)
        plt.figure(figsize=DEFAULT_FIGSIZE)
        plt.imshow(img_to_plot)

    def plotImage(self, figsize = DEFAULT_FIGSIZE):
        """
            Method to plot the image.

            figsize is a tuple tuning the plot size.
        """
        plt.figure(figsize=DEFAULT_FIGSIZE)
        plt.imshow(self.img_), plt.show()



class ImageComparator:
    """
        Class for image comparison.

        Parameters
        ----------
        matcher :   str, optional, default = 'bf'
                    This is the string controlling the feature matching method.
                    Default is Brute Force.
                    Admitted values:
                        'bf', 'flann'

        threshold :     float, optional, default = 0.7
                        This is the threshold value to consider whether a match is good or to be rejected.
                        The default value corresponds to the 1999 Lowe paper.

        Attributes
        ----------
        matcher :   object,
                    which model has been used to calculate keypoints.


        Examples
        --------

        Notes
        -----

    """

    def __init__(self, matcher):
        """
            Constructor method for comparator.
            It takes one argument, the matcher (a str) indicating the kind of matcher we want
        """
        if matcher not in ['bf', 'flann']:
            raise NotImplementedError('Only brute-force and Flann methods are implemented for matching.')
        else:
            self.matcher_ = matcher
            self.match_model_ = __match_selection(matcher)

    @staticmethod
    def __match_selection(match_model):
        """
            Private method to select the matching scheme.

            For the moment being the only two admitted arguments are
        """
        if match_model == 'bf': # feature matching method - Brute Force
            matcher =  cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
        elif match_model == 'flann': # feature matching method - Flann method
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = N_FLANN_TREES)
            search_params = dict(checks = N_FLANN_CHECKS)   # or pass empty dictionary

            matcher = cv2.FlannBasedMatcher(index_params,search_params)

        return matcher
        
    def match(self, descriptors1, descriptors2):
        """
            match method.
            
            It takes two arrays of descriptors as input.
            It returns a list of matches objects.
        """
        return self.match_model_.match(descriptors1, descriptors2)
