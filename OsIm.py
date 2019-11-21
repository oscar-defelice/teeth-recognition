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
### 13/11/2019 - Oscar: Image class - defined all basic attributes and methods.
### 14/11/2019 - Oscar: Image class - added the plot methods.
### 15/11/2019 - Oscar: Image comparator class - defined all basic attributes and methods.
### 17/11/2019 - Oscar: Image comparator class - match method added.
### 18/11/2019 - Oscar: Image class - added sift as default feature detection model.
### 18/11/2019 - Oscar: ImageComparator class - match method updated with model selection.
### 19/11/2019 - Oscar: ImageComparator class - match method updated with knn methods.
### 19/11/2019 - Oscar: ImageComparator class - added orb algorithm for feature detection.
### 19/11/2019 - Oscar: ImageComparator class - Lowe score method added.
### 20/11/2019 - Oscar: ImageComparator class - match method modified.
### 20/11/2019 - Oscar: ImageComparator class - __ratio_test method modified.
### 20/11/2019 - Oscar: Error classes - custom exceptions added.
### 21/11/2019 - Oscar: ImageComparator class - knnmatch method modified.
###

### import Libraries ###
import numpy as np
import cv2
import matplotlib.pyplot as plt


### constants definition ###
LOWE_THRS = 0.7
DEFAULT_FIGSIZE = (10,15) # Default size for image plots.
DEFAULT_FEATURE_MODEL = 'sift'
FLANN_INDEX_KDTREE = 1
N_FLANN_TREES = 5
N_FLANN_CHECKS = 50
EDGE_THRS = 20 # SIFT edgeThreshold parameter
N_MATCHES_PLOT = 15
DEFAULT_N_FEATURES = 15000 # Default number of features for ORB algorithm)

### models definitions ###
sift = cv2.xfeatures2d.SIFT_create(edgeThreshold = EDGE_THRS)
surf = cv2.xfeatures2d.SURF_create(extended = True)
orb = cv2.ORB_create(nfeatures = DEFAULT_N_FEATURES)

### exceptions classes ###
class Error(Exception):
    """Base class for other exceptions"""
    pass

class NotFittedError(Error):
    """Raised when the Image has not fitted the keypoints"""
    pass

class NotMatchedError(Error):
    """Raised when the ImageComparator has not runned the match"""
    pass

### Image class ###

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

        keypoints_  : list
                      keypoints objects

        descriptors_    : list
                          descriptors objects


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
        elif model_name == 'orb':
            model = orb
        else:
            raise ValueError('The only implemented models are sift, surf and orb.')

        return model


    def find_keypoints(self, model_name = DEFAULT_FEATURE_MODEL):
        """
            Method to calculate keypoints and descriptors.

            The argument model_name is a string the name of the model.
            Default value is 'sift'.
            Admitted values are 'sift', 'surf' and 'orb'.

            returns the object itself with new attributes.
            keypoints is a list of keypoint objects.
            descriptors is a list of arrays encoding the features vector.
        """
        model = self.__model_selection(model_name)
        keypoints, descriptors = model.detectAndCompute(self.img_, None)

        self.keypoints_ = keypoints
        self.descriptors_ = descriptors

        return self


    def plotKeypoints(self, figsize = DEFAULT_FIGSIZE):
        """
            Method to plot the image with keypoints.

            figsize is a tuple tuning the plot size.
        """
        if not hasattr(self, 'keypoints_'):
            raise NotFittedError('Run find_keypoints before plotting')

        img_to_plot = cv2.drawKeypoints(self.__toGray(), self.keypoints_, self.img_)
        plt.figure(figsize=DEFAULT_FIGSIZE)
        plt.imshow(img_to_plot)

    def plotImage(self, figsize = DEFAULT_FIGSIZE):
        """
            Method to plot the image.

            figsize is a tuple tuning the plot size.
        """
        plt.figure(figsize=DEFAULT_FIGSIZE)
        plt.imshow(self.img_), plt.show()

### Image comparator class ###

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
        matcher_ :  object,
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
            self.match_model_ = self.__match_selection(matcher)

    @staticmethod
    def __match_selection(match_model):
        """
            Private method to select the matching scheme.

            For the moment being the only two admitted arguments are
        """
        if match_model == 'bf': # feature matching method - Brute Force
            matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
        elif match_model == 'flann': # feature matching method - Flann method
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = N_FLANN_TREES)
            search_params = dict(checks = N_FLANN_CHECKS)   # or pass empty dictionary

            matcher = cv2.FlannBasedMatcher(index_params, search_params)

        return matcher

    def match(self, Image_1, Image_2, model_name = DEFAULT_FEATURE_MODEL):
        """
            match method.

            It takes two objects of type Image as input.
            The model_name argument indicates the model to use to calculate images keypoints.
            It returns self object updated with a list of matches as attribute.
        """
        if hasattr(Image_1, 'keypoints_') and hasattr(Image_2, 'keypoints_'):
            matches = self.match_model_.match(Image_1.descriptors_,
                                              Image_2.descriptors_)
            matches = sorted(matches, key = lambda x: x.distance)

        else:
            Image_1.find_keypoints(model_name)
            Image_2.find_keypoints(model_name)

            matches = self.match_model_.match(Image_1.descriptors_,
                                              Image_2.descriptors_)
            matches = sorted(matches, key = lambda x: x.distance)


        self.matches_ = matches

        return self

    def knnmatch(self, Image_1, Image_2, model_name = DEFAULT_FEATURE_MODEL, k=2):
        """
            match method.

            It takes two objects of type Image as input.
            The model_name argument indicates the model to use to calculate images keypoints.
            k (default 2) argument is an int and indicates the number of classes to collect the matches.
            It returns the self object updated with the list of knnmatches as attribute.
        """
        # change crossCheck argument of bf matcher for knn method.
        if self.matcher_ == 'bf':
            self.match_model_ = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)

        if hasattr(Image_1, 'keypoints_') and hasattr(Image_2, 'keypoints_'):
            matches = self.match_model_.knnMatch(Image_1.descriptors_,
                                                 Image_2.descriptors_, k)
            matches = sorted(matches, key = lambda x: x[0].distance)

        else:
            Image_1.find_keypoints(model_name)
            Image_2.find_keypoints(model_name)

            matches = self.match_model_.knnMatch(Image_1.descriptors_,
                                                 Image_2.descriptors_, k)
            matches = sorted(matches, key = lambda x: x[0].distance)

        self.knnmatches_ = matches

        return self

    def score(self, threshold = LOWE_THRS):
        """
            Lowe score of similarity.
            Defined for knnmatch only.

            The argument threshold (default 0.7) indicates the limit splitting good/bad matches.
            It returns a float indicating the score of the match.

            Theoretically, we make use of the Lowe distance to calculate the score.
        """

        if not hasattr(self, 'knnmatches_'):
            raise NotMatchedError('Call knnmatch before calculating the score.')

        all_matches = self.knnmatches_

        return self.__ratio_test(all_matches, threshold, option = 'Score')

    def __ratio_test(self, matches, threshold, option):
        """
            Private method to calculate the ratio test and in Lowe's paper defining SIFT.

            It takes the list of couple of matches as one argument.
            As second argument the threshold value.
            As third argument the option value indicates whether we want the drawing mask,
            the good matches list or the score.
            option addmitted values: ['Mask', 'List', 'Score']

        """
        if not hasattr(self, 'knnmatches_'):
            raise NotMatchedError('Call knnmatch before calculating ratio test.')

        if option == 'Mask':
            matchesMask = [[0,0] for i in range(len(matches))]

            # ratio test as per Lowe's paper
            for i,(m,n) in enumerate(matches):
                if m.distance < threshold*n.distance:
                    matchesMask[i]=[1,0]

            return matchesMask

        elif option == 'List':
            good_matches = []

            # ratio test as per Lowe's paper
            for (m,n) in matches:
                if m.distance < threshold*n.distance:
                    good_matches.append(m)

            return good_matches

        elif option == 'Score':
            score = 0

            # ratio test as per Lowe's paper
            for (m,n) in matches:
                if m.distance < threshold*n.distance:
                    score +=1

            return score

    def __matches_to_plot_classic(self, n_matches):
        """
            Private method to choose how many matches to plot.

            It takes two optional arguments.
            The model_name argument indicates the model to use to calculate images keypoints.
            An int n_matches will give the number of matches.
            A float greater than 0 will give all the matches whose score is greater than the argument.
        """
        all_matches = self.matches_

        try:
            return all_matches[:n_matches]
        except ValueError:
            try:
                if (n_matches < 0):
                    raise ValueError('n_matches can only be an integer or a positive float')

                good_matches = [match for match in all_matches if match.distance >= n_matches]
                return good_matches
            except:
                raise ValueError('%s is not an integer nor a float' %n_matches)

    def __matches_to_plot_knn(self, n_matches):
        """
            Private method to choose how many matches to plot in case of knnmatch method.

            An int n_matches will give the number of matches.
            A float between 0 qnd 1 will give all the matches whose score is greater than the argument.

            returns a dictionary of drawing parameters.
        """

        if n_matches < 0:
            raise ValueError('n_matches cannot be negative.')
        elif n_matches < 1:
            all_matches = self.knnmatches_

            matchesMask = self.__ratio_test(all_matches, threshold = n_matches, option = 'Mask')

            draw_params = dict(matchColor = (0,255,0),
                               singlePointColor = (255,0,0),
                               matchesMask = matchesMask,
                               flags = cv2.DrawMatchesFlags_DEFAULT)
            return draw_params
        else:
            all_matches = self.knnmatches_

            matchesMask = self.__ratio_test(all_matches, threshold = LOWE_THRS, option = 'Mask')
            good_matches = self.__ratio_test(all_matches, threshold = LOWE_THRS, option = 'List')
            n = len(good_matches)

            draw_params = dict(matchColor = (0,255,0),
                               singlePointColor = (255,0,0),
                               matchesMask = matchesMask[:n],
                               flags = cv2.DrawMatchesFlags_DEFAULT)
            return draw_params

    def __matches_to_plot(self, n_matches = N_MATCHES_PLOT):
        """
            Private method to choose how many matches to plot.

            It takes one optional argument.

            An int n_matches will give the number of matches.
            A float between 0 qnd 1 will give all the matches whose score is greater than the argument.
        """
        if hasattr(self, 'knnmatches_'):
            return self.__matches_to_plot_knn(n_matches)
        elif hasattr(self, 'matches_'):
            return self.__matches_to_plot_classic(n_matches)
        else:
            raise NotMatchedError('Run a match method before plotting.')


    def plot_matching(self, figsize = DEFAULT_FIGSIZE,
                      n_matches = N_MATCHES_PLOT,
                      threshold = LOWE_THRS):
        """
            method to plot the images with feature matching.

            The model_name argument indicates the model to use to calculate images keypoints.
            figsize is a tuple tuning the plot size.
            n_matches can be int and will give the number of matches to show.
            n_matches can be a positive float and the function will show all the matches
            whose score is greater than the n_matches.
        """

        try:
            img_to_plot = cv2.drawMatches(Image_1.img_, keypoints_1,
                                          Image_2.img_, keypoints_2,
                                          self.__matches_to_plot(Image_1, Image_2, model_name, n_matches, threshold),
                                          Image_2.img_, flags=2)
        except:
            try:
                img_to_plot = cv2.drawMatches(Image_1.img_, Image_1.keypoints(model_name)[0],
                                              Image_2.img_, Image_2.keypoints(model_name)[0],
                                              self.__matches_to_plot(Image_1, Image_2, model_name, n_matches, threshold),
                                              Image_2.img_, flags=2)
            except:
                draw_params = self.__matches_to_plot_knn(Image_1, Image_2,
                                                         model_name,
                                                         threshold)
                try:
                    img_to_plot = cv2.drawMatchesKnn(Image_1.img_, keypoints_1,
                                                     Image_2.img_, keypoints_2,
                                                     self.__matches_to_plot(Image_1, Image_2, model_name, n_matches, threshold),
                                                     None, **draw_params)
                except:
                    img_to_plot = cv2.drawMatchesKnn(Image_1.img_, Image_1.keypoints(model_name)[0],
                                                     Image_2.img_, Image_2.keypoints(model_name)[0],
                                                     self.__matches_to_plot(Image_1, Image_2, model_name, n_matches, threshold),
                                                     None, **draw_params)

        plt.figure(figsize=figsize)
        plt.imshow(img_to_plot), plt.show()
