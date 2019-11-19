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
### 18/11/2019 - Oscar: image class - added sift as default feature detection model.
### 18/11/2019 - Oscar: image comparator class - match method updated with model selection.
### 19/11/2019 - Oscar: image comparator class - match method updated with knn methods.
### 19/11/2019 - Oscar: image comparator class - added orb algorithm for feature detection.
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
        elif model_name == 'orb':
            model = orb
        else:
            raise ValueError('The only implemented models are sift, surf and orb.')

        return model


    def keypoints(self, model_name = DEFAULT_FEATURE_MODEL):
        """
            Method to calculate keypoints and descriptors.

            The argument model_name is a string the name of the model.
            Default value is 'sift'.
            Admitted values are 'sift', 'surf' and 'orb'.

            returns a tuple of lists.
            keypoints is a list of keypoint objects.
            descriptors is a list of arrays encoding the features vector.
        """
        model = self.__model_selection(model_name)
        keypoints, descriptors = model.detectAndCompute(self.img_, None)
        return keypoints, descriptors


    def plotKeypoints(self, model_name = DEFAULT_FEATURE_MODEL, figsize = DEFAULT_FIGSIZE):
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

            matcher = cv2.FlannBasedMatcher(index_params, search_params)

        return matcher

    def match(self, Image_1, Image_2, model_name = DEFAULT_FEATURE_MODEL):
        """
            match method.

            It takes two objects of type Image as input.
            The model_name argument indicates the model to use to calculate images keypoints.
            It returns a list of match objects.
        """
        matches = self.match_model_.match(Image_1.keypoints(model_name)[1],
                                          Image_2.keypoints(model_name)[1])
        matches = sorted(matches, key = lambda x:x.distance)
        return matches
        
    def knnmatch(self, Image_1, Image_2, model_name = DEFAULT_FEATURE_MODEL, k=2):
        """
            match method.

            It takes two objects of type Image as input.
            The model_name argument indicates the model to use to calculate images keypoints.
            k (default 2) argument is an int and indicates the number of classes to collect the matches.
            It returns a list of couples of match objects.
        """
        matches = self.match_model_.knnMatch(Image_1.keypoints(model_name)[1],
                                          Image_2.keypoints(model_name)[1], k=2)
        matches = sorted(matches, key = lambda x:np.abs(x[0].distance - x[1].distance))
        
        return matches
    
    def score(self, Image_1, Image_2, model_name = DEFAULT_FEATURE_MODEL):
        """
            score of the similarity.

            It takes two objects of type Image as input.
            The model_name argument indicates the model to use to calculate images keypoints.
            It returns a float indicating the score of the match.

            Theoretically, we make use of the Abiyoyo distance to calculate the score.
        """

        return score
        
    def __matches_to_plot_classic(self, Image_1, Image_2,
                                  model_name = DEFAULT_FEATURE_MODEL,
                                  n_matches = N_MATCHES_PLOT):
        """
            Private method to choose how many matches to plot.

            It takes two optional arguments.
            The model_name argument indicates the model to use to calculate images keypoints.
            An int n_matches will give the number of matches.
            A float greater than 0 will give all the matches whose score is greater than the argument.
        """
        all_matches = self.match(Image_1, Image_2, model_name)

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
                
    def __matches_to_plot_knn(self, Image_1, Image_2,
                              model_name = DEFAULT_FEATURE_MODEL,
                              n_matches = None):
        """
            Private method to choose how many matches to plot.

            It makes use of knnmatch method.
            It takes two optional arguments.
            The model_name argument indicates the model to use to calculate images keypoints.
            An int n_matches will give the number of matches.
            A float greater than 0 will give all the matches whose score is greater than the argument.
            
            returns a dictionary of drawing parameters.
        """
        all_matches = self.knnmatch(Image_1, Image_2, model_name, k = 2)
           
        try:
            self.__ratio_test(all_matches, threshold)

            draw_params = dict(matchColor = (0,255,0),
                              singlePointColor = (255,0,0),
                              matchesMask = matchesMask[:n_matches],
                              flags = cv2.DrawMatchesFlags_DEFAULT)
        except:
            self.__ratio_test(all_matches)

            draw_params = dict(matchColor = (0,255,0),
                               singlePointColor = (255,0,0),
                               matchesMask = matchesMask,
                               flags = cv2.DrawMatchesFlags_DEFAULT)
            
        return draw_params
                
    def __matches_to_plot(self, Image_1, Image_2, knnmatch = True,
                          model_name = DEFAULT_FEATURE_MODEL,
                          n_matches = N_MATCHES_PLOT, threshold = LOWE_THRS):
        """
            Private method to choose how many matches to plot.

            It takes three optional arguments.
            knnmatch is a boolean argument (default True), and it is True if one wants to use Lowe's ratio test.
            The model_name argument indicates the model to use to calculate images keypoints.
            An int n_matches will give the number of matches.
            A float greater than 0 will give all the matches whose score is greater than the argument.
        """
        try:
            return self.__matches_to_plot_knn(Image_1, Image_2,
                                              model_name = DEFAULT_FEATURE_MODEL,
                                              n_matches = n_matches,
                                              threshold = LOWE_THRS)
        except:
            return self.__matches_to_plot_classic(Image_1, Image_2,
                                                  model_name = DEFAULT_FEATURE_MODEL,
                                                  n_matches = n_matches)
            

    def plot_matching(self, Image_1, Image_2,
                      keypoints_1 = None, keypoints_2 = None,
                      model_name = DEFAULT_FEATURE_MODEL,
                      figsize = DEFAULT_FIGSIZE,
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
            img_to_plot = cv2.drawMatches(Image_1.img_, Image_1.keypoints(model_name)[0],
                                          Image_2.img_, Image_2.keypoints(model_name)[0],
                                          self.__matches_to_plot(Image_1, Image_2, model_name, n_matches, threshold),
                                          Image_2.img_, flags=2)
        
        plt.figure(figsize=figsize)
        plt.imshow(img_to_plot), plt.show()
    
    def __ratio_test(self, matches, threshold)
        """
            Private method to calculate the ratio test and in Lowe's paper defining SIFT.
            
            It takes the list of couple of matches as one argument.
            As second argument the threshold value.
            
            It returns a list of lists, being the mask for plotting matches.
        """
        matchesMask = [[0,0] for i in range(len(matches))]
        
        # ratio test as per Lowe's paper
        for i,(m,n) in enumerate(matches):
            if m.distance < threshold*n.distance:
                matchesMask[i]=[1,0]
                
        return matchesMask
