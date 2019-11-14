###
### theet-recognition.py
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
### theet-recognition.py
### This is the script to make image comparison. It makes use of OsIm module.
###
### 08/11/2019 - Oscar: creation of repository and first commit.
### 10/11/2019 - Oscar: creation of modules and first version of the script.
### 14/11/2019 - Oscar: Istanciate images - script version 1.1.
###

### import Libraries ###
import OsIm
import glob # to have a list of images

### images paths
img_zero_path = 'path/to/image'
img_test_path = 'path/to/test'

### image objects istances
img_zero = OsIm.Image(img_zero_path)
img_test = OsIm.Image(img_test_path)

### plot images to verify
img_zero.plotImage()
img_test.plotImage()

### Model selection
model = 'sift'

### plot keypoints on images
img_zero.plotKeypoints(model_name = model)
img_test.plotKeypoints(model_name = model)
