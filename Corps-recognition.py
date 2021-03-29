###
### Corps-recognition.py
###
### Created by Oscar de Felice on 21/11/2019.
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
### Corps-recognition.py
### This is the script to find the most similar image to a given one.
### It makes use of OsIm module.
###
### 21/11/2019 - Oscar: Creation of this script.
### 22/11/2019 - Oscar: Instantiate Image objects.
###
###

### Import libraries ###
import OsIm
import cv2
import glob

### Model selection
model = 'surf'
model_compare = 'bf'

### Constants of the script
images_dir = 'path/to/images_dir'
img_zero_path = 'path/to/img_zero'

### List of image paths
images_path = glob.glob(images_dir)

### Instantiate Image objects
image_zero = OsIm.Image(img_zero_path) # Training image

images = {}
for i, image in enumerate(images_path):
    images[i] = OsIm.Image(image) # Test images

### Find keypoints using the model selected
image_zero.find_keypoints(model_name = model)

for image in images.values():
    image.find_keypoints(model_name = model)

### Comparator object istance
comparator = OsIm.ImageComparator(matcher = model_compare)

### Compute matches and scores
scores = []
for image in images.values():
    comparator.knnmatch(image_zero, image, model_name = model)
    scores.append(comparator.score())

### Find the maximum score
maximum_index, maximum_score = max(enumerate(scores), key=lambda x: x[1])

print('We found an image with score %.1f' %maximum_score)

### Plot the image with maximum similarity score
comparator.plot_matching(image_zero, images[maximum_index])

cv2.waitKey(0)
cv2.destroyAllWindows()
