# Teeth Recognition

Repository to collect the code used to implement an image feature comparison.

## Introduction and motivations

The main goal is to apply such algorithm, firstly implemented to compare generic images features, to recognise teeth characteristics. This would have several important applications of high importance, e.g. the identification of people from the dental features. 

The main idea is to create a library collecting methods from `openCV`, to make comparison between images to identify subjects up to a certain level of confidence.

A motivational example might be the following: imagine the case of a [mass disaster](https://www.sciencedirect.com/topics/social-sciences/mass-disaster). 
In such an event it is crucial the precise and rapid identification of victims, a [process](http://www.weareforensic.co.uk/mass-disasters-disaster-victim-identification/) that can be quite complicate.
An image comparison system might help the forensic odontologist to compare hundreds of radiographs and find the one with highest similarity score with the given one.

![title](http://www.weareforensic.co.uk/wp-content/uploads/2014/07/WTC_911-620x300.jpg)
*An image from one of the most famous mass disaster in hystory: $11^\mathrm{th}$ September 2001.*

### Shape recognition in images

### Mass disaster and other applications

## Scale-Invariant Feature Transform

The aim of this section is to answer the question 

> what is SIFT? 

Well, SIFT -- which stands for _Scale Invariant Feature Transform_ -- is a method for extracting feature vectors that describe local patches of an image. Not only are these feature vectors scale-invariant, but they are also invariant to translation, rotation, and illumination. In other words: everything a descriptor should be!

These descriptors are useful for matching objects are patches between images. For example, consider creating a panorama. Assuming each image has some overlapping parts, you need some way to align them so we can stitch them together. If we have some points in each image that we know correspond, we can warp one of the images using a homography. SIFT helps with automatically finding not only corresponding points in each image, but points that are easy to match.
![title](https://miro.medium.com/max/2448/1*pwOhfFcv28p90fgwWhBzzw.png)

![title](https://miro.medium.com/max/2672/1*5LTZZiJIGUNnPXr_EGm_nQ.png)


### The optimised version: Speeded Up Robust Feature

## Getting dirty: examples

### Easy task: Compare two coins

### A not-so-easy task: Compare two radiographs

## Metric: a score to quantify how similar are two images

## Conclusions and further developments

### Acknowledgements

I would like to acknowledge both [`openCV`](https://opencv.org/) and [`menpo`](https://www.menpo.org/) projects.
SIFT algorithm was patented in Canada by the [University of British Columbia](https://patents.google.com/patent/US6711293) and published by David Lowe in 1999.
