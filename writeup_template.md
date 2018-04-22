## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/hog_visualization.png
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in code cells 24,25 and 26 of the IPython notebook (#Jupyter Notebook "Project 5 - CarND - Advanced Vehicle Detection")

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=12`, `pixels_per_cell=18` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and settled on the below values which provided highest vehicle detection accuracy and less false positives. I used different color spaces and and tuned HOG parameters. Here are the parameters that worked best for me. (#Jupyter Notebook "Project 5 - CarND - Advanced Vehicle Detection", code block 26 and 27)

```python
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 12  # HOG orientations
pix_per_cell = 16 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [None, None] # Min and max in y to search in slide_window()
```


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the provided labeled data for vehicle and none-vehicle examples  (#Jupyter Notebook "Project 5 - CarND - Advanced Vehicle Detection", code block 28)

```python
# Use a linear SVC 
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)

```

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I used Multiple regions of varying scale for window search (Multi-Scale). Here is the approach i followed :

1. I restrict the search for the area in the image where cars might appear (function: find_cars jupyter notebook [49] )

  ```python
    img_tosearch = draw_img[ystart:ystop,600:draw_img.shape[1],:]
  ```
2. I used multiple search regions of varying scale (Jupyter Notebook block : [86]). Below is the code I used to identify the the different search regions I used

    # ystart  ystop   scale  step
    # 400     480     1      15
    # 400     530     1.5    30
    # 400     560     2.0    45
    # 400     660     2.5    60
    # 400     550     3.0    75
    
    
```python

class Scan_Region():
    '''
    scan region class define a unique scan region for sliding window 
    '''
    def __init__(self,_ystart,_ystop,_scale,_step):
        self.ystart=_ystart
        self.ystop=_ystop
        self.scale=_scale
        self.step=_step

#Jupyter Notebook block : [86]
scan_regions = []

scan_regions.append(Scan_Region(400,480,1,15))
scan_regions.append(Scan_Region(400,530,1.5,30))
scan_regions.append(Scan_Region(400,560,2.0,45))
scan_regions.append(Scan_Region(400,660,2.5,60))
scan_regions.append(Scan_Region(400,660,3,75))
scan_regions.append(Scan_Region(400,660,3.5,75))




def process_image_Ex(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    # TODO: Build your pipeline that will draw lane lines on the test_images
    global frame_count
    global vehicles_detected
    
    Vehicle_Detection=[]
    Vehicle_Detection_Flat=[]
    
    #ystart  ystop   scale  step
    #400     480     1      15
    #400     530     1.5    30
    #400     560     2.0    45
    #400     660     2.5    60
    #400     550     3.0    75
    
    
    for region in scan_regions:
        ystart=region.ystart
        ystop=region.ystop
        scale=region.scale
        step=region.step
        
        while ystart<ystop and ystop-ystart>step :
            out_img,box_list = find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
            Vehicle_Detection.append(box_list)
            ystart=ystart+step
            #print(ystart)
    



```

<img src="./sliding_search_animation.gif" width="800" alt="Combined Image" /> 


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. I used different color spaces and and tuned HOG parameters. Here are the parameters that worked best for me.

```python
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 12  # HOG orientations
pix_per_cell = 16 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [None, None] # Min and max in y to search in slide_window()
```


Here are some example images:

![alt text][image4]
---
https://youtu.be/eDbFndTHccI
### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_out.mp4)


##### CarND | Project 5 - CarND-Vehicle-Detection | Project Video

<a href="https://www.youtube.com/watch?v=eDbFndTHccI&feature=youtu.be" target="_blank"><img src="http://img.youtube.com/vi/eDbFndTHccI/0.jpg" 
alt="CarND-Vehicle-Detection | Project 5 Video" width="240" height="180" border="10" /></a>


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video using the find_cars() function. From the positive detections, I created a heat map for the current frame and stored the current heatmap in vision_memory class that keeps track of the heatmaps of the last n Frames. and then retrieved the residual heatmap for past n Frames using the vision_memory.get_residual_vision() function and finally thresholded that map to identify vehicle positions. I then used scipy.ndimage.measurements.label() to identify individual blobs in the heat map. I then assumed each blob corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

```python
#Jupyter Notebook "Project 5 - CarND - Advanced Vehicle Detection", block 86
def process_image_Ex(image):
'''
Main processing pipeline
'''
    global frame_count
    global vehicles_detected
    
    Vehicle_Detection=[]
    Vehicle_Detection_Flat=[]
    
    #Iterate through all the defined region and run sliding search using find_cars() 
    for region in scan_regions:
        ystart=region.ystart
        ystop=region.ystop
        scale=region.scale
        step=region.step
        
        while ystart<ystop and ystop-ystart>step :
            out_img,box_list = find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
            Vehicle_Detection.append(box_list)
            ystart=ystart+step
            
     
    #Flattening the list
    for sublist in Vehicle_Detection:
        for item in sublist:
                Vehicle_Detection_Flat.append(item)
    
    #initialize heatmap with the same size as img
    heatmap_img = np.zeros_like(img[:,:,0])
 
    #generate heatmap for current frame from detected bounding boxes
    heatmap_img = add_heat(heatmap_img, Vehicle_Detection_Flat)
   
    #Store current heatmap for in vision_memory queue
    vision_memory.capture_heatmap(heatmap_img)
    
    #get residual heatmap by summing all heatmaps for past n frames
    heatmap_img = vision_memory.get_residual_vision()
    
    #Filtering out transiet and false positive detection
    heatmap_img = apply_threshold(heatmap_img, 10)
    
    #use lable function to identify neighboring blobs 
    labels = label(heatmap_img)
    
    #finally draw the label bounding boxes for each detected label
    draw_img, rect = draw_labeled_bboxes(np.copy(image), labels)
    

    cv2.putText(draw_img, 'Frame : {:.0f} '.format(frame_count), (50, draw_image.shape[0]-50), font, 1, fontColor, 1)
    frame_count+=1
    return draw_img
 
```

The following class help with filtering false positives by storing heatmap for past n Frames and identifying false postives detection that spans only one or two frames.
    
```python
        ##Jupyter Notebook "Project 5 - CarND - Advanced Vehicle Detection", block 66
        class nVision():
            '''
            Residual Heatmap vision class help with filtering out false positives and duplicate detections.
            '''
            def __init__(self,n):
                self.heatmaps=[]
                self.memory=n

            def set_memory(self, n):
                self.memory=n

            def capture_heatmap(self, heatmap):
                self.heatmaps.append(heatmap)
                if len(self.heatmaps) > self.memory:
                    self.heatmaps = self.heatmaps[len(self.heatmaps)-self.memory:]

            def get_residual_vision(self):
                return np.asarray(self.heatmaps).sum(axis=0)
            
  ```    

### Here are sample frames and their corresponding heatmaps:

![alt text][image5]



### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Using SVM classifier and HOG based feature extraction provided satisfactory results for this project. However, this won't work well for real-time application due to the relatively low accuracy and the performance overhead introduced by the sliding window technique used in this pipeline. The current classifier used in my pipeline can still classify the none-car object as cars however this can be mitigated by the use of heat maps and calculating the residual heatmaps for past n frames and apply thresholds the removes false identifications. and possibly increase the accuracy by using more dataset for training the classifier. 

Generally, the following improvements can be made:
1. Further, tune the existing SVM classifier and train it with more datasets.
2. Use Convolutional Neural networks instead of SVM (i.e. YOLOv3 model)
3. Further Tune sliding window
3. Use perspective transform technique to measure relative distance and  sizes of neighboring cars
4. Tune sliding window algorithm to increase efficiency and decrease performance overhead.


[References]
1. Udacity Self-Driving Car Term1
2. Implementing queue : https://www.pythoncentral.io/use-queue-beginners-guide/
3. SVM : http://scikit-learn.org/stable/modules/svm.html
