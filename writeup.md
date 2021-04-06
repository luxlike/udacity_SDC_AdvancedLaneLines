# Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # "Image References"

[image_org]: ./camera_cal/calibration2.jpg "Original"
[image_cal]: ./output_images/calibration2.jpg "Calibrated"
[image0]: ./test_images/test1.jpg  "Original"
[image1]: ./output_images/undistorted_test1.jpg "Undistorted"
[image2]: ./output_images/thresholed_binary_test1.jpg "Thresholed binary"
[image3]: ./output_images/birdseye_test1.jpg "Birds-eye View"
[image4]: ./output_images/fit_polynomial_sliding_window_test1.jpg "Fit Polynomial"
[image5]: ./output_images/laneline_weighted_test1.jpg "Output"
[video1]: ./project_video.mp4 "Video"

### Camera Calibration

I use  `cv2.findChessboardCorners` method to get corners and add it to object points and image point. I calibrate camera by  `cv2.calibrateCamera` with those object and image points.
Undistorted image can get by  `cv2.undistort` method.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image_org]    ![alt text][image_cal]

### Pipeline (single images)

#### 1. Distortion-corrected image.

I apply the distortion correction to one of the test images like this:

```python
img_size = (img.shape[1], img.shape[0])
    
# Do camera calibration given object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)

undist = cv2.undistort(img, mtx, dist, None, mtx)
```

![alt text][image0]    ![alt text][image1]

#### 2. Color transforms, gradients,  thresholded binary image.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `combined_binary_img` method).  Here's an example of my output for this step.

![alt text][image0]    ![alt text][image2]

#### 3. Performed a perspective transform and example of a transformed image.

The code for my perspective transform includes a function called `warp()`

```python
img_size = (undist_img.shape[1], undist_img.shape[0])   
    
M = cv2.getPerspectiveTransform(src, dst) 
Minv = cv2.getPerspectiveTransform(dst, src)
warp = cv2.warpPerspective(undist_img, M, img_size, flags=cv2.INTER_LINEAR)
```

The `warp()` function takes as inputs an image (`undist_img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src_pts = np.float32([
                        [180, 700], # bottom-left 
                        [590, 460],  # top-left
                        [750, 460],  # top-right
                        [1170,700]  # bottom-right                        
                        ])
    
dst_pts = np.float32([
                        [210, 720], # bottom-left 
                        [210, 0],    # top-left
                        [1100, 0],   # top-right
                        [1100, 720]  # bottom-right                        
                        ])
```

This resulted in the following source and destination points:

|  Source   | Destination |
| :-------: | :---------: |
| 180, 700  |  210, 720   |
| 590, 460  |   210, 0    |
| 750, 460  |   1100, 0   |
| 1170, 700 |  1100, 720  |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image3]

#### 4. Identified lane-line pixels and fit their positions with a polynomial

To identify lane-line, I use `histogram` of binary_image to get left_x,right_x mid_point, then apply sliding window method to extract left and right line pixel positions.
With `np.polyfit()` fit a second order polynomial of lane pixels
Then, I can get line pixel positions with `fit_poly()` method like this:

```python
### Fit a second order polynomial to each with np.polyfit() ###
left_fit = np.polyfit(lefty,leftx,2)
right_fit = np.polyfit(righty,rightx,2)
# Generate x and y values for plotting
ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
### Calc both polynomials using ploty, left_fit and right_fit ###
left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
```
![alt text][image4]

#### 5. Calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

To calculate radius of curvature, I use polynomial function.
And to adjust to real world, multiply coefficient of x,y per meter
I did get radius of curvature like this:

```python
fit_cr = np.polyfit(self.ally*self.ym_per_pix, self.allx*self.xm_per_pix, 2)
        
# Define y-value where we want radius of curvature
# We'll choose the maximum y-value, corresponding to the bottom of the image
y_eval = np.max(self.ally)

# Implement the calculation of R_curve (radius of curvature) 
curvrad = ((1 + (2*fit_cr[0]*y_eval*self.ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
```
To calculate distance from center, get image center and lane center.
And subtract its like this:

```python
image = binary_warped_image    
image_center = image.shape[1] // 2
lane_center = self.allx[-1]
center_offset = (image_center - lane_center) * self.xm_per_pix
```

#### 6. Example image of my result plotted back down onto the road 

I warp the blank back to original image space using inverse perspective matrix (Minv)

```python
# Create an image to draw the lines on
warp_zero = np.zeros_like(thresholded_warped_image).astype(np.uint8)        
color_warp = np.dstack((warp_zero, warp_zero, warp_zero))       

# Recast the x and y points into usable format for cv2.fillPoly()
pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
pts = np.hstack((pts_left, pts_right))

# Draw the lane onto the warped blank image
cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

# Warp the blank back to original image space using inverse perspective matrix (Minv)
newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 
# Combine the result with the original image

result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
```

![alt text][image5]

---

### Pipeline (video)

I implemented `LaneLine` class to process pipe line
And the pipe line process like this:

  1.calibrate camera\
  2.undistorted image\
  3.wraped image\
  4.threshold wraped image\
  5.search each left,right line by blind and found method\
  6.calcurate left,right radius of curvature\
  7.calcurate distance from center\
  8.draw the lane and put text\

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

When I implement pipe line with LaneLine class, it didn't  work like upper implemented method.
The reason was that the use of the init variable and the class method did not match.
After long time debugging, I could solve problem.
Right now, my pipe line not robust in challenge video.
I think that I need to improve my search method (blind and found) to robust in find lane line.
Maybe add check sanity will be helpful.
