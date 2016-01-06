#!/usr/bin/env python

""" Fire Detect and Track ...........inspired from ROS by examples scripts 
"""
import roslib; roslib.load_manifest('srproject')
import rospy
import cv2
from cv2 import cv as cv
from ros2opencv2 import ROS2OpenCV2
from std_msgs.msg import String
from sensor_msgs.msg import Image
import numpy as np

class FireMatchDetectTrack(ROS2OpenCV2):
    def __init__(self, node_name):
        ROS2OpenCV2.__init__(self, node_name)

        self.node_name = node_name
        

        self.match_threshold = rospy.get_param("~match_threshold", 0.7)
        self.find_multiple_targets = rospy.get_param("~find_multiple_targets", False)
        self.n_pyr = rospy.get_param("~n_pyr", 2)
        self.min_template_size = rospy.get_param("~min_template_size", 25)

        self.scale_factor = rospy.get_param("~scale_factor", 1.2)
        self.scale_and_rotate = rospy.get_param("~scale_and_rotate", False)
        
        self.use_depth_for_detection = rospy.get_param("~use_depth_for_detection", False)
        self.fov_width = rospy.get_param("~fov_width", 1.094)
        self.fov_height = rospy.get_param("~fov_height", 1.094)
        self.max_object_size = rospy.get_param("~max_object_size", 0.28)
        topicname1 = rospy.get_param("~topicname1", "/roi")

        # The minimum saturation of the tracked color in HSV space,
        # as well as the min and max value (the V in HSV) and a 
        # threshold on the backprojection probability image.
        self.smin = rospy.get_param("~smin", 85)
        self.vmin = rospy.get_param("~vmin", 50)
        self.vmax = rospy.get_param("~vmax", 254)
        self.threshold = rospy.get_param("~threshold", 50)
                       
        # Create a number of windows 
        cv.NamedWindow("Backproject_Fire_Window", 0)
        cv.MoveWindow("Backproject_Fire_Window", 300, 400)
        
    
        # Initialize a number of variables
        self.hist = None
        self.track_window = None
        self.show_backproj = False



        self.detector_loaded = False


    # The main processing function computes the histogram and backprojection
    def process_image(self, cv_image):

        # First blur the image
        frame = cv2.blur(cv_image, (5, 5))

        # Convert from RGB to HSV space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create a mask using the current saturation and value parameters
        mask = cv2.inRange(hsv, np.array((0., self.smin, self.vmin)), np.array((180., 255., self.vmax)))
       



        # STEP 1. Load a detector if one is specified
        if not self.detector_loaded and self.selection is None:
            """ Read in the template image """              
            template1 = rospy.get_param("~template1", "")
            self.detector_loaded = self.load_template_detector(template1)
            hsv_template1 = cv2.cvtColor(self.template, cv2.COLOR_BGR2HSV)
            #cv2.imshow("Template", hsv_template1)
            mask_template1 = cv2.inRange(hsv_template1, np.array((0., self.smin, self.vmin)), np.array((180., 255., self.vmax)))
            self.track_window = self.match_template(cv_image)
            self.hist = cv2.calcHist( [hsv_template1], [0], mask_template1, [16], [0, 180] )
            cv2.normalize(self.hist, self.hist, 0, 255, cv2.NORM_MINMAX);
            self.hist = self.hist.reshape(-1)



        # If the user is making a selection with the mouse, 
        # calculate a new histogram to track
        if self.selection is not None:
            x0, y0, w, h = self.selection
            x1 = x0 + w
            y1 = y0 + h
            self.track_window = (x0, y0, x1, y1)
            hsv_roi = hsv[y0:y1, x0:x1]
            mask_roi = mask[y0:y1, x0:x1]
            self.hist = cv2.calcHist( [hsv_roi], [0], mask_roi, [16], [0, 180] )
            cv2.normalize(self.hist, self.hist, 0, 255, cv2.NORM_MINMAX);
            self.hist = self.hist.reshape(-1)
            # self.show_hist()

        if self.detect_box is not None:
            self.selection = None


        
        # If we have a histogram, track it with CamShift
        if self.hist is not None:
            # Compute the backprojection from the histogram
            backproject = cv2.calcBackProject([hsv], [0], self.hist, [0, 180], 1)
            
            # Mask the backprojection with the mask created earlier
            backproject &= mask

            # Threshold the backprojection
            ret, backproject = cv2.threshold(backproject, self.threshold, 255, cv.CV_THRESH_TOZERO)

            x, y, w, h = self.track_window
            if self.track_window is None or w <= 0 or h <=0:
                self.track_window = 0, 0, self.frame_width - 1, self.frame_height - 1
            
            # Set the criteria for the CamShift algorithm
            term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
            
            # Run the CamShift algorithm
            self.track_box, self.track_window = cv2.CamShift(backproject, self.track_window, term_crit)

            # Display the resulting backprojection
            cv2.imshow("Backproject_Fire_Window", backproject)

        return cv_image
        

    def match_template(self, frame):
        H,W = frame.shape[0], frame.shape[1]
        h,w = self.template.shape[0], self.template.shape[1]

        # Make sure that the template image is smaller than the source
        if W < w or H < h:
            rospy.loginfo( "Template image must be smaller than video frame." )
            return None
        
        if frame.dtype != self.template.dtype: 
            rospy.loginfo("Template and video frame must have same depth and number of channels.")
            return None
        
        # Create a copy of the frame to modify
        frame_copy = frame.copy()
        
        for i in range(self.n_pyr):
            frame_copy = cv2.pyrDown(frame_copy)
            
        template_height, template_width  = self.template.shape[:2]
        
        # Cycle through all scales starting with the last successful scale

        scales = self.scales[self.last_scale:] + self.scales[:self.last_scale - 1]

        # Track which scale and rotation gives the best match
        maxScore = -1
        best_s = 1
        best_r = 0
        best_x = 0
        best_y = 0
        
        for s in self.scales:
            for r in self.rotations:
                # Scale the template by s
                template_copy = cv2.resize(self.template, (int(template_width * s), int(template_height * s)))

                # Rotate the template through r degrees
                rotation_matrix = cv2.getRotationMatrix2D((template_copy.shape[1]/2, template_copy.shape[0]/2), r, 1.0)
                template_copy = cv2.warpAffine(template_copy, rotation_matrix, (template_copy.shape[1], template_copy.shape[0]), borderMode=cv2.BORDER_REPLICATE)
    
                # Use pyrDown() n_pyr times on the scaled and rotated template
                for i in range(self.n_pyr):
                    template_copy = cv2.pyrDown(template_copy)
                
                # Create the results array to be used with matchTempate()
                h,w = template_copy.shape[:2]
                H,W = frame_copy.shape[:2]
                
                result_width = W - w + 1
                result_height = H - h + 1
                
                try:
                    result_mat = cv.CreateMat(result_height, result_width, cv.CV_32FC1)
                    result = np.array(result_mat, dtype = np.float32)
                except:
                    continue
                
                # Run matchTemplate() on the reduced images
                cv2.matchTemplate(frame_copy, template_copy, cv.CV_TM_CCOEFF_NORMED, result)
                
                # Find the maximum value on the result map
                (minValue, maxValue, minLoc, maxLoc) = cv2.minMaxLoc(result)
                
                if maxValue > maxScore:
                    maxScore = maxValue
                    best_x, best_y = maxLoc
                    best_s = s
                    best_r = r
                    best_template = template_copy.copy()
                    self.last_scale = self.scales.index(s)
                    best_result = result.copy()
                
        # Transform back to original image sizes
        best_x *= int(pow(2.0, self.n_pyr))
        best_y *= int(pow(2.0, self.n_pyr))
        h,w = self.template.shape[:2]
        h = int(h * best_s)
        w = int(w * best_s)
        best_result = cv2.resize(best_result, (int(pow(2.0, self.n_pyr)) * best_result.shape[1], int(pow(2.0, self.n_pyr)) * best_result.shape[0]))
        display_result = np.abs(best_result)**3

        # cv2.imshow("Result", display_result)
        best_template = cv2.resize(best_template, (int(pow(2.0, self.n_pyr)) * best_template.shape[1], int(pow(2.0, self.n_pyr)) * best_template.shape[0]))
        # cv2.imshow("Best Template", best_template)
        
        #match_box = ((best_x + w/2, best_y + h/2), (w, h), -best_r)
        return (best_x, best_y, w, h)



    def load_template_detector(self,template1):
            try:
                
                
                self.template = cv2.imread(template1, cv.CV_LOAD_IMAGE_COLOR)
                
                # cv2.imshow("Template", self.template)
                
                if self.scale_and_rotate:
                    """ Compute the min and max scales """
                    width_ratio = float(self.frame_size[0]) / self.template.shape[0]
                    height_ratio = float(self.frame_size[1]) / self.template.shape[1]
                    
                    max_scale = 0.9 * min(width_ratio, height_ratio)
                    
                    max_template_dimension = max(self.template.shape[0], self.template.shape[1])
                    min_scale = 1.1 * float(self.min_template_size) / max_template_dimension
                    
                    self.scales = list()
                    scale = min_scale
                    while scale < max_scale:
                        self.scales.append(scale)
                        scale *= self.scale_factor
                                        
                    self.rotations = [-45, 0, 45]
                else:
                    self.scales = [1]
                    self.rotations = [0]
                                        
                self.last_scale = 0 # index in self.scales
                self.last_rotation = 0
                
                #self.rotations = [0]
                
                print self.scales
                print self.rotations
                
                return True
            except:
                rospy.loginfo("Exception loading  detector!")
                return False

if __name__ == '__main__':
    try:
        node_name = "FireMatchDetectTrack"
        FireMatchDetectTrack(node_name)
        try:
            rospy.init_node(node_name)
        except:
            pass
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down vision node."
        cv.DestroyAllWindows()

