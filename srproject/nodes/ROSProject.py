#!/usr/bin/env python
"""Main ROS Project """
#import some libraries

import rospy
import smach
from smach import State, StateMachine, Concurrence, Container, UserData
from smach_ros import MonitorState,SimpleActionState, IntrospectionServer
from geometry_msgs.msg import Twist
from rbx2_tasks.task_setup import *
#import easygui
import datetime
from collections import OrderedDict
from std_msgs.msg import String

import actionlib
from actionlib import GoalStatus
from geometry_msgs.msg import Pose, PoseWithCovarianceStamped,Point, Quaternion, Twist
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal, MoveBaseActionFeedback
from tf.transformations import quaternion_from_euler
from math import  pi

from actionlib_msgs.msg import *

#----------------------
import shlex, subprocess, os,signal
#----------------------
from sensor_msgs.msg import Image, RegionOfInterest, CameraInfo

# A list of Main Points and tasks
task_list = {'Point_A':['Navigate_Detect1'], 'Point_B':['Navigate_Detect2'], 'Point_C':['Navigate_Detect3'], 'Point_D':['Navigate_Detect1']}


def setup_task_environment(self):

    
    # How long do we have to get to each waypoint?
    self.move_base_timeout = rospy.get_param("~move_base_timeout", 10) #seconds
    
    # Initialize the patrol counter
    self.patrol_count = 0
    
    # Subscribe to the move_base action server
    self.move_base = actionlib.SimpleActionClient("move_base", MoveBaseAction)
    
    rospy.loginfo("Waiting for move_base action server...")
    
    # Wait up to 60 seconds for the action server to become available
    self.move_base.wait_for_server(rospy.Duration(60))    
    
    rospy.loginfo("Connected to move_base action server")
    
    # Create a list to hold the target quaternions (orientations)
    quaternions = list()
    
    # First define the corner orientations as Euler angles
    euler_angles = (pi/2 , 0 , pi , 3*pi/2)
    
    # Then convert the angles to quaternions
    for angle in euler_angles:
        q_angle = quaternion_from_euler(0, 0, angle, axes='sxyz')
        q = Quaternion(*q_angle)
        quaternions.append(q)
    
    # Create a list to hold the waypoint poses
    self.waypoints = list()
            
    # Append each of the four waypoints to the list.  Each waypoint
    # is a pose consisting of a position and orientation in the map frame.

     #   locations['Point1_GoalA'] = Pose(Point(0.685, 1.698, 0.000), Quaternion(0.000, 0.000, 0.000, 1.000))
     #   locations['Point2_GoalB'] = Pose(Point(-1.145, 2.028, 0.000), Quaternion(0.000, 0.000, 0.000, 1.000))
     #   locations['Point3_GoalC'] = Pose(Point(-2.740, -0.241, 0.000), Quaternion(0.000, 0.000, 0.000, 1.000))
     #   locations['Point4_GoalD'] = Pose(Point(-0.467, -0.373, 0.000), Quaternion(0.000, 0.000, 0.000, 1.000))

    self.waypoints.append(Pose(Point(0.685, 1.698, 0.000), quaternions[0]))
    self.waypoints.append(Pose(Point(-1.145, 2.028, 0.000), quaternions[1]))
    self.waypoints.append(Pose(Point(-2.940, 1.000, 0.000), quaternions[2]))
    self.waypoints.append(Pose(Point(-0.467, -0.150, 0.000), quaternions[3]))
    
    # Create a mapping of room names to waypoint locations
    points_locations = (('Point_A', self.waypoints[0]),
                        ('Point_B', self.waypoints[3]),
                        ('Point_C', self.waypoints[2]),
                        ('Point_D', self.waypoints[1]))
    
    # Store the mapping as an ordered dictionary so we can visit the rooms in sequence
    self.points_locations = OrderedDict(points_locations)
    

    # Publisher to manually control the robot (e.g. to stop it)
    self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist)
    rospy.loginfo("Starting Tasks")

class FireDetection(State):
    def __init__(self):
        State.__init__(self, outcomes=['succeeded','aborted','preempted'])
        self.task = 'Fire_Escap'
        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist)
        self.timer = 300 

    def execute(self, userdata):
        rospy.loginfo('Detecting Fire............. ')

        self.FireDetect = subprocess.Popen('roslaunch srproject FireMatechDetectTrack.launch',shell =True, preexec_fn =os.setsid)

        counter = self.timer
        while counter > 0:
            if self.preempt_requested():
                self.service_preempt()
                break;
            counter -= 1
            rospy.sleep(1)

        message = "Finished Fetecting Fire.............!"
        rospy.loginfo(message)
        #easygui.msgbox(message, title="Succeeded")
        os.killpg(self.FireDetect.pid, signal.SIGTERM)
        rospy.sleep(3)
        #update_task_list(self.Point_Location, self.task)
    
        return 'succeeded'


class FireEscap(State):
    def __init__(self):
        State.__init__(self, outcomes=['FireDetect','succeeded','aborted','preempted'])
        
        self.task = 'Fire_Escap'
        self.timer = 40 
        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist)

    def execute(self, userdata):
        rospy.loginfo('Escaping Fire............. ')

        self.FireEscaping = subprocess.Popen('roslaunch srproject fire_escaper.launch',shell =True, preexec_fn =os.setsid)
     
        counter = self.timer
        while counter > 0:
            if self.preempt_requested():
                self.service_preempt()
                return 'preempted'
            counter -= 1
            rospy.sleep(1)

        message = "Finished Escaping Fire.............!"
        rospy.loginfo(message)
        #easygui.msgbox(message, title="Succeeded")
        os.killpg(self.FireEscaping.pid, signal.SIGTERM)
        rospy.sleep(3)
        #update_task_list(self.Point_Location, self.task)
    
        return 'preempted'

class MatchDetectTrack(State):
    def __init__(self, Point_Location, timer):
        State.__init__(self, outcomes=['succeeded','aborted','preempted'])
        
        self.task = 'Navigate_Detect1'
        self.Point_Location = Point_Location
        self.timer = timer
        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist)

    def execute(self, userdata):
        rospy.loginfo('Detecting and following an Object ' + str(self.Point_Location))

        self.DetectionProcess = subprocess.Popen('roslaunch srproject ObjectMatchDetectTrack.launch',shell =True, preexec_fn =os.setsid)
        self.FollowingProcess = subprocess.Popen('roslaunch srproject object_follower.launch',shell =True, preexec_fn =os.setsid)

        counter = self.timer
        while counter > 0:
            if self.preempt_requested():
                self.service_preempt()
                return 'preempted'
            counter -= 1
            rospy.sleep(1)

        message = "Finished Detecting an Object  the " + str(self.Point_Location) + "!"
        rospy.loginfo(message)
        #easygui.msgbox(message, title="Succeeded")
        os.killpg(self.FollowingProcess.pid, signal.SIGTERM)
        os.killpg(self.DetectionProcess.pid, signal.SIGTERM)
        rospy.sleep(3)
        #update_task_list(self.Point_Location, self.task)
	
        return 'succeeded'



class ObjectRecognizition(State):
    def __init__(self, Point_Location, timer):
        State.__init__(self, outcomes=['Object1Detected','Object2Detected','aborted','preempted'])
        
        self.task = 'Navigate_Detect2'
        self.Point_Location = Point_Location
        self.timer = timer
        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist)
        self.Object_sub = rospy.Subscriber("/ObjectDetected", String, self.Object_callback)
        self.ObjectDetected = "aborted"
   
    def execute(self, userdata):
        rospy.loginfo('Detecting Cone or Cylinder' + str(self.Point_Location))

        self.ObjectRecognize = subprocess.Popen('roslaunch srproject ObjectRecognizition.launch',shell =True, preexec_fn =os.setsid)

        counter = self.timer
        while counter > 0:
            if self.preempt_requested():
                self.service_preempt()
                return 'preempted'
            counter -= 1
            rospy.sleep(1)

        message = "Finished Detecting Cone or Cylinder the " + str(self.Point_Location) + "!"
        rospy.loginfo(message)
        rospy.loginfo(self.ObjectDetected)
        #easygui.msgbox(message, title="Succeeded")
        os.killpg(self.ObjectRecognize.pid, signal.SIGTERM)
        rospy.sleep(3)

        return self.ObjectDetected
    
    def Object_callback(self, data):
        self.ObjectDetected = data.data

class DetectTag(State):
    def __init__(self, Point_Location, timer):
        State.__init__(self, outcomes=['succeeded','aborted','preempted'])
        
        self.task = 'Navigate_Detect3'
        self.Point_Location = Point_Location
        self.timer = timer
        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist)

    def execute(self, userdata):
        rospy.loginfo('Detecting Detecting Tag' + str(self.Point_Location))

        self.DetectTagg = subprocess.Popen('roslaunch srproject ar_indiv_kinect.launch',shell =True, preexec_fn =os.setsid)

        counter = self.timer
        while counter > 0:
            if self.preempt_requested():
                self.service_preempt()
                return 'preempted'
            counter -= 1
            rospy.sleep(1)

        message = "Finished Detecting Tag " + str(self.Point_Location) + "!"
        rospy.loginfo(message)
        #easygui.msgbox(message, title="Succeeded")
        os.killpg(self.DetectTagg.pid, signal.SIGTERM)
        rospy.sleep(3)

        return 'succeeded'
        
        

def update_task_list(Point_Location, task):
    task_list[Point_Location].remove(task)
    if len(task_list[Point_Location]) == 0:
        del task_list[Point_Location]
        
    
  
class main():
    def __init__(self):
        rospy.init_node('Top_Project', anonymous=False)
        
        # Set the shutdown function (stop the robot)
        rospy.on_shutdown(self.shutdown)
        
        # Initialize a number of parameters and variables
        setup_task_environment(self)
        
        self.last_nav_state = None

        # A variable to hold the initial pose of the robot to be set by 
        # the user in RViz
        initial_pose = PoseWithCovarianceStamped()     
        # Get the initial pose from the user
        rospy.loginfo("*** Click the 2D Pose Estimate button in RViz to set the robot's initial pose...")
        rospy.wait_for_message('initialpose', PoseWithCovarianceStamped)
        self.last_location = Pose()
        rospy.Subscriber('initialpose', PoseWithCovarianceStamped, self.update_initial_pose)
        

        


        # Make sure we have the initial pose
        while initial_pose.header.stamp == "":
            rospy.sleep(1)
            
        rospy.loginfo("Starting navigation test")
        
        
        # Turn the locations into SMACH move_base action states
        nav_states = {}
        
        for Points in self.points_locations.iterkeys():         
            nav_goal = MoveBaseGoal()
            nav_goal.target_pose.header.frame_id = 'map'
            nav_goal.target_pose.pose = self.points_locations[Points]
            move_base_state = SimpleActionState('move_base', MoveBaseAction, goal=nav_goal, result_cb=self.move_base_result_cb, 
                                                exec_timeout=rospy.Duration(60.0),
                                                server_wait_timeout=rospy.Duration(10.0))
            nav_states[Points] = move_base_state

        ''' Create individual state machines for assigning tasks to each Point '''

        # Create a state machine for the point_A subtask(s)
        sm_Point_A = StateMachine(outcomes=['succeeded','aborted','preempted'])
        
        # Then add the subtask(s)
        with sm_Point_A:
            StateMachine.add('Navigate_Detect1', MatchDetectTrack('Point_A', 40), transitions={'succeeded':'','aborted':'','preempted':''})

        # Create a state machine for the point_B subtask(s)
        sm_Point_B = StateMachine(outcomes=['Object1Detected','Object2Detected','succeeded','aborted','preempted'])
        
        # Then add the subtask(s)
        with sm_Point_B:
            StateMachine.add('Navigate_Detect2', ObjectRecognizition('Point_B', 6), transitions={'Object1Detected':'','Object2Detected':'','aborted':'','preempted':''})

        # Create a state machine for the point_C subtask(s)
        sm_Point_C = StateMachine(outcomes=['Object1Detected','Object2Detected','succeeded','aborted','preempted'])
        
        # Then add the subtasks
        with sm_Point_C:
            StateMachine.add('Navigate_Detect3', ObjectRecognizition('Point_C', 6), transitions={'Object1Detected':'','Object2Detected':'','aborted':'','preempted':''})

        # Create a state machine for the point_D subtask(s)
        sm_Point_D = StateMachine(outcomes=['succeeded','aborted','preempted'])
        
        # Then add the subtasks
        with sm_Point_D:
            StateMachine.add('Navigate_Detect3', DetectTag('Point_D', 10), transitions={'succeeded':'Navigate_Detect1','aborted':'Navigate_Detect1','preempted':'Navigate_Detect1'})
            StateMachine.add('Navigate_Detect1', MatchDetectTrack('Point_D', 40), transitions={'succeeded':'','aborted':'','preempted':''})

        # Initialize the overall state machine
        self.sm_Main_Project = StateMachine(outcomes=['Object1Detected','Object2Detected','succeeded','aborted','preempted'])
            
        # Build the whole project state machine from the nav states and points states
        with self.sm_Main_Project:            
            StateMachine.add('START', nav_states['Point_D'], transitions={'succeeded':'Point_A','aborted':'Point_A'})
            
            ''' Add the Point_A subtask(s) '''
            StateMachine.add('Point_A', nav_states['Point_A'], transitions={'succeeded':'Point_A_TASKS','aborted':'Point_B'})
            
            # When the tasks are done, continue on to the Point_B
            StateMachine.add('Point_A_TASKS', sm_Point_A, transitions={'succeeded':'Point_B','aborted':'Point_B'})
            
            ''' Add the Point_B subtask(s) '''
            StateMachine.add('Point_B', nav_states['Point_B'], transitions={'succeeded':'Point_B_TASKS','aborted':'Point_C'})
            
            # When the tasks are done, continue on to the Point_C
            StateMachine.add('Point_B_TASKS', sm_Point_B, transitions={'Object2Detected':'Point_C','Object1Detected':'Point_A','aborted':'Point_C'})
            
            ''' Add the Point_C subtask(s) '''
            StateMachine.add('Point_C', nav_states['Point_C'], transitions={'succeeded':'Point_C_TASKS','aborted':'Point_D'})
            
            # When the tasks are done, return to the Point_D
            StateMachine.add('Point_C_TASKS', sm_Point_C, transitions={'Object1Detected':'Point_B','Object2Detected':'Point_D','aborted':'Point_D'})         
            
            ''' Add the hallway subtask(s) '''
            StateMachine.add('Point_D', nav_states['Point_D'], transitions={'succeeded':'Point_D_TASKS','aborted':''})
            
            # When the tasks are done, stop
            StateMachine.add('Point_D_TASKS', sm_Point_D, transitions={'succeeded':'','aborted':''})   

        #-------------------------------------------------------------
        # Register a callback function to fire on state transitions within the sm_nav state machine
        self.sm_Main_Project.register_transition_cb(self.nav_transition_cb, cb_args=[])

        # Create  Concurrence container
        sm_con_Project = Concurrence(outcomes=['FireDetect','succeeded', 'aborted','preempted'],
                                        default_outcome='succeeded',
                                        child_termination_cb=self.concurrence_child_termination_cb,
                                        outcome_cb=self.concurrence_outcome_cb)
        
        # Add the sm_nav machine and a battery MonitorState to the nav_patrol machine             
        with sm_con_Project:
           Concurrence.add('Main_Project', self.sm_Main_Project)
           Concurrence.add('MONITOR_Fire_ROI', MonitorState("/Fire_ROI", RegionOfInterest, self.FireEscape_cb))

        # Create a state machine for the point_C subtask(s)
        sm_FireEscap = StateMachine(outcomes=['FireDetect','succeeded','aborted','preempted'])
        
        # Then add the subtasks
        with sm_FireEscap:
            StateMachine.add('Fire_Escap', FireEscap(), transitions={'FireDetect':'','preempted':''})


        sm_TopX1_Project = StateMachine(outcomes=['succeeded','aborted','preempted','FireDetect']) 
        with sm_TopX1_Project:            
            StateMachine.add('con_Project', sm_con_Project, transitions={'succeeded':'','aborted':'', 'FireDetect':'FireEscap'})
            StateMachine.add('FireEscap', sm_FireEscap, transitions={'succeeded':'','aborted':'','FireDetect':''})
        
        sm_Top_Project = Concurrence(outcomes=['FireDetect','Object1Detected','Object2Detected','succeeded','aborted','preempted'],
                                        default_outcome='succeeded',
                                        child_termination_cb=self.concurrence_child_termination_cb1,
                                        outcome_cb=self.concurrence_outcome_cb1)
        # Add the sm_nav machine and a battery MonitorState to the nav_patrol machine             
        with sm_Top_Project:
           Concurrence.add('TopX1_Project', sm_TopX1_Project)
           Concurrence.add('Fire_Detection', FireDetection())
                        
        # Create and start the SMACH introspection server
        intro_server = IntrospectionServer('Top_Project', sm_Top_Project, '/SM_ROOT')
        intro_server.start()
        
        # Execute the state machine
        sm_outcome = sm_Top_Project.execute()

        #--------------------------------------------------------------
                
        # if len(task_list) > 0:
        #     message = "Ooops! Not all locations were completed."
        #     message += "The following Points need to be revisited: "
        #     message += str(task_list)
        # else:
        #     message = "All locations complete!"
            
        #rospy.loginfo(message)
        #easygui.msgbox(message, title="Finished tasks")
        
        intro_server.stop()
#-----------------------------------------------------------
    # Gets called when ANY child state terminates
    def concurrence_child_termination_cb1(self, outcome_map):
        return True
                # Gets called when ALL child states are terminated
    def concurrence_outcome_cb1(self, outcome_map):
        # If the battery is below threshold, return the 'recharge' outcome
        return True



    # Gets called when ANY child state terminates
    def concurrence_child_termination_cb(self, outcome_map):
        # If the current navigation task has succeeded, return True
        if outcome_map['Main_Project'] == 'succeeded':
            return True
        # If the MonitorState state returns False (invalid), store the current nav go
        if outcome_map['MONITOR_Fire_ROI'] == 'invalid':
            rospy.loginfo("Fire! Need TO Escape...")
            if self.last_nav_state is not None:
                self.sm_Main_Project.set_initial_state(self.last_nav_state, UserData())
            return True
        else:
            return False
    
    # Gets called when ALL child states are terminated
    def concurrence_outcome_cb(self, outcome_map):
        # If the battery is below threshold, return the 'recharge' outcome
        if outcome_map['MONITOR_Fire_ROI'] == 'invalid':
            return 'FireDetect'

        
    def FireEscape_cb(self, userdata, msg):
        counter = 5
        counter1=0
        while counter > 0:
            if msg.width <= 30 and msg.height <= 30:
                counter1 +=1
            counter -= 1
            rospy.sleep(1)

        if counter1==5:
            #rospy.loginfo("Hello1....")
            return True
        else:
            #rospy.loginfo("Hello2....")
            return False

    def nav_transition_cb(self, userdata, active_states, *cb_args):
        self.last_nav_state = active_states
#------------------------------------------------------------------
    def update_initial_pose(self, initial_pose):
        self.initial_pose = initial_pose
            
    def move_base_result_cb(self, userdata, status, result):
        if status == actionlib.GoalStatus.SUCCEEDED:
            pass
            
    def cleaning_task_cb(self, userdata):
        rooms_to_clean.remove(userdata.Point_Location)
    
    def shutdown(self):
        os.killpg(self.FireDetect.pid, signal.SIGTERM)
        rospy.loginfo("Stopping the robot...")
        #sm_nav.request_preempt()
        self.cmd_vel_pub.publish(Twist())
        rospy.sleep(1)
                


        
if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo("The Main Project test finished.")

