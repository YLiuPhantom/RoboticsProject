import rclpy
import numpy as np
import tf2_ros

from math               import pi, sin, cos, acos, atan2, sqrt, fmod, exp

from asyncio            import Future
from rclpy.node         import Node
from rclpy.qos          import QoSProfile, DurabilityPolicy
from geometry_msgs.msg  import PoseStamped, TwistStamped, Point, Vector3, Quaternion
from geometry_msgs.msg  import TransformStamped
from sensor_msgs.msg    import JointState
from std_msgs.msg       import Header, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

# Grab the Utilities
from utils.TransformHelpers     import *
from utils.TrajectoryUtils      import *

# Grab the general fkin from HW5 P5.
from hw5code.KinematicChain     import KinematicChain

#
#   Trajectory Generator Node Class
#
#   This inherits all the standard ROS node stuff, but adds an
#   update() method to be called regularly by an internal timer and a
#   shutdown method to stop the timer.
#
#   Arguments are the node name and a future object (to force a shutdown).


class TrajectoryNode(Node):
    # Initialization.
    def __init__(self, name, future):
        # Initialize the node and store the future object (to end).
        super().__init__(name)
        self.future = future

        ##############################################################
        # INITIALIZE YOUR TRAJECTORY DATA!

        # Define the list of joint names MATCHING THE JOINT NAMES IN THE URDF!
        # The chain from base_footprint to virtual_endeffector has 7 active DOFs
        self.jointnames=['shoulder_roll_joint', 'shoulder_pitch_joint', 'shoulder_yaw_joint', \
'elbow_pitch_joint', 'elbow_yaw_joint', 'wrist_pitch_joint', 'wrist_roll_joint']

        # Set up the kinematic chain object.
        self.chain = KinematicChain(self, 'world', 'tip', self.jointnames)

        # Define the matching initial joint/task positions.
        # 7 values for the 7 active DOFs (fixed joints don't need values)
        self.q0 = np.radians(np.array([180.0, 60.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        (self.p0, self.R0, _, _) = self.chain.fkin(self.q0)
        self.p = self.p0.copy()

        # Initialize the stored joint command position and task errors.
        self.qc = self.q0.copy()
        self.qcdot = vzero()
        self.ep = vzero()
        self.eR = vzero()
        
        # initialize position and time trackers for return journey
        self.qc_swing_end = vzero()
        self.time_swing_end = 0.0
        self.return_to_start = False

        # lambda and gamme
        self.lam = 20
        self.gamma = 0.1
        
        # Ball physics setup
        self.ball_radius = 0.03
        x_bias = 0.1*(np.random.rand()-0.5)
        y_bias = 0.1*(np.random.rand()-0.5)
        z_bias = 0.1*(np.random.rand()-0.5)
        self.ball_p = np.array([0.20 + x_bias, 0.45 + y_bias, self.ball_radius+0.2 + z_bias])
        self.ball_v = np.array([0.0, 0.0, 0.0])
        self.ball_a = np.array([0.0, 0.0, 0.0])
        self.ball_p0 = self.ball_p.copy()
        self.ball_v0 = self.ball_v.copy()
        
        # Ball marker setup
        diam = 2 * self.ball_radius
        self.ball_marker = Marker()
        self.ball_marker.header.frame_id = "world"
        self.ball_marker.header.stamp = self.get_clock().now().to_msg()
        self.ball_marker.action = Marker.ADD
        self.ball_marker.ns = "ball"
        self.ball_marker.id = 1
        self.ball_marker.type = Marker.SPHERE
        self.ball_marker.pose.orientation = Quaternion()
        self.ball_marker.pose.position = Point_from_p(self.ball_p)
        self.ball_marker.scale = Vector3(x=diam, y=diam, z=diam)
        self.ball_marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.8)
        self.markerarray = MarkerArray(markers=[self.ball_marker])
        
        # State machine for hitting behavior
        self.mode = "track"  # modes: "track", "swing", "return"
        self.hit= False
        self.p_start = self.p0.copy()  # Starting position for current trajectory

        # secondary task information
        self.q_center = np.radians(np.array([120.0, 80.0, -40.0,  10.0,  70.0, 70.0, -80.0])) # comfortable position when hitting, from one looking-good trial
        # self.q_center = self.q0 # just return to starting position
        self.lambda2 = 10.0  # secondary task gain

        ##############################################################
        # Setup the logistics of the node:
        # Add publishers to send the joint and task commands.  Also
        # add a TF broadcaster, so the desired pose appears in RVIZ.
        self.pubjoint = self.create_publisher(JointState, '/joint_states', 10)
        self.pubpose  = self.create_publisher(PoseStamped, '/pose', 10)
        self.pubtwist = self.create_publisher(TwistStamped, '/twist', 10)
        self.tfbroad  = tf2_ros.TransformBroadcaster(self)
        quality = QoSProfile(durability=DurabilityPolicy.TRANSIENT_LOCAL, depth=1)
        self.pubball = self.create_publisher(MarkerArray, '/visualization_marker_array', quality)

        # Wait for a connection to happen.  This isn't necessary, but
        # means we don't start until the rest of the system is ready.
        self.get_logger().info("Waiting for a /joint_states subscriber...")
        while(not self.count_subscribers('/joint_states')):
            pass

        # Set up the timer to update at 100Hz, with (t=0) occuring in
        # the first update cycle (dt) from now.
        self.dt    = 0.001                       # 100Hz.
        self.t     = -self.dt                   # Seconds since start
        self.now   = self.get_clock().now()     # ROS time since 1970
        self.timer = self.create_timer(self.dt, self.update)
        self.get_logger().info("Running with dt of %f seconds (%fHz)" %
                               (self.dt, 1/self.dt))

    # Shutdown
    def shutdown(self):
        # Destroy the timer, then shut down the node.
        self.timer.destroy()
        self.destroy_node()
        
        
    # function to estimate contact position (start simple, then expand to output desired contact metrics)
    def contact_metrics(self):
        # returns contact position, time to contact, and desired impact velocity
        # TO DO: EXPAND TO INCORPORATE TIMING CODE TO GENERATE CORRECT CONTACT METRICS
        return self.ball_p0, 2.0, np.array([0.1,0.1,0.1])


    # Update - send a new joint command every time step.
    def update(self):
        # Increment time.  We do so explicitly to avoid system jitter.
        self.t   = self.t   + self.dt
        self.now = self.now + rclpy.time.Duration(seconds=self.dt)

        # Update ball kinematics
        self.ball_v += self.dt * self.ball_a
        self.ball_p += self.dt * self.ball_v
        
        # Check for bounce
        if self.ball_p[2] < self.ball_radius:
            self.ball_p[2] = self.ball_radius + (self.ball_radius - self.ball_p[2])
            self.ball_v = self.ball_v+2*np.outer([0,0,1],[0,0,1])@(-self.ball_v)
        
        # Update ball marker
        self.ball_marker.header.stamp = self.now.to_msg()
        self.ball_marker.pose.position = Point_from_p(self.ball_p)
        self.pubball.publish(self.markerarray)
        
        # Gather contact metrics
        p_ball_estimate, time_to_contact, desired_tip_velocity = self.contact_metrics()

        # Get current tip position
        (ptip, Rtip, Jv, Jw) = self.chain.fkin(self.qc)

        # Calculate relative time for goto
        t_rel = self.t % time_to_contact
        
        # desired orientation (fixed for now)
        #Rd = Rotz(0) # @ Rotx(-np.pi/2)
        #wd = vzero()
        nd = np.array([1.0,1.0,1.0])

        # Generate trajectory
        if self.hit==False: # before contact
            (pd, vd) = spline(t_rel, time_to_contact, self.p0, p_ball_estimate, vzero(), desired_tip_velocity)
            # Grab the last joint command position and task errors.
            qclast = self.qc
            eplast = self.ep

            # compute errors
            vr = vd + self.lam * eplast

            # add extra task
            y = Rtip[:,1]
            J     = np.vstack((Jv, np.cross(y,nd).T @ Jw))
            extra_task = np.array([-self.lam*nd.T @ y])
            xrdot = np.concatenate((vr,extra_task))
            
            
            # FOR THE REPORT: MAY NEED TO DO STUDY ON CORRECT GAMMA
            Jdinv = J.T @ np.linalg.inv(J @ J.T + self.gamma**2 * np.eye(np.size(J, 0)))
            qcdot_primary = Jdinv @ xrdot
            
            # secondary task to move towards comfortable center position
            qcdot_secondary = -self.lambda2 * (qclast - self.q_center)
            qcdot = qcdot_primary + (np.eye(len(self.jointnames)) - Jdinv @ J) @ qcdot_secondary
            self.qcdot = qcdot

            # Integrate the joint position.
            qc = qclast + self.dt * qcdot

            # Compute the new forward kinematics for equivalent task commands.
            (pc, Rc, _, _) = self.chain.fkin(qc)

            # Save the joint command position and task errors.
            self.qc = qc
            self.ep = ep(pd, pc)

            # Calculate distance between tip and ball center.
            distance_to_ball = np.linalg.norm(pc - self.ball_p)
            
            # Check for collision (tip within ball radius + small tolerance)
            collision_threshold = 0.02
            
            if distance_to_ball < collision_threshold:
                # Get the tip velocity from joint velocities using Jacobian
                (_, _, Jv_collision, _) = self.chain.fkin(qc)
                tip_velocity = Jv_collision @ qcdot

                # use technique from class to compute ball kinematics
                collision_normal = (self.ball_p - pc) / distance_to_ball
                self.ball_v = self.ball_v + 2*np.outer(collision_normal,collision_normal) @ (tip_velocity-self.ball_v)
                
                # Log the hit
                self.get_logger().info(
                    f"HIT! Distance: {distance_to_ball:.4f}m, "
                    f"Ball velocity: [{self.ball_v[0]:.3f}, {self.ball_v[1]:.3f}, {self.ball_v[2]:.3f}]"
                )
                self.hit = True
                self.get_logger().info(f"Joint positions at hit: {np.degrees(self.qc)}")
                
        else: # after contact (slow down joints and return to starting position)
            
            if (np.linalg.norm(self.qc-self.q0)>0.01) and (np.linalg.norm(self.qcdot)<0.01):
                self.return_to_start = True # begin to return to start position
                
            if self.return_to_start == False: # continue to slow down swing
                slow_factor = 0.99
                qcdot = self.qcdot*slow_factor
                self.qcdot = qcdot
            
                # update qc
                qclast = self.qc
                qc = qclast+self.dt*qcdot
                self.qc = qc
            
                # calculate desired position and velocity
                J     = np.vstack((Jv, Jw))
                (pd, _, _, _) = self.chain.fkin(qc)
                vd = J@qcdot
                
                # store end position and time
                self.qc_swing_end = qc
                self.time_swing_end = self.t
            else: # continue to return to original position
                time_to_start = 0.5
                if np.linalg.norm(self.qc-self.q0)<0.01:
                    # if here we have successfully returned to start position
                    qc = self.qc
                    qcdot = self.qcdot
                    vd = vzero()
                    pd = self.p0
                else:
                    (qc,qcdot) = goto(self.t-self.time_swing_end,self.time_swing_end+time_to_start,self.qc_swing_end,self.q0)
                    self.qc = qc
                    self.qcdot = qcdot
                    # calculate desired position and velocity
                    J     = np.vstack((Jv, Jw))
                    (pd, _, _, _) = self.chain.fkin(qc)
                    vd = J@qcdot
        # PUBLISH
        header=Header(stamp=self.now.to_msg(), frame_id='world')
        self.pubjoint.publish(JointState(
            header=header,
            name=self.jointnames,
            position=qc.tolist(),
            velocity=qcdot.tolist()))
        #self.pubpose.publish(PoseStamped(
        #    header=header,
        #    pose=Pose_from_Rp(Rd,pd)))
        #self.pubtwist.publish(TwistStamped(
        #    header=header,
        #    twist=Twist_from_vw(vd,wd)))
        #self.tfbroad.sendTransform(TransformStamped(
        #    header=header,
        #    child_frame_id='desired',
        #    transform=Transform_from_Rp(Rd,pd)))

#
#  Main Code
#
def main(args=None):
    # Initialize ROS.
    rclpy.init(args=args)

    # Create a future object to signal when the trajectory ends.
    future = Future()

    # Initialize the trajectory generator node.
    trajectory = TrajectoryNode('trajectory', future)

    # Spin, meaning keep running (taking care of the timer callbacks
    # and message passing), until interrupted or the trajectory is
    # complete (as signaled by the future object).
    rclpy.spin_until_future_complete(trajectory, future)

    # Report the reason for shutting down.
    if future.done():
        trajectory.get_logger().info("Stopping: " + future.result())
    else:
        trajectory.get_logger().info("Stopping: Interrupted")

    # Shutdown the node and ROS.
    trajectory.shutdown()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
