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
#
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
        self.jointnames=['shoulder_roll_joint', 'shoulder_pitch_joint', 'shoulder_yaw_joint', 'elbow_pitch_joint', 'elbow_yaw_joint', 'wrist_pitch_joint', 'wrist_roll_joint']

        # Set up the kinematic chain object.
        self.chain = KinematicChain(self, 'world', 'virtual_endeffector', self.jointnames)

        # Define the matching initial joint/task positions.
        # 7 values for the 7 active DOFs (fixed joints don't need values)
        self.q0 = np.radians(np.array([180, 60, 0, 0, 0, 0, 0]))
        #self.p0 = np.array([0.0, 0.55, 1.0])
        #self.R0 = Reye()
        (self.p0, self.R0, _, _) = self.chain.fkin(self.q0)
        self.p = self.p0.copy()

        # Define the other points.
        self.pleft  = np.array([ 0.3, 0.5, 0.15])
        self.pright = np.array([-0.3, 0.5, 0.15])
        self.Rleft  = Rotx(-np.pi/2) @ Roty(-np.pi/2)
        self.Rleft  = Rotz( np.pi/2) @ Rotx(-np.pi/2)
        self.Rright = Reye()

        # Initialize the stored joint command position and task errors.
        self.qc = self.q0.copy()
        self.ep = vzero()
        self.eR = vzero()

        # Pick the convergence bandwidth.
        self.lam = 20
        self.gamma = 0.1
        
        # Ball physics setup
        self.ball_radius = 0.03
        self.ball_p = np.array([0.20, 0.20, self.ball_radius+1])
        self.ball_v = np.array([0.0, 0.0, 0.0])
        self.ball_a = np.array([0.0, 0.0, -0.5])
        
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
        self.traj_start_time = 0.0  # When current trajectory started
        self.traj_duration = 1.5    # How long the trajectory should take (seconds)
        self.p_start = self.p0.copy()  # Starting position for current trajectory

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
        self.dt    = 0.01                       # 100Hz.
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


    # Update - send a new joint command every time step.
    def update(self):
        # Increment time.  We do so explicitly to avoid system jitter.
        self.t   = self.t   + self.dt
        self.now = self.now + rclpy.time.Duration(seconds=self.dt)

        ##############################################################
        # COMPUTE THE TRAJECTORY AT THIS TIME INSTANCE.

        # # Stop everything after 8s - makes the graphing nicer.
        #if self.t > 20.0:
         #   self.future.set_result("Trajectory has ended")
          #  return
        '''
        # Cyclic (sinusoidal) movements, after the first 3s.
        s    =            cos(pi/2.5 * (self.t-3))
        sdot = - pi/2.5 * sin(pi/2.5 * (self.t-3))

        # Use the path variables to compute the position trajectory.
        pd = np.array([0.0, 0.95 - 0.25 * np.cos(self.t), 0.60 + 0.25 * np.sin(self.t)])
        vd = np.array([0.0, 0.25 * np.sin(self.t), 0.25 * np.cos(self.t)])
        '''
        # UPDATE BALL PHYSICS
        # Integrate the ball velocity, then position
        self.ball_v += self.dt * self.ball_a
        self.ball_p += self.dt * self.ball_v
        
        # Check for bounce
        if self.ball_p[2] < self.ball_radius:
            self.ball_p[2] = self.ball_radius + (self.ball_radius - self.ball_p[2])
            self.ball_v[2] *= -1.0 
            self.ball_v[0] *= -1.0  
        
        # Update ball marker
        self.ball_marker.header.stamp = self.now.to_msg()
        self.ball_marker.pose.position = Point_from_p(self.ball_p)
        self.pubball.publish(self.markerarray)
        
        # Ball prediction time (how far ahead to look)
        t_predict = 0.5  # Predict 0.5 seconds ahead

        # Predict where ball will be
        pball_predict = self.ball_p + self.ball_v * t_predict + 0.5 * self.ball_a * t_predict**2

        # Get current tip position
        (ptip, Rtip, Jv, Jw) = self.chain.fkin(self.qc)

        # Calculate relative time for goto
        t_rel = self.t % t_predict

        # Generate trajectory 
        if np.linalg.norm(pball_predict - ptip) > 1e-6: #t_rel < self.traj_duration:
            (pd, vd) = goto(t_rel, t_predict, ptip, pball_predict)
        if self.hit:
            pd = ptip
            vd = vzero()
			
        # Grab the last joint command position and task errors.
        qclast = self.qc
        eplast = self.ep
        eRlast = self.eR
        # COLLISION DETECTION AND MOMENTUM TRANSFER
        

        # Compute the old forward kinematics to get the Jacobians.
        #(_, _, Jv, Jw) = self.chain.fkin(qclast)

        # Compute the reference velocities (with errors of last cycle).
        wd = vzero()
        Rd = Rotz(0) @ Rotx(-np.pi/2)
        vr = vd + self.lam * eplast
        wr = wd + self.lam * eRlast

        # Compute the inverse kinematics.
        J     = np.vstack((Jv, Jw))
        xrdot = np.concatenate((vr, wr))
        Jdinv = J.T @ np.linalg.inv(J @ J.T + self.gamma**2 * np.eye(np.size(J, 0)))
        qcdot = Jdinv @ xrdot
        #qcdot = np.linalg.pinv(J) @ xrdot

        # Integrate the joint position.
        qc = qclast + self.dt * qcdot

        # Compute the new forward kinematics for equivalent task commands.
        (pc, Rc, _, _) = self.chain.fkin(qc)

        # Save the joint command position and task errors.
        self.qc = qc
        self.ep = ep(pd, pc)
        self.eR = eR(Rd, Rc)

        # Calculate distance between tip and ball center
        distance_to_ball = np.linalg.norm(pc - self.ball_p)
        
        # Check for collision (tip within ball radius + small tolerance)
        collision_threshold = self.ball_radius + 0.01  # 1cm tolerance
        
        if distance_to_ball < collision_threshold:
            # Get the tip velocity from joint velocities using Jacobian
            # We already have Jv from the forward kinematics
            (_, _, Jv_collision, _) = self.chain.fkin(qc)
            tip_velocity = Jv_collision @ qcdot
            
            # Calculate collision normal (from tip to ball center)
            if distance_to_ball > 1e-6:  # Avoid division by zero
                collision_normal = (self.ball_p - pc) / distance_to_ball
            else:
                collision_normal = np.array([0.0, 0.0, 1.0])  # Default upward
            
            # Calculate relative velocity (tip velocity in direction of ball)
            relative_velocity = np.dot(tip_velocity, collision_normal)
            
            # Only transfer momentum if tip is moving toward ball
            if relative_velocity > 0.05:  # Minimum impact velocity threshold (5 cm/s)
                
                # Transfer momentum - simplified collision model
                impulse_magnitude = relative_velocity
                
                # Apply impulse to ball
                self.ball_v += collision_normal * impulse_magnitude * 1.5  # 1.5 is a scaling factor
                
                # Optional: Add some random spin/deviation for realism
                # deviation = np.random.normal(0, 0.1, 3)
                # self.ball_v += deviation
                
                # Log the hit
                self.get_logger().info(
                    f"HIT! Distance: {distance_to_ball:.4f}m, "
                    f"Impact velocity: {relative_velocity:.3f}m/s, "
                    f"Ball velocity: [{self.ball_v[0]:.3f}, {self.ball_v[1]:.3f}, {self.ball_v[2]:.3f}]"
                )
                self.hit = True

        ##############################################################
        # Finish by publishing the data (joint and task commands).
        #  qc and qcdot = Joint Commands  as  /joint_states  to view/plot
        #  pd and Rd    = Task pos/orient as  /pose & TF     to view/plot
        #  vd and wd    = Task velocities as  /twist         to      plot
        header=Header(stamp=self.now.to_msg(), frame_id='world')
        self.pubjoint.publish(JointState(
            header=header,
            name=self.jointnames,
            position=qc.tolist(),
            velocity=qcdot.tolist()))
        self.pubpose.publish(PoseStamped(
            header=header,
            pose=Pose_from_Rp(Rd,pd)))
        self.pubtwist.publish(TwistStamped(
            header=header,
            twist=Twist_from_vw(vd,wd)))
        self.tfbroad.sendTransform(TransformStamped(
            header=header,
            child_frame_id='desired',
            transform=Transform_from_Rp(Rd,pd)))


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
