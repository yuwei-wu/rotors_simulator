#!/usr/bin/env python3
import rospy
import csv
from collections import defaultdict
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint
from geometry_msgs.msg import Transform, Vector3, Quaternion, Twist
from std_msgs.msg import Header
from rospy.rostime import Duration
import math

DEFAULT_Z = 1.0
DEFAULT_DT = 1.0
TARGET_SPEED = 1.5


from nav_msgs.msg import Odometry

current_positions = {}  # store latest pose per UAV

def odom_callback(msg, uav_id):
    """Update UAV's current position from odometry"""
    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y
    z = msg.pose.pose.position.z
    current_positions[uav_id] = (x, y, z)


    
def min_jerk_coeffs(p0, pf, T):
    a0 = p0
    a1 = 0
    a2 = 0
    a3 = (10*(pf - p0)) / (T**3)
    a4 = (-15*(pf - p0)) / (T**4)
    a5 = (6*(pf - p0)) / (T**5)
    return a0, a1, a2, a3, a4, a5

def min_jerk_eval(coeffs, t):
    a0, a1, a2, a3, a4, a5 = coeffs
    pos = a0 + a1*t + a2*t**2 + a3*t**3 + a4*t**4 + a5*t**5
    vel = a1 + 2*a2*t + 3*a3*t**2 + 4*a4*t**3 + 5*a5*t**4
    acc = 2*a2 + 6*a3*t + 12*a4*t**2 + 20*a5*t**3
    return pos, vel, acc

def create_minjerk_trajectory(start, goal, duration, num_samples=50):
    traj = MultiDOFJointTrajectory()
    traj.header = Header()
    traj.header.stamp = rospy.Time.now()
    traj.header.frame_id = "world"
    traj.joint_names.append("")

    coeffs_x = min_jerk_coeffs(start[0], goal[0], duration)
    coeffs_y = min_jerk_coeffs(start[1], goal[1], duration)
    coeffs_z = min_jerk_coeffs(start[2], goal[2], duration)

    for i in range(num_samples+1):
        t = (i / num_samples) * duration
        px, vx, ax = min_jerk_eval(coeffs_x, t)
        py, vy, ay = min_jerk_eval(coeffs_y, t)
        pz, vz, az = min_jerk_eval(coeffs_z, t)

        point = MultiDOFJointTrajectoryPoint()
        transform = Transform()
        transform.translation = Vector3(px, py, pz)
        transform.rotation = Quaternion(0.0, 0.0, 0.0, 1.0)
        point.transforms.append(transform)

        vel = Twist()
        vel.linear = Vector3(vx, vy, vz)
        point.velocities.append(vel)

        acc = Twist()
        acc.linear = Vector3(ax, ay, az)
        point.accelerations.append(acc)

        point.time_from_start = Duration.from_sec(t)
        traj.points.append(point)

    return traj

def load_csv(filename):
    """Load waypoints and group by UAV index"""
    uav_waypoints = defaultdict(list)
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        has_z = "z" in reader.fieldnames
        has_time = "time_from_start" in reader.fieldnames

        for row in reader:
            uav_id = int(row["uav_id"])  # first column
            x = float(row["x"])
            y = float(row["y"])
            z = float(row["z"]) if has_z else DEFAULT_Z
            if has_time:
                t = float(row["time_from_start"])
            else:
                t = len(uav_waypoints[uav_id]) * DEFAULT_DT
            uav_waypoints[uav_id].append((x, y, z, t))

    return uav_waypoints


def create_trajectory(waypoints):
    traj = MultiDOFJointTrajectory()
    traj.header = Header()
    traj.header.stamp = rospy.Time.now()
    traj.header.frame_id = "world"

    traj.joint_names.append("")  # placeholder

    n = len(waypoints)

    for i, (x, y, z, t) in enumerate(waypoints):
        point = MultiDOFJointTrajectoryPoint()

        # Position & orientation
        transform = Transform()
        transform.translation = Vector3(x, y, z)
        transform.rotation = Quaternion(0.0, 0.0, 0.0, 1.0)
        point.transforms.append(transform)

        # --- Velocity estimation ---
        if i == 0:  # forward difference
            x2, y2, z2, t2 = waypoints[i + 1]
            dt = max(1e-6, t2 - t)
            vx, vy, vz = (x2 - x) / dt, (y2 - y) / dt, (z2 - z) / dt
        elif i == n - 1:  # backward difference
            x1, y1, z1, t1 = waypoints[i - 1]
            dt = max(1e-6, t - t1)
            vx, vy, vz = (x - x1) / dt, (y - y1) / dt, (z - z1) / dt
        else:  # central difference
            x1, y1, z1, t1 = waypoints[i - 1]
            x2, y2, z2, t2 = waypoints[i + 1]
            dt = max(1e-6, t2 - t1)
            vx, vy, vz = (x2 - x1) / dt, (y2 - y1) / dt, (z2 - z1) / dt

        vel = Twist()
        #vel.linear = Vector3(vx, vy, vz)
        point.velocities.append(vel)

        # Accelerations: leave zero for now
        acc = Twist()
        point.accelerations.append(acc)

        # Timing
        point.time_from_start = Duration.from_sec(t)

        traj.points.append(point)
    
    #print(traj)

    return traj

def interpolate_waypoints(waypoints, start_time=0.0):
    """Insert intermediate waypoints based on distance and target speed,
    accumulating time correctly."""
    new_wps = []
    t_accum = start_time

    for i in range(len(waypoints) - 1):
        x1, y1, z1, _ = waypoints[i]
        x2, y2, z2, _ = waypoints[i + 1]

        dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)

        # time needed at target speed
        seg_time = dist / 0.5
        steps = max(2, int(seg_time))  # at least 2 points

        for s in range(steps):
            alpha = float(s) / steps
            x = x1 + alpha * dx
            y = y1 + alpha * dy
            z = z1 + alpha * dz
            t = t_accum + alpha * seg_time
            new_wps.append((x, y, z, t))

        # advance the accumulated time by the full segment
        t_accum += seg_time

    # add final point
    x2, y2, z2, _ = waypoints[-1]
    new_wps.append((x2, y2, z2, t_accum))

    print(f"Interpolated {len(waypoints)} waypoints to {len(new_wps)} waypoints over {t_accum - start_time:.2f} seconds")

    return new_wps


if __name__ == "__main__":
    rospy.init_node("multi_uav_trajectory_publisher")

    csv_file = "waypoints.csv"


    # Load CSV as a flat list of (uav_id, x, y, z)
    waypoints = []
    with open(csv_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            uav_id = int(row["uav_id"])
            x = float(row["x"])
            y = float(row["y"])
            z = float(row["z"])
            waypoints.append((uav_id, x, y, z))

    rospy.loginfo("Loaded %d waypoints (sequential execution)", len(waypoints))

    # Create publishers and subscribers on demand
    publishers = {}

    # Make subscribers for each uav_id
    uav_ids = set([wp[0] for wp in waypoints])
    for uav_id in uav_ids:
        ns = f"/hummingbird{uav_id}/command/trajectory"
        publishers[uav_id] = rospy.Publisher(ns, MultiDOFJointTrajectory, queue_size=10)
        rospy.Subscriber(f"/hummingbird{uav_id}/ground_truth/odometry", Odometry, odom_callback, uav_id)
        rospy.loginfo("Publisher+Subscriber created for UAV %d", uav_id)
    
    
    
    rospy.sleep(1.0)
    t_accum = 0.0
    for i in range(len(waypoints)):

        uav_id, x2, y2, z2 = waypoints[i]

        # Instead of wp1 from CSV, use current odometry if available
        if uav_id in current_positions:
            x1, y1, _ = current_positions[uav_id]
            rospy.loginfo("Using current odom for UAV %d: (%.2f, %.2f)", uav_id, x1, y1)
        else:
            # fallback: CSV start
            rospy.logwarn("No odometry yet for UAV %d, using CSV start", uav_id)

        z1 = waypoints[i][3]
        # ensure publisher exists
        if uav_id not in publishers:
            ns = f"/hummingbird{uav_id}/command/trajectory"
            publishers[uav_id] = rospy.Publisher(ns, MultiDOFJointTrajectory, queue_size=10)
            rospy.loginfo("Publisher created for UAV %d -> %s", uav_id, ns)

        pub = publishers[uav_id]

        # compute distance and duration
        dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        if dist < 0.01:
            rospy.logwarn("Waypoints %d and %d for UAV %d are too close, skipping", i, i+1, uav_id)
            continue

        duration = dist * 1.5 / TARGET_SPEED 
        wp1 = (x1, y1, z1, t_accum)
        wp2 = (x2, y2, z2, t_accum + duration)

        duration = max(1.0, duration)
        #rospy.loginfo("Publishing UAV %d trajectory from %s to %s", uav_id, wp1, wp2)
        print(f"Publishing UAV {uav_id} trajectory from {wp1} to {wp2}, dist={dist:.2f}m, duration={duration:.2f}s")


        if dist < 1.0:
            wp_dense = interpolate_waypoints([wp1, wp2], start_time=t_accum)
            traj = create_trajectory(wp_dense)
            print(f"Publishing UAV {uav_id} trajectory (simple) from {wp1} to {wp2}")
        else:
            traj = create_minjerk_trajectory(wp1, wp2, duration)
        
        
        pub.publish(traj)
        t_accum += duration

        rospy.sleep(duration + 1.5)

    rospy.loginfo("All trajectories published in CSV order.")