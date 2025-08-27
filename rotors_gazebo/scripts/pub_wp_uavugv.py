#!/usr/bin/env python3
import rospy
import csv
from collections import defaultdict
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint
from geometry_msgs.msg import Transform, Vector3, Quaternion, Twist
from std_msgs.msg import Header
from rospy.rostime import Duration
import math
import tf
from nav_msgs.msg import Odometry

DEFAULT_Z = 1.0
DEFAULT_DT = 1.0
TARGET_SPEED = 1.0

init_ugv_position = (-1.5, 0, 0)
current_uav_positions = {}
current_ugv_positions = {}

def uav_odom_callback(msg, uav_id):
    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y
    z = msg.pose.pose.position.z
    current_uav_positions[uav_id] = (x, y, z)

ugv_orientations = {}

def ugv_odom_callback(msg, ugv_id):
    x = msg.pose.pose.position.x + init_ugv_position[0]
    y = msg.pose.pose.position.y + init_ugv_position[1]
    z = msg.pose.pose.position.z + init_ugv_position[2]
    q = msg.pose.pose.orientation
    current_ugv_positions[ugv_id] = (x, y, z)
    ugv_orientations[ugv_id] = (q.x, q.y, q.z, q.w)

    
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
    """Load waypoints from CSV, skip comments (# full line or inline).
       Returns dict: robot_id -> list of (x,y,z,t)"""
    waypoints = defaultdict(list)
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        has_z = "z" in reader.fieldnames
        has_time = "time_from_start" in reader.fieldnames

        for row in reader:
            # Clean each cell: drop inline comments, strip whitespace
            clean_row = {}
            for k, v in row.items():
                if v is None:
                    continue
                v = v.split("#")[0].strip()   # remove inline comments
                clean_row[k] = v

            robot_id = clean_row.get("robot_id") or clean_row.get("uav_id")
            if not robot_id or robot_id == "":
                continue  # skip blanks/comments

            try:
                x = float(clean_row["x"])
                y = float(clean_row["y"])
                z = float(clean_row["z"]) if has_z and clean_row.get("z", "") != "" else DEFAULT_Z
            except (KeyError, ValueError):
                rospy.logwarn(f"Skipping malformed row: {row}")
                continue

            if has_time and clean_row.get("time_from_start"):
                try:
                    t = float(clean_row["time_from_start"])
                except ValueError:
                    t = len(waypoints[robot_id]) * DEFAULT_DT
            else:
                t = len(waypoints[robot_id]) * DEFAULT_DT

            waypoints[robot_id].append((x, y, z, t))

    return waypoints


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
    rospy.init_node("multi_robot_controller")

    csv_file = "waypoints2.csv"

    # --- Load CSV ---
    waypoints = []
    with open(csv_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Clean each cell: remove inline comments and trim
            clean_row = {k: (v.split("#")[0].strip() if v else "") for k, v in row.items()}

            robot_id = clean_row.get("robot_id", "")
            if not robot_id:
                continue  # skip blank or comment-only rows

            try:
                x = float(clean_row["x"])
                y = float(clean_row["y"])
                z = float(clean_row["z"])
            except (KeyError, ValueError):
                rospy.logwarn(f"Skipping malformed row: {row}")
                continue

            waypoints.append((robot_id, x, y, z))


    rospy.loginfo("Loaded %d waypoints (sequential execution)", len(waypoints))

    # --- Publishers and Subscribers ---
    uav_publishers = {}
    ugv_publishers = {}

    # UAVs
    uav_ids = sorted(set([wp[0] for wp in waypoints if wp[0].startswith("u")]))
    for uav_id in uav_ids:
        ns = f"/hummingbird/command/trajectory"
        uav_publishers[uav_id] = rospy.Publisher(ns, MultiDOFJointTrajectory, queue_size=10)
        rospy.Subscriber(f"/hummingbird/ground_truth/odometry", Odometry, uav_odom_callback, uav_id)
        rospy.loginfo("UAV %s -> pub: %s", uav_id, ns)

    # UGVs
    ugv_ids = sorted(set([wp[0] for wp in waypoints if wp[0].startswith("g")]))
    for ugv_id in ugv_ids:
        ns = f"/jackal_velocity_controller/cmd_vel"
        ugv_publishers[ugv_id] = rospy.Publisher(ns, Twist, queue_size=10)
        rospy.Subscriber(f"/jackal_velocity_controller/odom", Odometry, ugv_odom_callback, ugv_id)
        rospy.loginfo("UGV %s -> pub: %s", ugv_id, ns)

    rospy.sleep(1.0)
    t_accum = 0.0
    # --- Execute Waypoints ---
    for robot_id, x2, y2, z2 in waypoints:
        if robot_id.startswith("u"):   # UAV
            uav_id = robot_id
            if uav_id in current_uav_positions:
                x1, y1, _ = current_uav_positions[uav_id]
                rospy.loginfo("Using odom for UAV %s: (%.2f, %.2f)", uav_id, x1, y1)
            else:
                x1, y1 = x2, y2
                rospy.logwarn("No odom yet for UAV %s, using CSV start", uav_id)

            z1 = z2
            pub = uav_publishers[uav_id]

            dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            if dist < 0.01:
                rospy.logwarn("Waypoint for UAV %s too close, skipping", uav_id)
                continue

            duration = max(1.0, dist * 1.5 / TARGET_SPEED)
            wp1 = (x1, y1, z1, t_accum)
            wp2 = (x2, y2, z2, t_accum + duration)

            if dist < 1.0:
                wp_dense = interpolate_waypoints([wp1, wp2], start_time=t_accum)
                traj = create_trajectory(wp_dense)
            else:
                traj = create_minjerk_trajectory(wp1, wp2, duration)

            pub.publish(traj)
            t_accum += duration
            rospy.sleep(duration + 1.0)

        elif robot_id.startswith("g"):   # UGV (Jackal)
            ugv_id = robot_id
            pub = ugv_publishers[ugv_id]

            # --- Tunables ---
            V_MAX = 0.6            # m/s max forward
            W_MAX = 1.2            # rad/s max yaw rate
            K_LIN = 0.9            # linear gain (toward goal)
            K_ANG = 2.0            # angular gain (steer to heading)
            ALIGN_THRESH = 0.15    # rad; must be within this to start driving
            KEEP_ALIGN = 0.25      # rad; if error grows past this, stop driving
            STOP_DIST = 0.08       # m; goal tolerance
            SLOW_DIST = 0.5        # m; start slowing when within this
            MIN_V = 0.05           # m/s; minimum forward once aligned
            YAW_ALPHA = 0.5        # low-pass on yaw (0=none, 1=all new)

            # store last yaw for simple LPF
            if 'last_yaw' not in globals():
                last_yaw = {}

            goal_reached = False
            rate = rospy.Rate(20)  # 20 Hz

            while not rospy.is_shutdown() and not goal_reached:
                if ugv_id not in current_ugv_positions or ugv_id not in ugv_orientations:
                    rospy.logwarn_throttle(2.0, "No odometry yet for UGV %s, waiting...", ugv_id)
                    rate.sleep()
                    continue

                x1, y1, _ = current_ugv_positions[ugv_id]
                qx, qy, qz, qw = ugv_orientations[ugv_id]

                # current yaw (optionally low-pass)
                _, _, yaw_meas = tf.transformations.euler_from_quaternion([qx, qy, qz, qw])
                if ugv_id in last_yaw:
                    yaw = (1.0 - YAW_ALPHA) * last_yaw[ugv_id] + YAW_ALPHA * yaw_meas
                else:
                    yaw = yaw_meas
                last_yaw[ugv_id] = yaw

                # position error
                dx, dy = x2 - x1, y2 - y1
                dist = math.hypot(dx, dy)
                if dist < STOP_DIST:
                    pub.publish(Twist())  # stop
                    rospy.loginfo("UGV %s reached goal (%.2f, %.2f)", ugv_id, x2, y2)
                    goal_reached = True
                    break

                # heading error
                goal_yaw = math.atan2(dy, dx)
                err = goal_yaw - yaw
                err = math.atan2(math.sin(err), math.cos(err))  # normalize to [-pi, pi]

                # angular cmd (saturated)
                wz = max(-W_MAX, min(W_MAX, K_ANG * err))

                # linear cmd:
                #  - zero if not aligned enough
                #  - otherwise scale by distance (slow down near goal) and by cos(err)
                aligned = abs(err) < ALIGN_THRESH
                still_aligned = abs(err) < KEEP_ALIGN
                if aligned or still_aligned:
                    # distance-based scaling with floor
                    v_des = K_LIN * dist
                    if dist < SLOW_DIST:
                        v_des *= (dist / SLOW_DIST)  # gentle taper
                    v_des = max(MIN_V, min(V_MAX, v_des))
                    # reduce forward motion when not perfectly aligned
                    v_des *= max(0.0, math.cos(err))
                else:
                    v_des = 0.0  # rotate in place until aligned

                twist = Twist()
                twist.linear.x = v_des
                twist.angular.z = wz
                pub.publish(twist)

                rospy.loginfo_throttle(0.5,
                    "UGV %s → goal(%.2f,%.2f) dist=%.2f yaw=%.2f err=%.2f | vx=%.2f wz=%.2f",
                    ugv_id, x2, y2, dist, yaw, err, v_des, wz)

                rate.sleep()


    rospy.loginfo("All waypoints executed.")
