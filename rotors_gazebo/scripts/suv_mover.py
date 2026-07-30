#!/usr/bin/env python3
"""Drives a static Gazebo model between two points at a constant speed by
calling /gazebo/set_model_state directly. Used for the SUV models in
outdoor_dynamic_llm_bbox.world -- they have no walking-actor skeleton, so
gazebo_ros_actor_command (which requires an <actor>) doesn't apply; this is
the simplest way to get repeatable, parameterized scripted motion for a
plain rigid model.

Two modes (~mode param):
  loop     back and forth between (x0,y0) and (x1,y1), pausing at each end.
  one_way  always drives (x0,y0) -> (x1,y1) (never the reverse leg), pauses
           at (x1,y1), then teleports straight back to (x0,y0) (no visible
           driving) before repeating. Use this for anything meant to
           *approach* the drone: a forward-facing camera can't see
           something behind it, and the reverse leg of "loop" mode is
           exactly that -- once the drone has flown past (x1,y1), driving
           back toward (x0,y0) means approaching the drone from its blind
           side. (x0,y0) should be placed well beyond wherever the drone is
           actually headed, so the one-way charge is always still ahead of
           it, in camera view, when it starts.

One node instance per vehicle (see dynamic_llm_bbox_experiment.launch),
distinguished by the ~model_name param.

Publish std_msgs/Bool True on ~external_control (e.g.
/suv_front_mover/external_control) to freeze this node's own updates --
otherwise any outside attempt to reposition the model via
/gazebo/set_model_state gets overwritten within one tick (up to
~1/~rate seconds later). See scenario_control.py, which drives this for
manual/interactive testing.
"""
import math

import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from std_msgs.msg import Bool
from tf.transformations import quaternion_from_euler


def lerp_pose(start, end, t):
    x = start[0] + (end[0] - start[0]) * t
    y = start[1] + (end[1] - start[1]) * t
    return x, y


def main():
    rospy.init_node("suv_mover")

    model_name = rospy.get_param("~model_name", "suv_front")
    mode = rospy.get_param("~mode", "loop")  # "loop" or "one_way"
    x0 = rospy.get_param("~x0")
    y0 = rospy.get_param("~y0")
    x1 = rospy.get_param("~x1")
    y1 = rospy.get_param("~y1")
    z = rospy.get_param("~z", 0.05)
    speed = rospy.get_param("~speed", 3.0)  # m/s
    pause_time = rospy.get_param("~pause_time", 2.0)  # s, at each end
    heading_offset_deg = rospy.get_param("~heading_offset_deg", 0.0)
    rate_hz = rospy.get_param("~rate", 20.0)

    rospy.wait_for_service("/gazebo/set_model_state")
    set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)

    start = (x0, y0)
    end = (x1, y1)
    leg_length = math.hypot(end[0] - start[0], end[1] - start[1])
    leg_time = leg_length / max(speed, 1e-3)
    heading_offset = math.radians(heading_offset_deg)

    rospy.loginfo(f"[suv_mover:{model_name}] mode={mode} {leg_length:.1f}m between "
                  f"{start} and {end}, {leg_time:.1f}s per leg + "
                  f"{pause_time}s pause")

    def publish_pose(x, y, yaw):
        state = ModelState()
        state.model_name = model_name
        state.pose.position.x = x
        state.pose.position.y = y
        state.pose.position.z = z
        qx, qy, qz, qw = quaternion_from_euler(0, 0, yaw)
        state.pose.orientation.x = qx
        state.pose.orientation.y = qy
        state.pose.orientation.z = qz
        state.pose.orientation.w = qw
        state.reference_frame = "world"
        try:
            set_state(state)
        except rospy.ServiceException as e:
            rospy.logwarn_throttle(5.0, f"[suv_mover:{model_name}] set_model_state failed: {e}")

    control_state = {"external": False}

    def external_control_cb(msg):
        control_state["external"] = bool(msg.data)

    rospy.Subscriber("~external_control", Bool, external_control_cb)

    rate = rospy.Rate(rate_hz)
    forward = True
    leg_start_time = rospy.Time.now().to_sec()
    phase = "moving"  # "moving" or "paused"
    approach_yaw = math.atan2(end[1] - start[1], end[0] - start[0]) + heading_offset
    was_external = False
    external_since = None

    while not rospy.is_shutdown():
        now = rospy.Time.now().to_sec()

        if control_state["external"]:
            # Under manual control: don't touch the model's pose at all, so
            # whatever an outside tool sets via /gazebo/set_model_state
            # sticks. Just remember when this started so the leg timer can
            # skip over the gap once control is handed back.
            if not was_external:
                external_since = now
                was_external = True
            rate.sleep()
            continue
        if was_external:
            leg_start_time += now - external_since
            was_external = False

        elapsed = now - leg_start_time

        if mode == "one_way":
            # Always the same direction of travel; the return trip is an
            # instant, invisible reset rather than a driven reverse leg.
            if phase == "moving":
                t = min(elapsed / leg_time, 1.0)
                x, y = lerp_pose(start, end, t)
                yaw = approach_yaw
                if t >= 1.0:
                    phase = "paused"
                    leg_start_time = now
            else:  # paused at the far end, then reset
                x, y = end
                yaw = approach_yaw
                if elapsed >= pause_time:
                    publish_pose(*start, approach_yaw)  # instant teleport back
                    phase = "moving"
                    leg_start_time = now
                    rate.sleep()
                    continue
        else:  # loop
            src, dst = (start, end) if forward else (end, start)
            yaw = math.atan2(dst[1] - src[1], dst[0] - src[0]) + heading_offset
            if phase == "moving":
                t = min(elapsed / leg_time, 1.0)
                x, y = lerp_pose(src, dst, t)
                if t >= 1.0:
                    phase = "paused"
                    leg_start_time = now
            else:  # paused
                x, y = dst
                if elapsed >= pause_time:
                    phase = "moving"
                    forward = not forward
                    leg_start_time = now

        publish_pose(x, y, yaw)
        rate.sleep()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
