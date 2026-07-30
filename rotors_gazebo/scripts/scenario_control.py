#!/usr/bin/env python3
"""One-shot CLI for manually driving the dynamic_llm_bbox_experiment scenario
during interactive testing -- reposition the drone, jump a SUV to a test
spot (pausing its automatic drive), or park/resume a pedestrian -- as an
alternative to run_ab_experiment.sh's fully-automated headless trials.

Run each command as its own invocation (like `rostopic pub -1`); this is a
one-shot tool, not a persistent node.

Examples:
  # Put the drone somewhere before sending it a goal (see caveat below)
  rosrun rotors_gazebo scenario_control.py drone -6 2 0.8

  # Freeze suv_front and place it exactly where you want to look at it
  rosrun rotors_gazebo scenario_control.py suv front goto 10 0 --yaw 180
  rosrun rotors_gazebo scenario_control.py suv front resume

  # Park road_walker at a specific spot for a screenshot, then let it go
  rosrun rotors_gazebo scenario_control.py pedestrian road_walker goto 5 -6
  rosrun rotors_gazebo scenario_control.py pedestrian road_walker resume

  # What's where, right now
  rosrun rotors_gazebo scenario_control.py status

Caveats:
  - `drone` teleports hummingbird directly via /gazebo/set_model_state.
    This only *sticks* if nothing is actively commanding it back -- i.e.
    before you've sent a 2D Nav Goal (ego-planner is idle) or while paused.
    Once it's flying a trajectory, the position controller will just fly it
    right back. To change the drone's actual spawn point, relaunch with
    init_x/init_y instead (see run_ab_experiment.sh's trial_params for the
    same pattern).
  - `suv ... goto/pause` only freezes *that* mover node's own updates
    (~external_control on suv_front_mover/suv_side_mover) -- it does not
    touch Gazebo physics, so nothing else stops it being pushed around by
    collisions if any are enabled.
  - `pedestrian ... goto` kills the road_walker_publisher.py /
    region_random_walker_publisher.py node outright (that's the only way to
    stop it from re-queuing new waypoints on top of your one-shot
    placement -- see gazebo_ros_actor_command's append-only path queue,
    documented in road_walker_publisher.py). `resume` relaunches it with
    the same defaults as dynamic_llm_bbox_experiment.launch.
  - `pedestrian ... goto` is NOT instant even after the kill: the C++ plugin
    (gazebo_ros_actor_command) appends incoming waypoints to its own
    internal queue instead of replacing it, and has no "clear" API. If the
    publisher had already queued a waypoint or two ahead (road_walker
    republishes ~half a leg-time early; random_walker republishes a fresh
    batch of 8 every ~8s), the actor will keep walking through those stale
    points first and only settle at your commanded (x, y) once it drains
    them -- this can take several seconds and is genuinely unpredictable
    from outside (confirmed by testing: it visibly walked through 2-3 stale
    waypoints before settling). Patching the plugin to replace-on-receive
    would fix this but was judged out of scope here, since
    region_random_walker_publisher.py's perpetual-motion trick relies on
    append semantics -- changing it would alter the actual experiment's
    pedestrian behavior, not just this manual tool.
"""
import argparse
import math
import subprocess
import sys
import time

import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetModelState, SetModelState
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from nav_msgs.msg import Path
from std_msgs.msg import Bool
from tf.transformations import quaternion_from_euler

SUV_MODEL = {"front": "suv_front", "side": "suv_side"}
PEDESTRIAN = {
    "random_walker": {
        "node": "region_random_walker_publisher",
        "script": "region_random_walker_publisher.py",
        "topic": "/random_walker/cmd_path",
        "resume_args": [],  # script's own defaults already match the launch file
    },
    "road_walker": {
        "node": "road_walker_publisher",
        "script": "road_walker_publisher.py",
        "topic": "/road_walker/cmd_path",
        "resume_args": [],
    },
}


def set_model_pose(model_name, x, y, z, yaw_deg=0.0):
    rospy.wait_for_service("/gazebo/set_model_state", timeout=5.0)
    set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
    state = ModelState()
    state.model_name = model_name
    state.pose.position.x = x
    state.pose.position.y = y
    state.pose.position.z = z
    qx, qy, qz, qw = quaternion_from_euler(0, 0, math.radians(yaw_deg))
    state.pose.orientation.x, state.pose.orientation.y = qx, qy
    state.pose.orientation.z, state.pose.orientation.w = qz, qw
    state.reference_frame = "world"
    set_state(state)


def get_model_pose(model_name):
    rospy.wait_for_service("/gazebo/get_model_state", timeout=5.0)
    get_state = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
    resp = get_state(model_name, "")
    return resp if resp.success else None


def publish_bool_latched(topic, value):
    pub = rospy.Publisher(topic, Bool, queue_size=1)
    time.sleep(0.3)  # let the subscriber connect -- classic ROS pub/sub race
    pub.publish(Bool(data=value))
    time.sleep(0.2)


def cmd_drone(args):
    set_model_pose(args.model, args.x, args.y, args.z, args.yaw)
    print(f"[drone] set {args.model} to ({args.x}, {args.y}, {args.z}), yaw={args.yaw} deg")
    print("  (sticks only if nothing is actively flying it right now -- see module docstring)")


def cmd_suv(args):
    model = SUV_MODEL[args.which]
    topic = f"/suv_{args.which}_mover/external_control"
    if args.action == "pause":
        publish_bool_latched(topic, True)
        print(f"[suv {args.which}] paused (external_control=True on {topic})")
    elif args.action == "resume":
        publish_bool_latched(topic, False)
        print(f"[suv {args.which}] resumed automatic driving")
    elif args.action == "goto":
        publish_bool_latched(topic, True)
        set_model_pose(model, args.x, args.y, 0.05, args.yaw)
        print(f"[suv {args.which}] paused + placed at ({args.x}, {args.y}), yaw={args.yaw} deg")
        print(f"  run: scenario_control.py suv {args.which} resume  -- to hand control back")


def cmd_pedestrian(args):
    info = PEDESTRIAN[args.which]
    node = "/" + info["node"]
    if args.action == "resume":
        subprocess.Popen(
            ["rosrun", "rotors_gazebo", info["script"]] + info["resume_args"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        print(f"[pedestrian {args.which}] relaunched {info['script']} with default params")
        return

    # goto: kill the publisher so it stops appending new waypoints on top
    # of this one, then send a single-point path.
    subprocess.run(["rosnode", "kill", node], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    pub = rospy.Publisher(info["topic"], Path, queue_size=1)
    time.sleep(0.3)
    path = Path()
    path.header.stamp = rospy.Time.now()
    path.header.frame_id = "world"
    pose = PoseStamped()
    pose.header.stamp = rospy.Time.now()
    pose.header.frame_id = "world"
    pose.pose.position = Point(args.x, args.y, 0.0)
    qx, qy, qz, qw = quaternion_from_euler(0, 0, math.radians(args.yaw))
    pose.pose.orientation = Quaternion(qx, qy, qz, qw)
    path.poses.append(pose)
    pub.publish(path)
    time.sleep(0.2)
    print(f"[pedestrian {args.which}] killed {node}, queued ({args.x}, {args.y}) as its next target")
    print("  it will settle there once it drains any waypoints the publisher had already "
          "queued -- may take a few seconds, not instant (see module docstring caveats)")
    print(f"  run: scenario_control.py pedestrian {args.which} resume  -- to restart normal walking")


def cmd_status(_args):
    names = ["hummingbird", "suv_front", "suv_side", "random_walker", "road_walker"]
    print(f"{'model':<14} {'x':>8} {'y':>8} {'z':>8}")
    for name in names:
        resp = get_model_pose(name)
        if resp is None:
            print(f"{name:<14} (not in the current world)")
            continue
        p = resp.pose.position
        print(f"{name:<14} {p.x:>8.2f} {p.y:>8.2f} {p.z:>8.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_drone = sub.add_parser("drone", help="teleport the drone (see caveat)")
    p_drone.add_argument("x", type=float)
    p_drone.add_argument("y", type=float)
    p_drone.add_argument("z", type=float)
    p_drone.add_argument("--yaw", type=float, default=0.0, help="degrees")
    p_drone.add_argument("--model", default="hummingbird")
    p_drone.set_defaults(func=cmd_drone)

    p_suv = sub.add_parser("suv", help="pause/resume/place a SUV")
    p_suv.add_argument("which", choices=SUV_MODEL.keys())
    suv_sub = p_suv.add_subparsers(dest="action", required=True)
    suv_sub.add_parser("pause")
    suv_sub.add_parser("resume")
    p_goto = suv_sub.add_parser("goto")
    p_goto.add_argument("x", type=float)
    p_goto.add_argument("y", type=float)
    p_goto.add_argument("--yaw", type=float, default=0.0, help="degrees")
    p_suv.set_defaults(func=cmd_suv)

    p_ped = sub.add_parser("pedestrian", help="park/resume a pedestrian actor")
    p_ped.add_argument("which", choices=PEDESTRIAN.keys())
    ped_sub = p_ped.add_subparsers(dest="action", required=True)
    ped_sub.add_parser("resume")
    p_pgoto = ped_sub.add_parser("goto")
    p_pgoto.add_argument("x", type=float)
    p_pgoto.add_argument("y", type=float)
    p_pgoto.add_argument("--yaw", type=float, default=0.0, help="degrees")
    p_ped.set_defaults(func=cmd_pedestrian)

    p_status = sub.add_parser("status", help="print current model_states positions")
    p_status.set_defaults(func=cmd_status)

    args = ap.parse_args()
    rospy.init_node("scenario_control", anonymous=True)
    args.func(args)


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSException as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
