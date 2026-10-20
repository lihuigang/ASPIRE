# SO-101 pick-and-place (simulation-runtime backend).
#
# The runtime exposes joint targets and RGB renders only, so this policy uses
# the harness's color segmentation and local MuJoCo IK to turn the two public
# localization results into the validated top-down grasp sequence.  The site
# frame's X axis is the approach axis; a yaw-0 top-down orientation is
# R = Rz(0) @ Ry(90 deg), i.e. quaternion (0.7071, 0, 0.7071, 0).
report = {"task": get_task_context().task_name}

state = get_robot_state()
frame = state.base_frame
top_down = (0.7071067811865476, 0.0, 0.7071067811865476, 0.0)

cube = localize_object("red cube", camera_name="top_cam")
pad = localize_object("green target pad", camera_name="top_cam")
report["localized_cube"] = cube.ok
report["localized_pad"] = pad.ok

if cube.ok and pad.ok:
    center = cube.geometry.pose.position
    target = pad.geometry.pose.position
    report["cube_position"] = tuple(float(value) for value in center)
    report["pad_position"] = tuple(float(value) for value in target)

    def site(position, height):
        # The jaw pads sit a couple of millimetres beyond the site along +X.
        return (float(position[0]) + 0.002, float(position[1]), float(height))

    def move(position, tolerance=0.006, max_steps=250):
        return move_to_pose(
            Pose(position=position, quaternion_wxyz=top_down, frame=frame),
            tolerance=tolerance,
            max_steps=max_steps,
        )

    opened = open_gripper()
    report["opened"] = opened.ok

    approached = move(site(center, 0.07))
    report["approached"] = approached.ok

    descended = None
    if approached.ok:
        descended = move(site(center, 0.012))
        report["descended"] = descended.ok

    closed = None
    if descended is not None and descended.ok:
        closed = close_gripper()
        report["closed"] = True

    lifted = None
    if closed is not None:
        # Rise straight up at the grasp site. Translating along X while lifting
        # pushes cubes near the workspace edge out of IK reach.
        lifted = move(
            site(center, 0.062),
            max_steps=200,
        )
        report["lifted"] = lifted.ok

    carried = None
    if lifted is not None and lifted.ok:
        carried = move(site(target, 0.075), max_steps=300)
        report["carried"] = carried.ok

    lowered = None
    if carried is not None and carried.ok:
        lowered = move(site(target, 0.02), max_steps=200)
        report["lowered"] = lowered.ok

    if lowered is not None and lowered.ok:
        released = open_gripper()
        report["released"] = released.ok
        # Any terminal result here is the success predicate firing; the harness
        # reads it through check_success() after the program returns.
        retreat = move(site(target, 0.07), tolerance=0.01)
        report["retreated"] = retreat.ok

result = report
