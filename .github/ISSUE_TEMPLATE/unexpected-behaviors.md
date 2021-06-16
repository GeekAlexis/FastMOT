---
name: Unexpected behaviors
about: Report unexpected output, bugs, or ask questions
title: Please read & provide the following
labels: ''
assignees: ''

---

Your issue may already be reported!
Please **search** the issues before creating one.

## Current Behavior
<!--- If describing a bug, tell us what happens instead of the expected behavior -->
<!--- If suggesting a change/improvement, explain the difference from current behavior -->

## How to Reproduce
<!--- Provide a link to a video, or commands you run to -->
<!--- reproduce this bug. Include code or configuration (mot.json) changes, if relevant -->

## Describe what you want to do
1. What inputs you will provide, if any:
2. What outputs you are expecting:
3. Ask your questions here, if any:
<!--- Questions unrelated to FastMOT will not be answered --->

## Your Environment
<!--- Include as many relevant details about the environment you experienced the bug in -->
<!--- I cannot help you with installation without the docker or install_jetson.sh --->
* Desktop
  * Operating System and version:
  * Used the docker image?
* NVIDIA Jetson
  *  Nano/TX2/Xavier NX/Xavier AGX?
  * Jetpack version:
  * Ran install_jetson.sh?
  * Reinstalled OpenCV from Jetpack?

## Common issues
1. GStreamer warnings are normal
2. Use FFMPEG instead of GStreamer if you have issues on Desktop
2. TensorRT plugin and engine files have to be built on the target platform you plan to run
