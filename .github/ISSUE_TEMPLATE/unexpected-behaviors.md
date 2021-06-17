---
name: Unexpected behaviors
about: Report unexpected behaviors/bugs or ask questions
title: Please read & provide the following
labels: ''
assignees: ''

---

Your issue may already be reported!
Please search the issues before creating one.

## Current Behavior
<!--- If describing a bug, tell us what happens instead of the expected behavior -->
<!--- Provide link to an output video, if possible -->

## How to Reproduce
<!--- Provide commands you run to reproduce this bug -->
<!--- Include code or configuration (mot.json) changes, if relevant -->

## Describe what you want to do
1. What input videos you will provide, if any:
2. What outputs you are expecting:
3. Ask your questions here, if any:
<!--- Questions unrelated to FastMOT will not be answered --->

## Your Environment
<!--- Include as many relevant details about the environment you experienced the bug in -->
<!--- I cannot help you with installation method not specified in README --->
* Desktop
  * Operating System and version:
  * NVIDIA Driver version:
  * Used the docker image?
* NVIDIA Jetson
  *  Which Jetson?
  * Jetpack version:
  * Ran install_jetson.sh?
  * Reinstalled OpenCV from Jetpack?

## Common issues
1. GStreamer warnings are normal
2. If you have issues with GStreamer on Desktop, disable GStreamer and build FFMPEG instead in Dockerfile
2. TensorRT plugin and engine files have to be built on the target platform you plan to run
