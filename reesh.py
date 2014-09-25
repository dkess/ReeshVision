#!/usr/bin/env python
#
# Author: Daniel Kessler, team 1836
# modified from Team 254's CheesyVision
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# The views and conclusions contained in the software and documentation are those
# of the authors and should not be interpreted as representing official policies,
# either expressed or implied, of the FreeBSD Project.

import random
import socket
import time

import cv2 as cv
import numpy as np

config = {}
WINDOW_NAME = "Reeeeeeesh"

with open("config") as configfile:
    for l in configfile.readlines():
        split = l[:-1].split("=", 1)
        if len(split) > 1:
            config[split[0]] = int(split[1])

HOST, PORT = "10.18.36.2", 1180

CAL_UL = (config["webcam_w"]/2 - 20, 200)
CAL_LR = (config["webcam_w"]/2 + 20, 240)

HAND_W = 80
HAND_H = 50

METER_TOP = 170
METER_BOTTOM = 460

# distance between the centers of the left and right meters
METER_DIST = 60

# From each meter we get stripes of color that are averaged and compared to the
# calibration color.  The STRIPE_LENGTH is how big these stripes are-- bigger
# stripes are less accurate but faster.
# 7 is the absolute min
STRIPE_LENGTH = 7

UPDATE_RATE_HZ = 40.0
PERIOD = (1.0 / UPDATE_RATE_HZ) * 1000.0
PERIOD = 0.3

def avg_color(img, box):
    ''' Return the average HSV color of a region in img. '''
    h = np.mean(img[box[0][1]+3:box[1][1]-3, box[0][0]+3:box[1][0]-3, 0])
    s = np.mean(img[box[0][1]+3:box[1][1]-3, box[0][0]+3:box[1][0]-3, 1])
    v = np.mean(img[box[0][1]+3:box[1][1]-3, box[0][0]+3:box[1][0]-3, 2])
    return (h,s,v)

def color_distance(c1, c2):
    ''' Compute the difference between two HSV colors.

    Currently this simply returns the "L1 norm" for distance,
    or delta_h + delta_s + delta_v.  This is not a very robust
    way to do it, but it has worked well enough in our tests.

    Recommended reading:
    http://en.wikipedia.org/wiki/Color_difference
    '''
    total_diff = 0
    for i in (0, 1, 2):
        diff = (c1[i]-c2[i])
        # Wrap hue angle...OpenCV represents hue on (0, 180)
        if i == 0:
            if diff < -90:
                diff += 180
            elif diff > 90:
                diff -= 180
        total_diff += abs(diff)
    return total_diff

def get_time_millis():
    ''' Get the current time in milliseconds. '''
    return int(round(time.time() * 1000))

def speeds_to_byte(left,right):
    o = abs(left) << 4
    o += abs(right)
    if left < 0:
        o += 128
    if right < 0:
        o += 8
    return o

last_t = get_time_millis()

connected = False
s = None

#b_l = 0
#b_r = 0
byte_to_send = 0

if __name__ == "__main__":
    cv.namedWindow(WINDOW_NAME, 1)

    capture = cv.VideoCapture(config["camera"])

    capture.set(15, config["exposure"])

    while 1:
        has_frame, frame = capture.read()
        if not has_frame:
            time.sleep(0.0254)
            continue

        cur_time = get_time_millis()
        if last_t + PERIOD <= cur_time:
            if not connected:
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.settimeout(.5)
                    s.connect(("10.18.36.2", 1180))
                    connected = True
                except:
                    print "failed to connect"
                    last_t = cur_time + 1000
            else:
                try:
                    write_bytes = bytearray()
                    write_bytes.append(byte_to_send)
                    s.send(write_bytes)
                    print byte_to_send
                    last_t = cur_time
                    connected = True
                except socket.timeout:
                    print "timed out"
                    connected = False

        # Flip the image so the the screen is like a mirror
        wc_img = cv.flip(cv.resize(frame, (config["webcam_w"], config["webcam_h"])), 1)
        wc_img_hsv = cv.cvtColor(wc_img, cv.COLOR_BGR2HSV)

        cal_avg = avg_color(wc_img_hsv, (CAL_UL, CAL_LR))

        bg = np.zeros((wc_img.shape[0], wc_img.shape[1] + 300, 3), dtype=np.uint8)
        bg[:, :config["webcam_w"], :] = wc_img

        # draw meter rectangles
        # calibration square
        cv.rectangle(bg, CAL_UL, CAL_LR, (255,255,255), 2)

        # left zero
        LEFT_ZERO_UL = ((config["webcam_w"]/2) - METER_DIST - (HAND_W/2), (METER_TOP + METER_BOTTOM - HAND_H)/2)
        LEFT_ZERO_LR = ((config["webcam_w"]/2) - METER_DIST + (HAND_W/2), (METER_TOP + METER_BOTTOM + HAND_H)/2)
        cv.rectangle(bg, LEFT_ZERO_UL, LEFT_ZERO_LR, (0, 255, 255), 2)

        # right zero
        RIGHT_ZERO_UL = ((config["webcam_w"]/2) + METER_DIST - (HAND_W/2), (METER_TOP + METER_BOTTOM - HAND_H)/2)
        RIGHT_ZERO_LR = ((config["webcam_w"]/2) + METER_DIST + (HAND_W/2), (METER_TOP + METER_BOTTOM + HAND_H)/2)
        cv.rectangle(bg, RIGHT_ZERO_UL, RIGHT_ZERO_LR, (0, 255, 255), 2)

        # left meter
        LEFT_METER_UL = (LEFT_ZERO_UL[0], METER_TOP)
        LEFT_METER_LR = (LEFT_ZERO_LR[0], METER_BOTTOM)
        cv.rectangle(bg, LEFT_METER_UL, LEFT_METER_LR, (255, 255, 0), 2)

        # right meter
        RIGHT_METER_UL = (RIGHT_ZERO_UL[0], METER_TOP)
        RIGHT_METER_LR = (RIGHT_ZERO_LR[0], METER_BOTTOM)
        cv.rectangle(bg, RIGHT_METER_UL, RIGHT_METER_LR, (255, 255, 0), 2)

        power_step = (STRIPE_LENGTH * 2) / float(METER_BOTTOM - METER_TOP)
        power_scale = np.arange(1.0, -1.0, -power_step)

        cal_avg = avg_color(wc_img_hsv, (CAL_UL, CAL_LR))

        left_diffs = []
        right_diffs = []
        for stripe_top in range(METER_TOP, METER_BOTTOM, STRIPE_LENGTH):
            if color_distance(avg_color(wc_img_hsv, ((LEFT_METER_UL[0], stripe_top),
                                                      (LEFT_METER_LR[0], stripe_top+STRIPE_LENGTH))),
                              cal_avg) > 100:
                left_diffs.append(1)
            else:
                left_diffs.append(0)

            if color_distance(avg_color(wc_img_hsv, ((RIGHT_METER_UL[0], stripe_top),
                                                      (RIGHT_METER_LR[0], stripe_top+STRIPE_LENGTH))),
                              cal_avg) > 100:
                right_diffs.append(1)
            else:
                right_diffs.append(0)

        try:
            left_power = np.average(power_scale, weights=left_diffs)
        except ZeroDivisionError:
            left_power = 0

        try:
            right_power = np.average(power_scale, weights=right_diffs)
        except ZeroDivisionError:
            right_power = 0

        #print left_power, right_power

        cv.putText(bg, str(left_power), (650, 50), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255))
        cv.putText(bg, str(right_power), (650, 80), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255))

        b_l = int(round(left_power * 7, 0))
        b_r = int(round(right_power * 7, 0))
        byte_to_send = speeds_to_byte(b_l, b_r)
        print "L {} R {} {} {}".format(b_l, b_r, speeds_to_byte(b_l,b_r), bin(speeds_to_byte(b_l,b_r)))

        cv.imshow(WINDOW_NAME, bg)

        key = cv.waitKey(10) & 255
