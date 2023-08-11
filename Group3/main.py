#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import math
import argparse

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=640)
    parser.add_argument("--height", help='cap height', type=int, default=360)

    parser.add_argument('--static_image_mode', action='store_true')
    parser.add_argument("--model_complexity",
                        help='model_complexity(0,1(default),2)',
                        type=int,
                        default=1)
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.5)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    parser.add_argument('--rev_color', action='store_true')

    args = parser.parse_args()

    return args


def main():
    
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    static_image_mode = args.static_image_mode
    model_complexity = args.model_complexity
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    rev_color = args.rev_color

    
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=static_image_mode,
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    
    if rev_color:
        color = (255, 255, 255)
        bg_color = (100, 33, 3)
    else:
        color = (100, 33, 3)
        bg_color = (255, 255, 255)

    while True:
        display_fps = cvFpsCalc.get()

        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  
        debug_image01 = copy.deepcopy(image)
        debug_image02 = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
        cv.rectangle(debug_image02, (0, 0), (image.shape[1], image.shape[0]),
                    bg_color,
                    thickness=-1)

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks is not None:
            
            debug_image01 = draw_landmarks(
                debug_image01,
                results.pose_landmarks,
            )
            debug_image02 = draw_stick_figure(
                debug_image02,
                results.pose_landmarks,
                color=color,
                bg_color=bg_color,
            )

        cv.putText(debug_image01, "FPS:" + str(display_fps), (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)
        cv.putText(debug_image02, "FPS:" + str(display_fps), (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv.LINE_AA)

        key = cv.waitKey(1)
        if key == 27:  # ESC
            break

        cv.imshow('Tokyo2020 Debug', debug_image01)
        cv.imshow('Tokyo2020 Pictogram', debug_image02)

    cap.release()
    cv.destroyAllWindows()


def draw_stick_figure(
        image,
        landmarks,
        color=(100, 33, 3),
        bg_color=(255, 255, 255),
        visibility_th=0.5,
):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []
    for index, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_z = landmark.z
        landmark_point.append(
            [index, landmark.visibility, (landmark_x, landmark_y), landmark_z])

    right_leg = landmark_point[23]
    left_leg = landmark_point[24]
    leg_x = int((right_leg[2][0] + left_leg[2][0]) / 2)
    leg_y = int((right_leg[2][1] + left_leg[2][1]) / 2)

    landmark_point[23][2] = (leg_x, leg_y)
    landmark_point[24][2] = (leg_x, leg_y)

    sorted_landmark_point = sorted(landmark_point,
                                   reverse=True,
                                   key=lambda x: x[3])

    (face_x, face_y), face_radius = min_enclosing_face_circle(landmark_point)

    face_x = int(face_x)
    face_y = int(face_y)
    face_radius = int(face_radius * 1.5)

    stick_radius01 = int(face_radius * (4 / 5))
    stick_radius02 = int(stick_radius01 * (3 / 4))
    stick_radius03 = int(stick_radius02 * (3 / 4))

   
    draw_list = [
        11, 
        12,  
        23,  
        24,  
    ]

    
    cv.rectangle(image, (0, 0), (image_width, image_height),
                 bg_color,
                 thickness=-1)

    
    cv.circle(image, (face_x, face_y), face_radius, color, -1)

    
    for landmark_info in sorted_landmark_point:
        index = landmark_info[0]

        if index in draw_list:
            point01 = [p for p in landmark_point if p[0] == index][0]
            point02 = [p for p in landmark_point if p[0] == (index + 2)][0]
            point03 = [p for p in landmark_point if p[0] == (index + 4)][0]

            if point01[1] > visibility_th and point02[1] > visibility_th:
                image = draw_stick(
                    image,
                    point01[2],
                    stick_radius01,
                    point02[2],
                    stick_radius02,
                    color=color,
                    bg_color=bg_color,
                )
            if point02[1] > visibility_th and point03[1] > visibility_th:
                image = draw_stick(
                    image,
                    point02[2],
                    stick_radius02,
                    point03[2],
                    stick_radius03,
                    color=color,
                    bg_color=bg_color,
                )

    return image


def min_enclosing_face_circle(landmark_point):
    landmark_array = np.empty((0, 2), int)

    index_list = [1, 4, 7, 8, 9, 10]
    for index in index_list:
        np_landmark_point = [
            np.array(
                (landmark_point[index][2][0], landmark_point[index][2][1]))
        ]
        landmark_array = np.append(landmark_array, np_landmark_point, axis=0)

    center, radius = cv.minEnclosingCircle(points=landmark_array)

    return center, radius


def draw_stick(
        image,
        point01,
        point01_radius,
        point02,
        point02_radius,
        color=(100, 33, 3),
        bg_color=(255, 255, 255),
):
    cv.circle(image, point01, point01_radius, color, -1)
    cv.circle(image, point02, point02_radius, color, -1)

    draw_list = []
    for index in range(2):
        rad = math.atan2(point02[1] - point01[1], point02[0] - point01[0])

        rad = rad + (math.pi / 2) + (math.pi * index)
        point_x = int(point01_radius * math.cos(rad)) + point01[0]
        point_y = int(point01_radius * math.sin(rad)) + point01[1]

        draw_list.append([point_x, point_y])

        point_x = int(point02_radius * math.cos(rad)) + point02[0]
        point_y = int(point02_radius * math.sin(rad)) + point02[1]

        draw_list.append([point_x, point_y])

    points = np.array((draw_list[0], draw_list[1], draw_list[3], draw_list[2]))
    cv.fillConvexPoly(image, points=points, color=color)

    return image


def draw_landmarks(
    image,
    landmarks,
    # upper_body_only,
    visibility_th=0.5,
):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    for index, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_z = landmark.z
        landmark_point.append([landmark.visibility, (landmark_x, landmark_y)])

        if landmark.visibility < visibility_th:
            continue

        if index == 0:  
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 1:  
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 2: 
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 3: 
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 4:  
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 5:  
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 6:  
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 7: 
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 8: 
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 9:  
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 10:  
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 11:  
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 12:  
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 13:  
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 14:  
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 15:  
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 16:  
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 17:  
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 18:  
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 19:  
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 20:  
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 21:  
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 22:  
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 23: 
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 24: 
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 25:  
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 26:  
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 27: 
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 28:  
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 29:  
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 30: 
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 31:  
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 32:  
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)

        # if not upper_body_only:
        if True:
            cv.putText(image, "z:" + str(round(landmark_z, 3)),
                       (landmark_x - 10, landmark_y - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                       cv.LINE_AA)

    
    if landmark_point[1][0] > visibility_th and landmark_point[2][
            0] > visibility_th:
        cv.line(image, landmark_point[1][1], landmark_point[2][1],
                (0, 255, 0), 2)
    if landmark_point[2][0] > visibility_th and landmark_point[3][
            0] > visibility_th:
        cv.line(image, landmark_point[2][1], landmark_point[3][1],
                (0, 255, 0), 2)

    
    if landmark_point[4][0] > visibility_th and landmark_point[5][
            0] > visibility_th:
        cv.line(image, landmark_point[4][1], landmark_point[5][1],
                (0, 255, 0), 2)
    if landmark_point[5][0] > visibility_th and landmark_point[6][
            0] > visibility_th:
        cv.line(image, landmark_point[5][1], landmark_point[6][1],
                (0, 255, 0), 2)

    
    if landmark_point[9][0] > visibility_th and landmark_point[10][
            0] > visibility_th:
        cv.line(image, landmark_point[9][1], landmark_point[10][1],
                (0, 255, 0), 2)

    
    if landmark_point[11][0] > visibility_th and landmark_point[12][
            0] > visibility_th:
        cv.line(image, landmark_point[11][1], landmark_point[12][1],
                (0, 255, 0), 2)

    
    if landmark_point[11][0] > visibility_th and landmark_point[13][
            0] > visibility_th:
        cv.line(image, landmark_point[11][1], landmark_point[13][1],
                (0, 255, 0), 2)
    if landmark_point[13][0] > visibility_th and landmark_point[15][
            0] > visibility_th:
        cv.line(image, landmark_point[13][1], landmark_point[15][1],
                (0, 255, 0), 2)

   
    if landmark_point[12][0] > visibility_th and landmark_point[14][
            0] > visibility_th:
        cv.line(image, landmark_point[12][1], landmark_point[14][1],
                (0, 255, 0), 2)
    if landmark_point[14][0] > visibility_th and landmark_point[16][
            0] > visibility_th:
        cv.line(image, landmark_point[14][1], landmark_point[16][1],
                (0, 255, 0), 2)

  
    if landmark_point[15][0] > visibility_th and landmark_point[17][
            0] > visibility_th:
        cv.line(image, landmark_point[15][1], landmark_point[17][1],
                (0, 255, 0), 2)
    if landmark_point[17][0] > visibility_th and landmark_point[19][
            0] > visibility_th:
        cv.line(image, landmark_point[17][1], landmark_point[19][1],
                (0, 255, 0), 2)
    if landmark_point[19][0] > visibility_th and landmark_point[21][
            0] > visibility_th:
        cv.line(image, landmark_point[19][1], landmark_point[21][1],
                (0, 255, 0), 2)
    if landmark_point[21][0] > visibility_th and landmark_point[15][
            0] > visibility_th:
        cv.line(image, landmark_point[21][1], landmark_point[15][1],
                (0, 255, 0), 2)

    
    if landmark_point[16][0] > visibility_th and landmark_point[18][
            0] > visibility_th:
        cv.line(image, landmark_point[16][1], landmark_point[18][1],
                (0, 255, 0), 2)
    if landmark_point[18][0] > visibility_th and landmark_point[20][
            0] > visibility_th:
        cv.line(image, landmark_point[18][1], landmark_point[20][1],
                (0, 255, 0), 2)
    if landmark_point[20][0] > visibility_th and landmark_point[22][
            0] > visibility_th:
        cv.line(image, landmark_point[20][1], landmark_point[22][1],
                (0, 255, 0), 2)
    if landmark_point[22][0] > visibility_th and landmark_point[16][
            0] > visibility_th:
        cv.line(image, landmark_point[22][1], landmark_point[16][1],
                (0, 255, 0), 2)

    
    if landmark_point[11][0] > visibility_th and landmark_point[23][
            0] > visibility_th:
        cv.line(image, landmark_point[11][1], landmark_point[23][1],
                (0, 255, 0), 2)
    if landmark_point[12][0] > visibility_th and landmark_point[24][
            0] > visibility_th:
        cv.line(image, landmark_point[12][1], landmark_point[24][1],
                (0, 255, 0), 2)
    if landmark_point[23][0] > visibility_th and landmark_point[24][
            0] > visibility_th:
        cv.line(image, landmark_point[23][1], landmark_point[24][1],
                (0, 255, 0), 2)

    if len(landmark_point) > 25:
        
        if landmark_point[23][0] > visibility_th and landmark_point[25][
                0] > visibility_th:
            cv.line(image, landmark_point[23][1], landmark_point[25][1],
                    (0, 255, 0), 2)
        if landmark_point[25][0] > visibility_th and landmark_point[27][
                0] > visibility_th:
            cv.line(image, landmark_point[25][1], landmark_point[27][1],
                    (0, 255, 0), 2)
        if landmark_point[27][0] > visibility_th and landmark_point[29][
                0] > visibility_th:
            cv.line(image, landmark_point[27][1], landmark_point[29][1],
                    (0, 255, 0), 2)
        if landmark_point[29][0] > visibility_th and landmark_point[31][
                0] > visibility_th:
            cv.line(image, landmark_point[29][1], landmark_point[31][1],
                    (0, 255, 0), 2)

        
        if landmark_point[24][0] > visibility_th and landmark_point[26][
                0] > visibility_th:
            cv.line(image, landmark_point[24][1], landmark_point[26][1],
                    (0, 255, 0), 2)
        if landmark_point[26][0] > visibility_th and landmark_point[28][
                0] > visibility_th:
            cv.line(image, landmark_point[26][1], landmark_point[28][1],
                    (0, 255, 0), 2)
        if landmark_point[28][0] > visibility_th and landmark_point[30][
                0] > visibility_th:
            cv.line(image, landmark_point[28][1], landmark_point[30][1],
                    (0, 255, 0), 2)
        if landmark_point[30][0] > visibility_th and landmark_point[32][
                0] > visibility_th:
            cv.line(image, landmark_point[30][1], landmark_point[32][1],
                    (0, 255, 0), 2)
    return image

if __name__ == '__main__':
    main()
