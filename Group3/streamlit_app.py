import copy
from multiprocessing import Queue, Process
from typing import NamedTuple, List

import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer, WebRtcMode, ClientSettings

import av
import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from main import draw_landmarks, draw_stick_figure

from fake_objects import FakeResultObject, FakeLandmarksObject, FakeLandmarkObject


_SENTINEL_ = "_SENTINEL_"
st.markdown("<h1 style='text-align: center;'>Full body generation</h1>", unsafe_allow_html=True)
def pose_process(
    in_queue: Queue,
    out_queue: Queue,
    static_image_mode,
    model_complexity,
    min_detection_confidence,
    min_tracking_confidence,
):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=static_image_mode,
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    while True:
        input_item = in_queue.get(timeout=10)
        if isinstance(input_item, type(_SENTINEL_)) and input_item == _SENTINEL_:
            break

        results = pose.process(input_item)
        picklable_results = FakeResultObject(pose_landmarks=FakeLandmarksObject(landmark=[
            FakeLandmarkObject(
                x=pose_landmark.x,
                y=pose_landmark.y,
                z=pose_landmark.z,
                visibility=pose_landmark.visibility,
            ) for pose_landmark in results.pose_landmarks.landmark
        ]))
        out_queue.put_nowait(picklable_results)


class Tokyo2020PictogramVideoProcessor(VideoProcessorBase):
    def __init__(self, static_image_mode,
                    model_complexity,
                    min_detection_confidence,
                    min_tracking_confidence,
                    rev_color,
                    display_mode,
                    show_fps) -> None:
        self._in_queue = Queue()
        self._out_queue = Queue()
        self._pose_process = Process(target=pose_process, kwargs={
            "in_queue": self._in_queue,
            "out_queue": self._out_queue,
            "static_image_mode": static_image_mode,
            "model_complexity": model_complexity,
            "min_detection_confidence": min_detection_confidence,
            "min_tracking_confidence": min_tracking_confidence,
        })
        self._cvFpsCalc = CvFpsCalc(buffer_len=10)

        self.rev_color = rev_color
        self.display_mode = display_mode
        self.show_fps = show_fps

        self._pose_process.start()

    def _infer_pose(self, image):
        self._in_queue.put_nowait(image)
        return self._out_queue.get(timeout=10)

    def _stop_pose_process(self):
        self._in_queue.put_nowait(_SENTINEL_)
        self._pose_process.join(timeout=10)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        display_fps = self._cvFpsCalc.get()

        # Color Designation
        if self.rev_color:
            color = (255, 255, 255)
            bg_color = (100, 33, 3)
        else:
            color = (100, 33, 3)
            bg_color = (255, 255, 255)

        # Camera Capture #####################################################
        image = frame.to_ndarray(format="bgr24")

        image = cv.flip(image, 1)  # ミラー表示
        debug_image01 = copy.deepcopy(image)
        debug_image02 = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
        cv.rectangle(debug_image02, (0, 0), (image.shape[1], image.shape[0]),
                    bg_color,
                    thickness=-1)


        # Conduct detection #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = self._infer_pose(image)
        # results = self._pose.process(image)

        # Drawing ################################################################
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

        if self.show_fps:
            cv.putText(debug_image01, "FPS:" + str(display_fps), (10, 30),
                    cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)
            cv.putText(debug_image02, "FPS:" + str(display_fps), (10, 30),
                    cv.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv.LINE_AA)

        if self.display_mode == "Pose":
            return av.VideoFrame.from_ndarray(debug_image01, format="bgr24")
        elif self.display_mode == "Pictogram":
            return av.VideoFrame.from_ndarray(debug_image02, format="bgr24")
        elif self.display_mode == "Both":
            new_image = np.zeros(image.shape, dtype=np.uint8)
            h, w = image.shape[0:2]
            half_h = h // 2
            half_w = w // 2

            offset_y = h // 4
            new_image[offset_y: offset_y + half_h, 0: half_w, :] = cv.resize(debug_image02, (half_w, half_h))
            new_image[offset_y: offset_y + half_h, half_w:, :] = cv.resize(debug_image01, (half_w, half_h))
            return av.VideoFrame.from_ndarray(new_image, format="bgr24")

    def __del__(self):
        print("Stop the inference process...")
        self._stop_pose_process()
        print("Stopped!")


def main():
    with st.expander("Model parameters (there parameters are effective only at initialization)"):
        
        static_image_mode = "Static image mode"
        model_complexity = 0
        min_detection_confidence = 0.5
        min_tracking_confidence = 0.5

    rev_color = st.checkbox("Reverse color")
    display_mode = st.radio("Display mode", ["Pictogram", "Pose", "Both"], index=0)
    show_fps = st.checkbox("Show FPS", value=True)
    

    def processor_factory():
        return Tokyo2020PictogramVideoProcessor(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            rev_color=rev_color,
            display_mode=display_mode,
            show_fps=show_fps
        )

    webrtc_ctx = webrtc_streamer(
        key="tokyo2020-Pictogram",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=processor_factory,
    )
    st.session_state["started"] = webrtc_ctx.state.playing

    if webrtc_ctx.video_processor:
        webrtc_ctx.video_processor.rev_color = rev_color
        webrtc_ctx.video_processor.display_mode = display_mode
        webrtc_ctx.video_processor.show_fps = show_fps


if __name__ == "__main__":
    main()
