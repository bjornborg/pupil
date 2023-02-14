"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2022 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import os
import sysv_ipc

import abc
import logging
import time
from typing import Iterable, List, Optional, Tuple, Type
from typing_extensions import Literal, NotRequired, TypedDict

# from enum import IntEnum, auto
# from time import monotonic, sleep
import uvc

# import gl_utils
import numpy as np
import numpy.typing as npt

# from plugin import Plugin
from pyglui import cygl, ui
from camera_models import Radial_Dist_Camera
from video_capture.base_backend import (
    Base_Manager,
    Base_Source,
    InitialisationError,
    SourceInfo,
)

from .cluon_shared_memory import SharedMemoryConsumer

logger = logging.getLogger(__name__)


class Uint8BufferFrame(abc.ABC):
    def __init__(
        self,
        buffer: bytes,
        timestamp: float,
        index: int,
        width: int,
        height: int,
    ):
        #
        self._buffer = self.interpret_buffer(buffer, width, height)
        self.timestamp = timestamp
        self.index = index
        self.width = width
        self.height = height
        # indicate that the frame does not have a native yuv or jpeg buffer
        self.yuv_buffer = None
        self.jpeg_buffer = None

    def interpret_buffer(
        self, buffer: bytes, width: int, height: int
    ) -> npt.NDArray[np.uint8]:
        # 2022-11-25 13:32:30 bb | To handle alpha channel
        return np.fromstring(buffer, dtype=np.uint8).reshape(height, width, self.depth)

    @property
    @abc.abstractmethod
    def depth(self) -> int:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    # dtype uint8, shape (height, width)
    def gray(self) -> npt.NDArray[np.uint8]:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def bgr(self) -> npt.NDArray[np.uint8]:
        # dtype uint8, shape (height, width, 3), memory needs to be allocated contiguous
        raise NotImplementedError

    @property
    def img(self) -> npt.NDArray[np.uint8]:
        # equivalent for bgr; kept for legacy reasons
        # 2022-11-25 13:47:26 bb | Removing alpha channel, they assume 3 channels
        # 2022-11-25 13:54:31 bb | Image converters in cython needs contiguous data
        return self.bgr
        # return np.ascontiguousarray(self.bgr[:,:,:3])


class BGRAFrame(Uint8BufferFrame):
    @property
    def depth(self) -> int:
        return 4

    @property
    def bgr(self) -> npt.NDArray[np.uint8]:
        return np.ascontiguousarray(self._buffer[:, :, :3])

    @property
    def gray(self):
        try:
            return self._gray
        except AttributeError:
            self._gray = np.mean(self._buffer, axis=-
                                 1).astype(self._buffer.dtype)
            return self._gray


class YUV420Frame(abc.ABC):
    def __init__(
        self,
        buffer: bytes,
        timestamp: float,
        index: int,
        width: int,
        height: int,
    ):
        #
        self.timestamp = timestamp
        self.index = index
        self.width = width
        self.height = height
        self.yuv_buffer = self.interpret_buffer(buffer, width, height)
        # indicate that the frame does not have a native yuv or jpeg buffer
        self.jpeg_buffer = None

    def interpret_buffer(
        self, buffer: bytes, width: int, height: int
    ) -> npt.NDArray[np.uint8]:
        # 2022-11-25 13:32:30 bb | To handle alpha channel
        return np.fromstring(buffer, dtype=np.uint8)
    # @property
    # # dtype uint8, shape (height, width)
    # def gray(self) -> npt.NDArray[np.uint8]:
    #     raise NotImplementedError

    # @property
    # def bgr(self) -> npt.NDArray[np.uint8]:
    #     # dtype uint8, shape (height, width, 3), memory needs to be allocated contiguous
    #     raise NotImplementedError

    # @property
    # def img(self) -> npt.NDArray[np.uint8]:
    #     # equivalent for bgr; kept for legacy reasons
    #     # 2022-11-25 13:47:26 bb | Removing alpha channel, they assume 3 channels
    #     # 2022-11-25 13:54:31 bb | Image converters in cython needs contiguous data
    #     return self.bgr
        # return np.ascontiguousarray(self.bgr[:,:,:3])

# class YUV420Frame(Uint8BufferFrame):
#     @property
#     def depth(self) -> int:
#         return 3

# @property
# def yuv_buffer()
# @property
# def bgr(self) -> int:
#     raise NotImplementedError
# @property
# def gray(self):
#     try:
#         return self._gray
#     except AttributeError:
#         self._gray = np.mean(self._buffer, axis=-1).astype(self._buffer.dtype)
#         return self._gray
# class RGBFrame(BGRFrame):
#     @property
#     def bgr(self) -> npt.NDArray[np.uint8]:
#         try:
#             return self._bgr
#         except AttributeError:
#             self._bgr = np.ascontiguousarray(np.flip(self._buffer, (0, 2)))
#             return self._bgr

#     @property
#     def gray(self):
#         try:
#             return self._gray
#         except AttributeError:
#             self._gray = np.mean(self._buffer, axis=-1).astype(self._buffer.dtype)
#             return self._gray


class Shared_Memory(Base_Source):
    """
    Shared memory consumer for reading image data in /tmp

    Abstract source class

    All source objects are based on `Base_Source`.

    A source object is independent of its matching manager and should be
    initialisable without it.

    Initialization is required to succeed. In case of failure of the underlying capture
    the follow properties need to be readable:

    - name
    - frame_rate
    - frame_size

    The recent_events function is allowed to not add a frame to the `events` object.

    Attributes:
        g_pool (object): Global container, see `Plugin.g_pool`
    """

    def __init__(
        self,
        g_pool,
        key=None,
        *args,
        **kwargs,
    ):
        super().__init__(g_pool, *args, **kwargs)
        self.fps = 0
        self.healthy = True
        self.frame_size = (1440, 1600)
        self.startUnixTimestampNs = 0
        self.previousTimestampUnixNs = 0
        self.syncedPupilTime = (
            self.g_pool.get_timestamp()
        )  # add this to normalised time
        if key:
            self.shm = SharedMemoryConsumer(name=key)
            self.shm.read(1)
            self.startUnixTimestampNs = self.shm.currentUnixTimestampNs
            self.previousTimestampUnixNs = self.shm.currentUnixTimestampNs
        else:
            self.shm = None
            self.healthy = False
        self.projection_matrix = np.array(
            [
                [500, 0, self.frame_size[0] / 2],
                [0, 500, self.frame_size[1] / 2],
                [0, 0, 1]
            ]
        )
        # 2023-02-14 15:08:35 bb | where (k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4[,τx,τy]]]]) are the distortion coefficients vector distCoeffs.
        self.distortion_coeffs = np.array(
            [-0.20518164004611725, -0.040357307230613204, 0, 0, -0.022284161177407984]
        )
        self._intrinsics = Radial_Dist_Camera(
            key, self.frame_size, self.projection_matrix, self.distortion_coeffs
        )  # 2022-11-25 14:49:44 bb | We dont have support for this atm
        # self.g_pool.capture = self
        # self._recent_frame = None
        # Three relevant cases for initializing source_mode:
        #   - Plugin started at runtime: use existing source mode in g_pool
        #   - Fresh start without settings: initialize to auto
        #   - Start with settings: will be passed as parameter, use those

    # 2022-11-24 12:21:41 bb | This is main function that is called for every frame grab
    def recent_events(self, events):
        if not self.healthy:
            time.sleep(0.02)
            return

        frame = self.get_frame()
        if frame:
            events["frame"] = frame
            self._recent_frame = frame

    def get_frame(self):
        if self.shm:
            try:
                buf = self.shm.read(timeout=1)  # timout in seconds
                # 2022-11-25 11:02:45 bb | Pupil labs (PyAv) does not support alpha channel
                # 2022-11-25 13:09:55 bb | remove the alpha channel
            except sysv_ipc.ExistentialError:
                # Error
                logger.error("sysv_ipc.ExistentialError")
                self.healthy = False
                self._recent_frame = None
            except KeyError as err:
                logger.debug(
                    f"Ill-formatted frame received. Missing key: {err}")

            if self.shm.currentUnixTimestampNs == self.previousTimestampUnixNs:
                self.healthy = False
                return
            self.fps = 10**9 / (
                self.shm.currentUnixTimestampNs - self.previousTimestampUnixNs
            )
            self.previousTimestampUnixNs = self.shm.currentUnixTimestampNs
            puplTimestampSeconds = (
                self.syncedPupilTime
                + (self.shm.currentUnixTimestampNs - self.startUnixTimestampNs)
                / 10**9
            )
            if self.frame_size[0] * self.frame_size[1] * 2 != len(buf):
                logger.error(
                    "Incorrect image dimension:{},{}. Size of buf {}".format(
                        self.frame_size[0], self.frame_size[1], len(buf)
                    )
                )
                self.healthy = False
            else:
                return YUV420Frame(
                    buf,
                    puplTimestampSeconds,
                    self.shm.index(),
                    self.frame_size[0],
                    self.frame_size[1],
                )

    @property
    def name(self):
        if self.shm:
            return self.shm.name()
        else:
            return "Null"

    @property
    def frame_size(self):
        return self._frame_size

    @frame_size.setter
    def frame_size(self, size):
        if size[0] < 1 or size[1] < 1:
            logger.error("Setting incorrect image size")
        self.healthy = True
        self._frame_size = size

    @property
    def frame_rate(self):
        return self.fps

    @property
    def jpeg_support(self):
        return False

    @property
    def healthy(self):
        return self._healthy

    @healthy.setter
    def healthy(self, val):
        self._healthy = val

    @property
    def online(self):
        return self._recent_frame is not None

    @property
    def intrinsics(self):
        return self._intrinsics

    def cleanup(self):
        if self.shm:
            self.shm.cleanup()
        self.shm = None

    def ui_elements(self):
        ui_elements = []
        ui_elements.append(ui.Info_Text(f"Image size"))

        def set_frame_size(new_size):
            self.frame_size = [int(ele) for ele in new_size.split(",")]

        ui_elements.append(ui.Text_Input(
            "frame_size", self, label="(width,height)"))

        return ui_elements


class Shared_Memory_Manager(Base_Manager):
    """Manages local shared memory sources."""

    def __init__(self, g_pool):
        super().__init__(g_pool)
        self.sourceList = []

    def get_devices(self):
        # self.updateSourceList()
        # if len(self.sourceList) == 0:
        return []
        # else:
        #     return [SourceInfo(label=f"{source} @ Local shared memory", manager=self, key=f"shm.{source}") for source in self.sourceList]

    def get_cameras(self):
        self.updateSourceList()
        if len(self.sourceList) == 0:
            return []
        else:
            return [
                SourceInfo(
                    label=f"{source} @ Local shared memory",
                    manager=self,
                    key=f"shm.{source}",
                )
                for source in self.sourceList
            ]

    def activate(self, key):
        source_key = key[4:]
        self.activate_source(source_key)

    def activate_source(self, source_key):
        if not source_key:
            return

        settings = {
            "key": source_key,
        }
        if self.g_pool.process == "world":
            self.notify_all(
                {"subject": "start_plugin", "name": "Shared_Memory", "args": settings}
            )
        else:
            return

    def updateSourceList(self):
        self.sourceList = [
            os.path.join(r"/tmp", file)
            for file in os.listdir(r"/tmp")
            # if file.endswith(".argb")
            if file.endswith(".i422")
        ]

    def cleanup(self):
        # self.devices.cleanup()
        self.shmList = None
