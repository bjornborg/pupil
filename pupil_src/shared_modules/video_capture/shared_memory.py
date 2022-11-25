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
from camera_models import Dummy_Camera
from video_capture.base_backend import Base_Manager, Base_Source, InitialisationError, SourceInfo

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
        return  np.fromstring(buffer, dtype=np.uint8).reshape(height, width, self.depth)

    @property
    @abc.abstractmethod
    def depth(self) -> int:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def gray(self) -> npt.NDArray[np.uint8]:  # dtype uint8, shape (height, width)
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
        return np.ascontiguousarray(self.bgr[:,:,:3])


class BGRFrame(Uint8BufferFrame):
    @property
    def depth(self) -> int:
        return 4

    @property
    def bgr(self) -> npt.NDArray[np.uint8]:
        return self._buffer

    @property
    def gray(self):
        try:
            return self._gray
        except AttributeError:
            self._gray = np.mean(self._buffer, axis=-1).astype(self._buffer.dtype)
            return self._gray


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


# class GrayFrame(Uint8BufferFrame):
#     @property
#     def depth(self) -> int:
#         return 1

#     @property
#     def bgr(self) -> npt.NDArray[np.uint8]:
#         try:
#             return self._bgr
#         except AttributeError:
#             self._bgr = np.ascontiguousarray(np.dstack([self._buffer] * 3))
#             return self._bgr

#     @property
#     def gray(self):
#         return self._buffer

#     def interpret_buffer(
#         self, buffer: bytes, width: int, height: int
#     ) -> npt.NDArray[np.uint8]:
#         array = super().interpret_buffer(buffer, width, height)
#         # since this will be our gray buffer, we need to get rid of our third dimension
#         array.shape = height, width
#         return array


# FRAME_CLASS_BY_FORMAT = {"rgb": RGBFrame, "bgr": BGRFrame, "gray": GrayFrame}

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
        self.startUnixTimestampNs = 0
        self.previousTimestampUnixNs = 0
        self.syncedPupilTime = self.g_pool.get_timestamp() #add this to normalised time
        if key:
          self.shm = SharedMemoryConsumer(name=key)
          self.shm.read(1)
          self.startUnixTimestampNs = self.shm.currentUnixTimestampNs
          self.previousTimestampUnixNs = self.shm.currentUnixTimestampNs
        else:
          self.shm = None
        
        self._intrinsics = Dummy_Camera(key, self.frame_size) # 2022-11-25 14:49:44 bb | We dont have support for this atm 
        # self.g_pool.capture = self
        # self._recent_frame = None
        # Three relevant cases for initializing source_mode:
        #   - Plugin started at runtime: use existing source mode in g_pool
        #   - Fresh start without settings: initialize to auto
        #   - Start with settings: will be passed as parameter, use those

    # 2022-11-24 12:21:41 bb | This is main function that is called for every frame grab  
    def recent_events(self, events):
        frame = self.get_frame()
        if frame:
            events["frame"] = frame
            self._recent_frame = frame
            
    
    def get_frame(self):
        if self.shm:
          try:
            buf = self.shm.read(timeout=1) # timout in seconds
            # 2022-11-25 11:02:45 bb | Pupil labs (PyAv) does not support alpha channel
            # width,height = self.frame_size()
            # channels = 4
            # imgBGR = np.frombuffer(buf, np.uint8).reshape(self.frame_size[1], self.frame_size[0], channels)[:,:,:3:-1]
            # frame: SerializedFrame = None
            # frame_class = FRAME_CLASS_BY_FORMAT["bgr"]
            # 2022-11-25 13:09:55 bb | remove the alpha channel
            
            self.fps = 10**9/(self.shm.currentUnixTimestampNs - self.previousTimestampUnixNs)
            self.previousTimestampUnixNs = self.shm.currentUnixTimestampNs
            puplTimestampSeconds = self.syncedPupilTime + (self.shm.currentUnixTimestampNs - self.startUnixTimestampNs)/10**9
            return BGRFrame(buf,puplTimestampSeconds,self.shm.index(),self.frame_size[0], self.frame_size[1])
          except sysv_ipc.ExistentialError:
            # Error
            logger.error("sysv_ipc.ExistentialError")
            self.shm.cleanup()
            self.shm = None
            self._recent_frame = None
          except KeyError as err:
              logger.debug(f"Ill-formatted frame received. Missing key: {err}")

        # if self.frame_sub.socket.poll(timeout=50):  # timeout in ms (50ms -> 20fps)
        #     num_frames_dropped = -1
        #     frame: SerializedFrame = None
        #     while self.frame_sub.new_data:  # drop all but the newest frame
        #         num_frames_dropped += 1
        #         frame = self.frame_sub.recv()[1]
        #     if num_frames_dropped and self.g_pool.process == "world":
        #         logger.debug(f"Number of dropped frames: {num_frames_dropped}")

        #     try:
        #         frame_format = frame["format"]
        #         if frame_format in FRAME_CLASS_BY_FORMAT:
        #             frame_class = FRAME_CLASS_BY_FORMAT[frame_format]
        #             return self._process_frame(frame_class, frame)
        #     except KeyError as err:
        #         logger.debug(f"Ill-formatted frame received. Missing key: {err}")
 
 
    @property
    def name(self): 
      if self.shm:
        return self.shm.name()
      else:
        return "Null"
      
    @property
    def frame_size(self):
      return (1280,720)
        # return (
        #     (self._recent_frame.width, self._recent_frame.height)
        #     if self._recent_frame
        #     # else (1440,1600) # 2022-11-24 11:08:23 bb | Htc vive pro, one eye
        #     else (1280,720) # 2022-11-24 14:17:49 bb | to be removed
        # )

    @property
    def frame_rate(self):
        return self.fps
      
    @property
    def jpeg_support(self):
        return False
    
    
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



# class Base_Manager(Plugin):
#     """Abstract base class for source managers.

#     Managers are plugins that enumerate and load accessible sources from different
#     backends, e.g. locally USB-connected cameras.

#     Supported sources can be either single cameras or whole devices. Identification and
#     activation of sources works via SourceInfo (see below).
#     """

#     # backend managers are always loaded and need to be loaded before the sources
#     order = -1

#     def __init__(self, g_pool):
#         super().__init__(g_pool)

#         # register all instances in g_pool.source_managers list
#         if not hasattr(g_pool, "source_managers"):
#             g_pool.source_managers = []

#         if self not in g_pool.source_managers:
#             g_pool.source_managers.append(self)

#     def get_devices(self) -> T.Sequence["SourceInfo"]:
#         """Return source infos for all devices that the backend supports."""
#         return []

#     def get_cameras(self) -> T.Sequence["SourceInfo"]:
#         """Return source infos for all cameras that the backend supports."""
#         return []

#     def activate(self, key: T.Any) -> None:
#         """Activate a source (device or camera) by key from source info."""
#         pass

class Shared_Memory_Manager(Base_Manager):
    """Manages local shared memory sources."""

    def __init__(self, g_pool):
        super().__init__(g_pool)
        self.sourceList = []

    def get_devices(self):
        self.updateSourceList()
        if len(self.sourceList) == 0:
            return []
        else:
            return [SourceInfo(label=f"{source} @ Local shared memory", manager=self, key=f"shm.{source}") for source in self.sourceList]

    def get_cameras(self):
        self.updateSourceList()
        if len(self.sourceList) == 0:
            return []
        else:
            return [SourceInfo(label=f"{source} @ Local shared memory", manager=self, key=f"shm.{source}") for source in self.sourceList]


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
        self.sourceList = [os.path.join(r'/tmp', file) for file in os.listdir(r'/tmp') if file.endswith(".argb")]
        

    def cleanup(self):
        # self.devices.cleanup()
        self.shmList = None


# class SourceInfo:
#     """SourceInfo is a proxy for a source (camera or device) from a manager.

#     Managers hand out source infos that can be activated from other places in the code.
#     A manager needs to identify a source uniquely by a key.
#     """

#     def __init__(self, label: str, manager: Base_Manager, key: T.Any):
#         self.label = label
#         self.manager = manager
#         self.key = key

#     def activate(self) -> None:
#         self.manager.activate(self.key)

#     def __str__(self) -> str:
#         return f"{self.label} - {self.manager.class_name}({self.key})"

