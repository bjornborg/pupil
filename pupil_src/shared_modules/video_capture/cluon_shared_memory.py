import os
import sysv_ipc
import logging

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class SharedMemoryConsumer:
  def __init__(self, name): 
    self.name = name
    self.keySharedMemory = sysv_ipc.ftok(name, 1, True)
    self.keySemMutex = sysv_ipc.ftok(name, 2, True)
    self.keySemCondition = sysv_ipc.ftok(name, 3, True)
    if (self.keySemMutex == -1 or self.keySharedMemory == -1 or self.keySemCondition == -1):
      raise ValueError('Could not create a file at{}'.format(self.name))
    
    # For timekeeping 
    self.fd = os.open(self.name, os.O_RDONLY)
    self.fstatResult = None
    self.currentUnixTimestampNs = 0
    
    # Instantiate the SharedMemory and Semaphore objects.
    self.shm = sysv_ipc.SharedMemory(self.keySharedMemory)
    self.mutex = sysv_ipc.Semaphore(self.keySemMutex)
    self.cond = sysv_ipc.Semaphore(self.keySemCondition)
      
    self.frameNo = 0
    logger.info("[Shared memory consumer] {}: SysV initiated.".format(self.name))
    
  def currentTimestamp(self):
    return self.currentUnixTimestampNs
  
  def name(self):
    return self.name
  
  def index(self):
    return self.frameNo
    
  def read(self,timeout=None):
      self.cond.Z(timeout)
      # Lock access to shared memory.
      self.mutex.acquire()
      # Attach to shared memory.
      self.shm.attach()
      # Read shared memory into own buffer.
      buf = self.shm.read()
      self.fstatResult = os.fstat(self.fd)
      # Detach to shared memory.
      self.shm.detach()
      # Unlock access to shared memory.
      self.mutex.release()
      
      self.currentUnixTimestampNs = self.fstatResult.st_mtime_ns
      self.frameNo += 1
      return buf
      
  def cleanup(self):
    os.close(self.fd)
    logger.info("[Shared memory consumer] Cleaned up " + str(self.name) + ". Read in total " + str(self.frameNo) + " frames.")


class SharedMemoryProducer:
  def __init__(self, name, logger, size):
    # atexit.register(self.cleanup)
    # super().__init__()
    self.name = name
    touch(self.name)
    self.keySharedMemory = sysv_ipc.ftok(self.name, 1, True)
    self.keySemMutex = sysv_ipc.ftok(self.name, 2, True)
    self.keySemCond = sysv_ipc.ftok(self.name, 3, True)
    if (self.keySemMutex == -1 or self.keySharedMemory == -1 or self.keySemCond == -1):
      raise ValueError('Could not create a file at{}'.format(self.name))
    # keySemCondition = sysv_ipc.ftok(self.name, 3, True)
    logger.info('''[Shared memory producer] The keys ({},{}) created for {}'''.format(self.keySharedMemory, self.keySemMutex, self.name))

    # Instantiate the SharedMemory and Semaphore objects.
    try:
      semaphore = sysv_ipc.Semaphore(self.keySemMutex)
    except sysv_ipc.ExistentialError:
      logger.debug('''[Shared memory producer] The semaphore with key "{}" doesn't exist. Creating one...'''.format(self.keySemMutex))
    else:
      semaphore.remove()
      logger.debug('[Shared memory producer] Removed the semaphore with key "{}".'.format(semaphore.id))
    finally:
      self.mutex = sysv_ipc.Semaphore(self.keySemMutex, sysv_ipc.IPC_CREX)

    try:
      memory = sysv_ipc.SharedMemory(self.keySharedMemory)
    except sysv_ipc.ExistentialError:
      logger.debug('''[Shared memory producer] The shared memory with key "{}" doesn't exist. Creating one...'''.format(self.keySharedMemory))
    else:
      memory.remove()
      logger.info('[Shared memory producer] Removed the shared memory with key "{}".'.format(memory.id))
    finally:
      self.shm = sysv_ipc.SharedMemory(self.keySharedMemory, sysv_ipc.IPC_CREX, size=size)

    try:
      cond = sysv_ipc.Semaphore(self.keySemCond)
    except sysv_ipc.ExistentialError:
      logger.debug('''[Shared memory producer] The cond semaphore with key "{}" doesn't exist. Creating one...'''.format(self.keySemCond))
    else:
      cond.remove()
      logger.info('[Shared memory producer] Removed the semaphore with key "{}".'.format(cond.id))
    finally:
      self.cond = sysv_ipc.Semaphore(self.keySemCond, sysv_ipc.IPC_CREX)
      
    self.shm.attach()
    self.mutex.release()
    self.cond.release()
    # self.cond.acquire()
    self.frameNo = 0
    logger.info("[Shared memory producer] SysV initiated.")

  # Credits: https://stackoverflow.com/questions/1158076/implement-touch-using-python
  def touch(fname, mode=0o666, dir_fd=None, **kwargs):
    flags = os.O_CREAT | os.O_APPEND
    with os.fdopen(os.open(fname, flags=flags, mode=mode, dir_fd=dir_fd)) as f:
        os.utime(f.fileno() if os.utime in os.supports_fd else fname,
          dir_fd=None if os.supports_fd else dir_fd, **kwargs)

  # def __del__(self):
  #   self.cleanup()

  def cleanup(self):
    # sysv_ipc.remove_semaphore(self.mutex.id)
    # sysv_ipc.remove_shared_memory(self.shm.id)
    self.cond.acquire()
    self.mutex.acquire()
    self.shm.detach()
    sysv_ipc.remove_shared_memory(self.shm.id)
    # self.cond.remove()
    # self.mutex.remove()
    # self.shm.remove()
    sysv_ipc.remove_semaphore(self.cond.id)
    sysv_ipc.remove_semaphore(self.mutex.id)
    os.remove(self.name)
    logger.info("[Shared memory producer] Cleaned up " + str(self.name) + ". Wrote in total " + str(self.frameNo) + " frames.")



  def write(self, data):
    # logger.info("[Shared memory producer] Trying to push data.")
    self.mutex.acquire()
    self.shm.write(data)
    self.mutex.release()
    self.cond.acquire(0)
    self.cond.release()
    self.frameNo += 1
    touch(self.name)
    # logger.info("[Shared memory producer] Successfully pushed data. nFrame: " + str(self.frameNo))
    