import ctypes
import cupy as cp
import cupyx
import tensorrt as trt


class HostDeviceMem:
    def __init__(self, size, dtype):
        self.size = size
        self.dtype = dtype
        self.host = cupyx.empty_pinned(size, dtype)
        self.device = cp.empty(size, dtype)

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

    @property
    def nbytes(self):
        return self.host.nbytes

    @property
    def hostptr(self):
        return self.host.ctypes.data

    @property
    def devptr(self):
        return self.device.data.ptr

    def copy_htod_async(self, stream):
        self.device.data.copy_from_host_async(self.hostptr, self.nbytes, stream)

    def copy_dtoh_async(self, stream):
        self.device.data.copy_to_host_async(self.hostptr, self.nbytes, stream)


class TRTInference:
    # initialize TensorRT
    TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(TRT_LOGGER, '')

    def __init__(self, model, batch_size):
        self.model = model
        self.batch_size = batch_size

        # load plugin if the model requires one
        if self.model.PLUGIN_PATH is not None:
            try:
                ctypes.cdll.LoadLibrary(self.model.PLUGIN_PATH)
            except OSError as err:
                raise RuntimeError('Plugin not found') from err

        # load trt engine or build one if not found
        if not self.model.ENGINE_PATH.exists():
            self.engine = self.model.build_engine(TRTInference.TRT_LOGGER, self.batch_size)
        else:
            runtime = trt.Runtime(TRTInference.TRT_LOGGER)
            with open(self.model.ENGINE_PATH, 'rb') as engine_file:
                self.engine = runtime.deserialize_cuda_engine(engine_file.read())
        if self.engine is None:
            raise RuntimeError('Unable to load the engine file')
        if self.engine.has_implicit_batch_dimension:
            assert self.batch_size <= self.engine.max_batch_size
        self.context = self.engine.create_execution_context()
        self.stream = cp.cuda.Stream()

        # allocate buffers
        self.bindings = []
        self.outputs = []
        self.input = None
        for binding in self.engine:
            shape = self.engine.get_binding_shape(binding)
            size = trt.volume(shape)
            if self.engine.has_implicit_batch_dimension:
                size *= self.batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            # allocate host and device buffers
            buffer = HostDeviceMem(size, dtype)
            # append the device buffer to device bindings
            self.bindings.append(buffer.devptr)
            if self.engine.binding_is_input(binding):
                if not self.engine.has_implicit_batch_dimension:
                    assert self.batch_size == shape[0]
                # expect one input
                self.input = buffer
            else:
                self.outputs.append(buffer)
        assert self.input is not None

        # timing events
        self.start = cp.cuda.Event()
        self.end = cp.cuda.Event()

    def __del__(self):
        if hasattr(self, 'context'):
            self.context.__del__()
        if hasattr(self, 'engine'):
            self.engine.__del__()

    def infer(self):
        self.infer_async()
        return self.synchronize()

    def infer_async(self, from_device=False):
        self.start.record(self.stream)
        if not from_device:
            self.input.copy_htod_async(self.stream)
        if self.engine.has_implicit_batch_dimension:
            self.context.execute_async(batch_size=self.batch_size, bindings=self.bindings,
                                       stream_handle=self.stream.ptr)
        else:
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.ptr)
        for out in self.outputs:
            out.copy_dtoh_async(self.stream)
        self.end.record(self.stream)

    def synchronize(self):
        self.stream.synchronize()
        return [out.host for out in self.outputs]

    def get_infer_time(self):
        self.end.synchronize()
        return cp.cuda.get_elapsed_time(self.start, self.end)
