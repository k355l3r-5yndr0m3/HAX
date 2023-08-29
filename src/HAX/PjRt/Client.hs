{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE MagicHash #-}
{-# LANGUAGE UnliftedFFITypes #-}
{-# LANGUAGE ViewPatterns #-}
module HAX.PjRt.Client (
  clientCreate 
, clientDestroy
, clientDestroy__ptr
, clientPlatformName
-- , clientDefaultDeviceAssignment
, clientCompile
, clientAddressableDevices
, clientBufferFromHostBuffer
) where
import HAX.PjRt.Plugin
import HAX.PjRt.HostBufferSemantics (HostBufferSemantics(..))
import HAX.PjRt.BufferType (BufferType(..))

import GHC.Exts

import Foreign
import Foreign.C

import Data.Primitive.ByteArray
import MLIR.IR (ByteCode(..))


foreign import ccall unsafe "client_create"
  clientCreate :: Ptr Api -> IO (Ptr Client)


foreign import ccall unsafe "client_destroy"
  clientDestroy :: Ptr Api -> Ptr Client -> IO ()
foreign import ccall unsafe "&client_destroy"
  clientDestroy__ptr :: FinalizerEnvPtr Api Client


foreign import ccall unsafe "client_platform_name"
  c__clientPlatformName :: Ptr Api -> Ptr Client -> Ptr CSize -> IO CString
clientPlatformName :: Ptr Api -> Ptr Client -> IO String
clientPlatformName api client = alloca $ \sizePtr -> do 
  name <- c__clientPlatformName api client sizePtr 
  size <- fromIntegral <$> peek sizePtr 
  peekCStringLen (name, size)


foreign import ccall unsafe "client_default_device_assignment"
  clientDefaultDeviceAssignment :: Ptr Api -> Ptr Client -> CInt -> CInt -> CSize -> Ptr CInt -> IO ()


foreign import ccall unsafe "client_compile"
  c__clientCompile :: Ptr Api -> Ptr Client -> ByteArray# -> CSize -> Ptr Char -> CSize -> IO (Ptr LoadedExecutable)
foreign import ccall unsafe "SerializeCompileOptions"
  serializeCompileOptions :: Ptr CSize -> IO (Ptr ())
clientCompile :: Ptr Api -> Ptr Client -> ByteCode -> IO (Ptr LoadedExecutable)
clientCompile api client (ByteCode (ByteArray code)) = do
  (a, b) <- alloca $ \ptr -> do 
    ops <- serializeCompileOptions ptr
    (ops, ) <$> peek ptr
  exe <- c__clientCompile api client code codesize (castPtr a) b
  free a
  return exe
  where codesize = fromIntegral $ I# (sizeofByteArray# code)


foreign import ccall unsafe "client_addressable_devices"
  c__clientAddressableDevices :: Ptr Api -> Ptr Client -> Ptr CSize -> IO (Ptr (Ptr Device))
clientAddressableDevices :: Ptr Api -> Ptr Client -> IO [Ptr Device]
clientAddressableDevices api client = alloca $ \numPtr -> do 
  devices <- c__clientAddressableDevices api client numPtr  
  num     <- peek numPtr 
  peekArray (fromIntegral num) devices


foreign import ccall unsafe "client_buffer_from_host_buffer"
  c__clientBufferFromHostBuffer :: Ptr Api -> Ptr Client -> 
                                   Ptr a -> BufferType -> 
                                   Ptr Int64 -> CSize -> 
                                   Ptr Int64 -> CSize -> 
                                   HostBufferSemantics -> Ptr Device -> 
                                   Ptr MemoryLayout -> 
                                   Ptr (Ptr Event) -> IO (Ptr Buffer)
clientBufferFromHostBuffer :: Ptr Api -> Ptr Client -> Ptr a -> BufferType -> ShapeInfo -> HostBufferSemantics -> Ptr Device -> IO (Ptr Event, Ptr Buffer)
clientBufferFromHostBuffer api client hostBuffer bufferType shapeInfo hostBufferSematics device = 
  withShapeInfo shapeInfo $ \ (dims, numDims) (bytestrides, numBytestrides) -> alloca $ \ eventPtr -> do
    deviceBuffer       <- c__clientBufferFromHostBuffer api client hostBuffer bufferType dims numDims bytestrides numBytestrides hostBufferSematics device nullPtr eventPtr 
    doneWithHostBuffer <- peek eventPtr
    return (doneWithHostBuffer, deviceBuffer)
  where withShapeInfo :: ShapeInfo -> ((Ptr Int64, CSize) -> (Ptr Int64, CSize) -> IO b) -> IO b
        withShapeInfo (Shape shape) f = 
          withArrayLen shape $ \ (fromIntegral -> numDims) dims -> 
            f (dims, numDims) (nullPtr, 0)
        withShapeInfo (ShapeAndByteStrides (unzip -> (shape, bytestrides))) f = 
          withArrayLen shape $ \ (fromIntegral -> numDims) dims -> 
            withArray bytestrides $ \ bytestridesPtr -> 
              f (dims, numDims) (bytestridesPtr, numDims)
