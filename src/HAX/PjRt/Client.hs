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
, clientBufferFromHostBufferGC
) where
import HAX.PjRt.Plugin
import HAX.PjRt.HostBufferSemantics (HostBufferSemantics(..))
import HAX.PjRt.BufferType (BufferType(..))

import GHC.Exts

import Foreign
import Foreign.C

import Data.Primitive.ByteArray
import MLIR.IR (ByteCode(..))

import Paths_HAX
import GHC.IO (unsafePerformIO)
import System.IO (withBinaryFile, IOMode (ReadMode), hFileSize, hGetBuf)

-- serialized compile operation
-- TODO: Implement a way to config it
{-# NOINLINE compileOptions #-}
compileOptions :: ByteArray
compileOptions = unsafePerformIO (do 
  path <- getDataFileName "serialized_compile_options"
  withBinaryFile path ReadMode (\handle -> do 
    size      <- fromInteger <$> hFileSize handle
    byteArray <- newPinnedByteArray size
    let byteArrayPtr = mutableByteArrayContents byteArray
    _ <- hGetBuf handle byteArrayPtr size
    unsafeFreezeByteArray byteArray))

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


-- foreign import ccall unsafe "client_default_device_assignment"
--   clientDefaultDeviceAssignment :: Ptr Api -> Ptr Client -> CInt -> CInt -> CSize -> Ptr CInt -> IO ()
foreign import ccall unsafe "client_compile"
  c__clientCompile :: Ptr Api -> Ptr Client -> ByteArray# -> CSize -> Ptr Char -> CSize -> IO (Ptr LoadedExecutable)
-- foreign import ccall unsafe "SerializeCompileOptions"
--   serializeCompileOptions :: Ptr CSize -> IO (Ptr ())
clientCompile :: Ptr Api -> Ptr Client -> ByteCode -> IO (Ptr LoadedExecutable)
clientCompile api client (ByteCode (ByteArray code)) = do
  c__clientCompile api client code codesize (castPtr $ byteArrayContents (cloneByteArray compileOptions 0 compileOptionsSize)) compileOptionsSize
  where codesize = fromIntegral $ I# (sizeofByteArray# code)
        compileOptionsSize :: Num a => a
        compileOptionsSize = fromIntegral $ sizeofByteArray compileOptions



foreign import ccall unsafe "client_addressable_devices"
  c__clientAddressableDevices :: Ptr Api -> Ptr Client -> Ptr CSize -> IO (Ptr (Ptr Device))
clientAddressableDevices :: Ptr Api -> Ptr Client -> IO [Ptr Device]
clientAddressableDevices api client = alloca $ \numPtr -> do 
  devices <- c__clientAddressableDevices api client numPtr  
  num     <- peek numPtr 
  peekArray (fromIntegral num) devices

withShapeInfo :: ShapeInfo -> ((Ptr Int64, CSize) -> (Ptr Int64, CSize) -> IO b) -> IO b
withShapeInfo (Shape shape) f = 
  withArrayLen shape $ \ (fromIntegral -> numDims) dims -> 
    f (dims, numDims) (nullPtr, 0)
withShapeInfo (ShapeAndByteStrides (unzip -> (shape, bytestrides))) f = 
  withArrayLen shape $ \ (fromIntegral -> numDims) dims -> 
    withArray bytestrides $ \ bytestridesPtr -> 
      f (dims, numDims) (bytestridesPtr, numDims)

foreign import ccall unsafe "client_buffer_from_host_buffer"
  c__clientBufferFromHostBuffer :: Ptr Api -> Ptr Client -> 
                                   ByteArray# -> BufferType -> 
                                   Ptr Int64 -> CSize -> 
                                   Ptr Int64 -> CSize -> 
                                   HostBufferSemantics -> Ptr Device -> 
                                   Ptr MemoryLayout -> 
                                   Ptr (Ptr Event) -> IO (Ptr Buffer)
clientBufferFromHostBuffer :: Ptr Api -> Ptr Client -> ByteArray -> BufferType -> ShapeInfo -> HostBufferSemantics -> Ptr Device -> IO (Ptr Event, Ptr Buffer)
clientBufferFromHostBuffer api client (ByteArray hostBuffer#) bufferType shapeInfo hostBufferSematics device = 
  withShapeInfo shapeInfo $ \ (dims, numDims) (bytestrides, numBytestrides) -> alloca $ \ eventPtr -> do
    deviceBuffer       <- c__clientBufferFromHostBuffer api client hostBuffer# bufferType dims numDims bytestrides numBytestrides hostBufferSematics device nullPtr eventPtr 
    doneWithHostBuffer <- peek eventPtr
    return (doneWithHostBuffer, deviceBuffer)

-- The memory given must be allocated by malloc and must not be managed by haskell garbage collector
foreign import ccall unsafe "client_buffer_from_host_buffer__gc"
  c__clientBufferFromHostBuffer__gc :: Ptr Api -> Ptr Client -> Ptr a -> BufferType -> Ptr Int64 -> CSize -> Ptr Int64 -> CSize -> Ptr Device -> Ptr MemoryLayout -> IO (Ptr Buffer)
clientBufferFromHostBufferGC :: Ptr Api -> Ptr Client -> Ptr a -> BufferType -> ShapeInfo -> Ptr Device -> Ptr MemoryLayout -> IO (Ptr Buffer)
clientBufferFromHostBufferGC api client content bufferType shapeInfo device memLayout =  
  withShapeInfo shapeInfo $ \ (dims, numDims) (bytestrides, numBytestrides) ->
    c__clientBufferFromHostBuffer__gc api client content bufferType dims numDims bytestrides numBytestrides device memLayout

