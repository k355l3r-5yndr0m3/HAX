{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE MagicHash #-}
{-# LANGUAGE UnliftedFFITypes #-}
module HAX.PjRt.Buffer (
--  bufferDestroy 
  bufferDestroy__ptr
, bufferToHostBuffer
, bufferToHostBuffer'
, bufferDimensions
, bufferElementType
) where

import HAX.PjRt.Plugin

import Foreign
import Foreign.C

import GHC.Exts
import Data.Primitive.ByteArray
import HAX.PjRt.BufferType (BufferType (BufferType))

-- foreign import ccall unsafe "buffer_destroy"
--   bufferDestroy :: Ptr Api -> Ptr Buffer -> IO ()
foreign import ccall unsafe "&buffer_destroy"
  bufferDestroy__ptr :: FinalizerEnvPtr Api Buffer

foreign import ccall unsafe "buffer_to_host_buffer__get_dst_size"
  c__bufferToHostBuffer__getDstSize :: Ptr Api -> Ptr Buffer -> Ptr MemoryLayout -> IO CSize
foreign import ccall unsafe "buffer_to_host_buffer__event_await"
  c__bufferToHostBuffer__eventAwait :: Ptr Api -> Ptr Buffer -> Ptr MemoryLayout -> MutableByteArray# RealWorld -> CSize -> IO () 
foreign import ccall unsafe "buffer_to_host_buffer__event_await"
  c__bufferToHostBuffer__eventAwait' :: Ptr Api -> Ptr Buffer -> Ptr MemoryLayout -> Ptr a -> CSize -> IO () 

bufferToHostBuffer :: Ptr Api -> Ptr Buffer -> IO ByteArray
bufferToHostBuffer api buffer = do 
  dstSize               <- c__bufferToHostBuffer__getDstSize api buffer nullPtr
  MutableByteArray dst# <- newByteArray $ fromIntegral dstSize
  c__bufferToHostBuffer__eventAwait api buffer nullPtr dst# dstSize 
  unsafeFreezeByteArray (MutableByteArray dst#)

bufferToHostBuffer' :: Ptr Api -> Ptr Buffer -> IO (Int, Ptr a)
bufferToHostBuffer' api buffer = do 
  bufferSize <- c__bufferToHostBuffer__getDstSize api buffer nullPtr
  hostBuffer <- mallocBytes $ fromIntegral bufferSize
  c__bufferToHostBuffer__eventAwait' api buffer nullPtr hostBuffer bufferSize
  return (fromIntegral bufferSize, hostBuffer)

foreign import ccall unsafe "buffer_dimensions"
  c__bufferDimensions :: Ptr Api -> Ptr Buffer -> Ptr CSize -> IO (Ptr Int64)
bufferDimensions :: Ptr Api -> Ptr Buffer -> IO [Int64]
bufferDimensions api buffer = do 
  alloca $ \ size -> do
    dims <- c__bufferDimensions api buffer size 
    _size <- peek size 
    peekArray (fromIntegral _size) dims

foreign import ccall unsafe "buffer_element_type"
  c__bufferElementType :: Ptr Api -> Ptr Buffer -> IO Int32
bufferElementType :: Ptr Api -> Ptr Buffer -> IO BufferType
bufferElementType api buffer = BufferType <$> c__bufferElementType api buffer
