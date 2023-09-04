{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE MagicHash #-}
{-# LANGUAGE UnliftedFFITypes #-}
module HAX.PjRt.Buffer (
  bufferDestroy__ptr
, bufferToHostBuffer
) where

import HAX.PjRt.Plugin

import Foreign
import Foreign.C

import GHC.Exts
import Data.Primitive.ByteArray

-- foreign import ccall unsafe "buffer_destroy"
--   bufferDestroy :: Ptr Api -> Ptr Buffer -> IO ()
foreign import ccall unsafe "&buffer_destroy"
  bufferDestroy__ptr :: FinalizerEnvPtr Api Buffer

foreign import ccall unsafe "buffer_to_host_buffer__get_dst_size"
  c__bufferToHostBuffer__getDstSize :: Ptr Api -> Ptr Buffer -> Ptr MemoryLayout -> IO CSize
foreign import ccall unsafe "buffer_to_host_buffer__event_await"
  c__bufferToHostBuffer__eventAwait :: Ptr Api -> Ptr Buffer -> Ptr MemoryLayout -> MutableByteArray# RealWorld -> CSize -> IO () 

bufferToHostBuffer :: Ptr Api -> Ptr Buffer -> IO ByteArray
bufferToHostBuffer api buffer = do 
  dstSize               <- c__bufferToHostBuffer__getDstSize api buffer nullPtr
  MutableByteArray dst# <- newPinnedByteArray $ fromIntegral dstSize
  c__bufferToHostBuffer__eventAwait api buffer nullPtr dst# dstSize 
  unsafeFreezeByteArray (MutableByteArray dst#)
