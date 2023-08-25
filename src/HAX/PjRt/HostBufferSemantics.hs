{-# LANGUAGE ForeignFunctionInterface, CApiFFI #-}
module HAX.PjRt.HostBufferSemantics (
  HostBufferSemantics (..)
, kImmutableOnlyDuringCall 
, kImmutableUntilTransferCompletes 
, kZeroCopy
) where
import Foreign

newtype HostBufferSemantics = HostBufferSemantics Int32


foreign import capi unsafe "pjrt_c_api.h value PJRT_HostBufferSemantics_kImmutableOnlyDuringCall"
  c__enum__PJRT_HostBufferSemantics_kImmutableOnlyDuringCall :: Int32
kImmutableOnlyDuringCall :: HostBufferSemantics
kImmutableOnlyDuringCall = HostBufferSemantics c__enum__PJRT_HostBufferSemantics_kImmutableOnlyDuringCall
foreign import capi unsafe "pjrt_c_api.h value PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes"
  c__enum__PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes :: Int32
kImmutableUntilTransferCompletes :: HostBufferSemantics
kImmutableUntilTransferCompletes = HostBufferSemantics c__enum__PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes
foreign import capi unsafe "pjrt_c_api.h value PJRT_HostBufferSemantics_kZeroCopy"
  c__enum__PJRT_HostBufferSemantics_kZeroCopy :: Int32
kZeroCopy :: HostBufferSemantics
kZeroCopy = HostBufferSemantics c__enum__PJRT_HostBufferSemantics_kZeroCopy

