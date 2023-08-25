{-# LANGUAGE ForeignFunctionInterface, CApiFFI #-}
module HAX.PjRt.BufferType (
  BufferType (..) 
, invalid
, pred
, s8
, s16
, s32
, s64
, u8
, u16
, u32
, u64
, f16
, f32
, f64
, bf16
, c64
, c128
, f8e5m2
, f8e4m3fn
, enum
, f8e5m2fnuz
, f8e4m3fnuz
, s4
, u4
) where
import Prelude hiding (pred)

import Foreign

newtype BufferType = BufferType Int32

foreign import capi unsafe "pjrt_c_api.h value PJRT_Buffer_Type_INVALID"
  c__enum__PJRT_Buffer_Type_INVALID :: Int32
invalid :: BufferType 
invalid = BufferType c__enum__PJRT_Buffer_Type_INVALID

foreign import capi unsafe "pjrt_c_api.h value PJRT_Buffer_Type_PRED"
  c__enum__PJRT_Buffer_Type_PRED :: Int32
pred :: BufferType 
pred = BufferType c__enum__PJRT_Buffer_Type_PRED

foreign import capi unsafe "pjrt_c_api.h value PJRT_Buffer_Type_S8"
  c__enum__PJRT_Buffer_Type_S8 :: Int32
s8 :: BufferType 
s8 = BufferType c__enum__PJRT_Buffer_Type_S8

foreign import capi unsafe "pjrt_c_api.h value PJRT_Buffer_Type_S16"
  c__enum__PJRT_Buffer_Type_S16 :: Int32
s16 :: BufferType 
s16 = BufferType c__enum__PJRT_Buffer_Type_S16

foreign import capi unsafe "pjrt_c_api.h value PJRT_Buffer_Type_S32"
  c__enum__PJRT_Buffer_Type_S32 :: Int32
s32 :: BufferType 
s32 = BufferType c__enum__PJRT_Buffer_Type_S32

foreign import capi unsafe "pjrt_c_api.h value PJRT_Buffer_Type_S64"
  c__enum__PJRT_Buffer_Type_S64 :: Int32
s64 :: BufferType 
s64 = BufferType c__enum__PJRT_Buffer_Type_S64

foreign import capi unsafe "pjrt_c_api.h value PJRT_Buffer_Type_U8"
  c__enum__PJRT_Buffer_Type_U8 :: Int32
u8 :: BufferType 
u8 = BufferType c__enum__PJRT_Buffer_Type_U8

foreign import capi unsafe "pjrt_c_api.h value PJRT_Buffer_Type_U16"
  c__enum__PJRT_Buffer_Type_U16 :: Int32
u16 :: BufferType 
u16 = BufferType c__enum__PJRT_Buffer_Type_U16

foreign import capi unsafe "pjrt_c_api.h value PJRT_Buffer_Type_U32"
  c__enum__PJRT_Buffer_Type_U32 :: Int32
u32 :: BufferType 
u32 = BufferType c__enum__PJRT_Buffer_Type_U32

foreign import capi unsafe "pjrt_c_api.h value PJRT_Buffer_Type_U64"
  c__enum__PJRT_Buffer_Type_U64 :: Int32
u64 :: BufferType 
u64 = BufferType c__enum__PJRT_Buffer_Type_U64

foreign import capi unsafe "pjrt_c_api.h value PJRT_Buffer_Type_F16"
  c__enum__PJRT_Buffer_Type_F16 :: Int32
f16 :: BufferType 
f16 = BufferType c__enum__PJRT_Buffer_Type_F16

foreign import capi unsafe "pjrt_c_api.h value PJRT_Buffer_Type_F32"
  c__enum__PJRT_Buffer_Type_F32 :: Int32
f32 :: BufferType 
f32 = BufferType c__enum__PJRT_Buffer_Type_F32

foreign import capi unsafe "pjrt_c_api.h value PJRT_Buffer_Type_F64"
  c__enum__PJRT_Buffer_Type_F64 :: Int32
f64 :: BufferType 
f64 = BufferType c__enum__PJRT_Buffer_Type_F64

foreign import capi unsafe "pjrt_c_api.h value PJRT_Buffer_Type_BF16"
  c__enum__PJRT_Buffer_Type_BF16 :: Int32
bf16 :: BufferType 
bf16 = BufferType c__enum__PJRT_Buffer_Type_BF16

foreign import capi unsafe "pjrt_c_api.h value PJRT_Buffer_Type_C64"
  c__enum__PJRT_Buffer_Type_C64 :: Int32
c64 :: BufferType 
c64 = BufferType c__enum__PJRT_Buffer_Type_C64

foreign import capi unsafe "pjrt_c_api.h value PJRT_Buffer_Type_C128"
  c__enum__PJRT_Buffer_Type_C128 :: Int32
c128 :: BufferType 
c128 = BufferType c__enum__PJRT_Buffer_Type_C128

foreign import capi unsafe "pjrt_c_api.h value PJRT_Buffer_Type_F8E5M2"
  c__enum__PJRT_Buffer_Type_F8E5M2 :: Int32
f8e5m2 :: BufferType 
f8e5m2 = BufferType c__enum__PJRT_Buffer_Type_F8E5M2

foreign import capi unsafe "pjrt_c_api.h value PJRT_Buffer_Type_F8E4M3FN"
  c__enum__PJRT_Buffer_Type_F8E4M3FN :: Int32
f8e4m3fn :: BufferType 
f8e4m3fn = BufferType c__enum__PJRT_Buffer_Type_F8E4M3FN

foreign import capi unsafe "pjrt_c_api.h value PJRT_Buffer_Type_F8E4M3B11FNUZ"
  c__enum__PJRT_Buffer_Type_F8E4M3B11FNUZ :: Int32
enum :: BufferType 
enum = BufferType c__enum__PJRT_Buffer_Type_F8E4M3B11FNUZ

foreign import capi unsafe "pjrt_c_api.h value PJRT_Buffer_Type_F8E5M2FNUZ"
  c__enum__PJRT_Buffer_Type_F8E5M2FNUZ :: Int32
f8e5m2fnuz :: BufferType 
f8e5m2fnuz = BufferType c__enum__PJRT_Buffer_Type_F8E5M2FNUZ

foreign import capi unsafe "pjrt_c_api.h value PJRT_Buffer_Type_F8E4M3FNUZ"
  c__enum__PJRT_Buffer_Type_F8E4M3FNUZ :: Int32
f8e4m3fnuz :: BufferType 
f8e4m3fnuz = BufferType c__enum__PJRT_Buffer_Type_F8E4M3FNUZ

foreign import capi unsafe "pjrt_c_api.h value PJRT_Buffer_Type_S4"
  c__enum__PJRT_Buffer_Type_S4 :: Int32
s4 :: BufferType 
s4 = BufferType c__enum__PJRT_Buffer_Type_S4

foreign import capi unsafe "pjrt_c_api.h value PJRT_Buffer_Type_U4"
  c__enum__PJRT_Buffer_Type_U4 :: Int32
u4 :: BufferType 
u4 = BufferType c__enum__PJRT_Buffer_Type_U4
