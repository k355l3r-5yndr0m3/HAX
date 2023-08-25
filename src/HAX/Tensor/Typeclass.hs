{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
module HAX.Tensor.Typeclass where
import HAX.PjRt.BufferType
import HAX.TList
import HAX.Tensor.Shape

import Data.Proxy
import Data.Kind as D

import Foreign
import Foreign.C

import MLIR as M
import MLIR.C.IR (Value)
import Data.IntMap.Strict (IntMap, empty)


class (Storable a) => Tensorial a where
  pjrtBufferType :: Proxy a -> BufferType
  shloTensorType :: Proxy a -> M.Type
  denseSplatAttr :: M.Type -> a -> Attribute

  unitElement :: a  
  nullElement :: a

class Traceable f where
  trace' :: CIntPtr -> f -> (BlockM (IntMap Value, [Value]), ([M.Type], [M.Type]))

instance Traceable (TList '[]) where 
  trace' _ (:@) = (return (empty, []), ([], []))

trace :: Traceable (a -> b) => (a -> b) -> (BlockM [Value], ([M.Type], [M.Type]))
trace = (\ (a, b) -> (snd <$> a, b)) . (trace' 0)

instance Tensorial Float where
  pjrtBufferType _ = f32
  shloTensorType _ = f32Type
  denseSplatAttr   = denseElementsAttrFloatSplat

  unitElement = 1
  nullElement = 0

class Trace (t :: Shape -> D.Type -> D.Type) where
--  placeholder :: (Typeable s, Typeable d) => CIntPtr -> (t s d, CIntPtr)

type T s t = (KnownShape s, Tensorial t)
