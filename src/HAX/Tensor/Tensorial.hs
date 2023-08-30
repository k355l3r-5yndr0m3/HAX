{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE TypeFamilies #-}
module HAX.Tensor.Tensorial where
import HAX.PjRt.BufferType
import HAX.HList

import Data.Proxy
import GHC.TypeLits

import Foreign
import Foreign.C

import MLIR 
import Data.IntMap.Strict (IntMap, empty)
import Data.Primitive.ByteArray

-- Shape
type Shape = [Nat]
class KnownShape (s :: Shape) where
  shapeVal :: Proxy s -> [Integer]
  
instance KnownShape '[] where
  shapeVal _ = []

instance (KnownNat a, KnownShape as) => KnownShape (a ': as) where
  shapeVal _ = natVal (Proxy :: Proxy a) : shapeVal (Proxy :: Proxy as)

-- Tensorial
class (Storable a, DenseIntOrFPElementsAttr (DenseElemsAttr a), DenseIntOrFPElementsAttr (DenseSplatAttr a), TypeGet (SHLOType a) ) => Tensorial a where
  type SHLOType a
  type DenseSplatAttr a
  type DenseElemsAttr a

  pjrtBufferType  :: Proxy a -> BufferType
  shloTensorType  :: Proxy a -> SHLOType a
  
  shloTensorType' :: Proxy a -> AnyType
  shloTensorType' = toAnyType . shloTensorType

  elemByteSize   :: Proxy a -> Int

  denseSplatAttr :: [Int64] -> a -> DenseSplatAttr a
  -- TODO: Change ByteArray to something else that is parameterized by a 
  denseElemsAttr :: [Int64] -> ByteArray -> Proxy a -> DenseElemsAttr a

  unitElement :: a  
  nullElement :: a

instance Tensorial Float where
  type SHLOType Float = F32Type
  type DenseSplatAttr Float = DenseIntOrFPElements (RankedTensorType NullAttr F32Type) Float
  type DenseElemsAttr Float = DenseElementsRawBuffer (RankedTensorType NullAttr F32Type) 

  pjrtBufferType _ = f32
  shloTensorType _ = F32Type
  elemByteSize   _ = sizeOf (0 :: Float)
  
  denseSplatAttr shape = DenseIntOrFPElements (RankedTensorType shape F32Type NullAttr)
  denseElemsAttr shape tensorData _ = DenseElementsRawBuffer (RankedTensorType shape F32Type NullAttr) tensorData

  unitElement = 1
  nullElement = 0

-- Traceable
class Traceable f where
  trace' :: CIntPtr -> f -> (BlockM (IntMap Value, [Value]), ([AnyType], [AnyType]))

instance Traceable (HList '[]) where 
  trace' _ (:@) = (return (empty, []), ([], []))

trace :: Traceable (a -> b) => (a -> b) -> (BlockM [Value], ([AnyType], [AnyType]))
trace f = (fmap snd _fst, _snd)
  where (_fst, _snd) = trace' 0 f

type T s t = (KnownShape s, Tensorial t)
tensorType :: forall a s t. T s t => Proxy (a s t) -> RankedTensorType NullAttr (SHLOType t)
tensorType _ = RankedTensorType shape _type NullAttr
  where shape = fromInteger <$> shapeVal (Proxy :: Proxy s)
        _type = shloTensorType (Proxy :: Proxy t)

tensorType' :: T s t => Proxy (a s t) -> AnyType 
tensorType' = toAnyType . tensorType
