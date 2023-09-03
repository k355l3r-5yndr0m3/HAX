module HAX.Transform where
import HAX.Tensor.Tensorial

import Data.Int
import Data.Word
import Data.Proxy

import MLIR
import Stablehlo.Dialect.Stablehlo.Attributes

data Transform = Id | V Int64 Transform deriving Eq


transformHeight :: Transform -> Word
transformHeight t = 
  case t of 
    Id     -> 0
    V _ t' -> 1 + transformHeight t'

transformTruncate :: Word -> Transform -> Transform
transformTruncate label tf = truncation tf (height - label)
  where height = transformHeight tf
        truncation :: Transform -> Word -> Transform
        truncation t 0 = t
        truncation t i = truncation (
          case t of 
            Id    -> error "Unexpectedly short transformation stack"
            V _ o -> o) (i - 1)


class ApplyTransform a where
  applyTransform :: Transform -> a -> a
instance ApplyTransform (RankedTensorType t e) where
  applyTransform tf r@(RankedTensorType shape t e) = 
    case tf of 
      Id          -> r
      V dim other -> applyTransform other (RankedTensorType (dim:shape) t e)


newtype BroadcastMap = BroadcastMap [Word64]
instance AttrGet BroadcastMap where 
  attrGet (BroadcastMap mapping) = 
    attrGet (DenseIntOrFPElements (VectorType [fromIntegral $ length mapping] I64) mapping)
instance DenseIntOrFPElementsAttr BroadcastMap
instance DenseIntElementsAttr BroadcastMap

getBroadcastMap :: KnownShape s => Proxy s -> BroadcastMap
getBroadcastMap = BroadcastMap . fmap fromInteger . shapeVal

instance ApplyTransform BroadcastMap where
  applyTransform tf b@(BroadcastMap idxmap) = 
    case tf of 
      Id          -> b
      V _ other   -> applyTransform other (BroadcastMap (0:fmap (+1) idxmap))

instance ApplyTransform DotDimensionNumbersAttr where
  applyTransform tf d = 
    case tf of 
      Id          -> d
      V _ other   -> 
        let incr (x, y) = (x + 1, y + 1)
            d' = d { getBatchingDims = (0,0):fmap incr (getBatchingDims d),
                     getContractingDims = fmap incr (getContractingDims d)}
        in  applyTransform other d'
