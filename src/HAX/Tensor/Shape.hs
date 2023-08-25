{-# LANGUAGE DataKinds, TypeFamilies #-}
module HAX.Tensor.Shape where
import Data.Proxy
import GHC.TypeLits

type Shape = [Nat]

class KnownShape (s :: Shape) where
  shapeVal :: Proxy s -> [Integer]
  
instance KnownShape '[] where
  shapeVal _ = []

instance (KnownNat a, KnownShape as) => KnownShape (a ': as) where
  shapeVal _ = natVal (Proxy :: Proxy a) : shapeVal (Proxy :: Proxy as)
