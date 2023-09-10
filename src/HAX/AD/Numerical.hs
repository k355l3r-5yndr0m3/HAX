{-# LANGUAGE MagicHash #-}
module HAX.AD.Numerical where -- This module is intended only for debugging of differentiating algorithms
import HAX.PjRt

import HAX.Tensor.Tensorial
import HAX.Tensor.Tensor

import Data.Proxy

import Foreign

import GHC.IO.Unsafe

class (Tensorial t, Num t) => NumericalMethod t where
  delta  :: t

  delhot :: T s t => [Tensor s t]
  delhot = valhot delta

  onehot :: T s t => [Tensor s t]
  valhot :: T s t => t -> [Tensor s t]

  valhot' :: forall s. T s t => (t, t) -> [Tensor s t]
  valhot' (zero, one) = [unsafePerformIO $ do 
    content <- mallocArray nelem 
    pokeArray content $ replicate nelem zero
    pokeElemOff content i one
    tensorFromHostBufferGC defaultDevice content | i <- [0..nelem - 1]]
    where nelem = fromInteger $ product $ shapeVal (Proxy :: Proxy s)

  

instance NumericalMethod Float where
  delta = 0.00491868473
  onehot = valhot' (0, 1)
  valhot t = valhot' (0, t)
  
