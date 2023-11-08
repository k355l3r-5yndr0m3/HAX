{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilyDependencies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE ViewPatterns #-}
module HAX.AD.Numerical where -- This module is intended only for debugging of differentiating algorithms

import HAX.Tensor.Tensorial
import HAX.Tensor.Tensor
import HAX.PjRt
import HAX.Utils

import Data.Proxy
import Data.Primitive hiding (newArray)
import Foreign

import GHC.IO.Unsafe

delta :: Fractional t => t
delta = 0.004918684734

class Tensorial t => Neighborhood t where
  neighborhood' :: KnownShape s => StorageType t -> Tensor s t -> [Tensor s t]
  neighborhood  :: KnownShape s => Tensor s t -> [Tensor s t]
  
  realGradToTensor :: (Real t', KnownShape s) => [t'] -> Tensor s t

instance {-# OVERLAPPABLE #-} (Tensorial t, Fractional (StorageType t)) => Neighborhood t where
  neighborhood' :: forall s. KnownShape s => StorageType t -> Tensor s t -> [Tensor s t]
  neighborhood' _delta (tensorToPrimArray -> tensor) = [unsafePerformIO $ do 
    buffer <- mallocArray nelem
    copyPrimArrayToPtr buffer tensor 0 nelem
    pokeElemOff buffer i (_delta + indexPrimArray tensor i)
    tensorFromHostBufferGC defaultDevice buffer | i <- [0..nelem - 1]]
    where nelem = fromInteger $ product $ shapeVal (Proxy :: Proxy s)
  neighborhood = neighborhood' delta

  realGradToTensor g = unsafePerformIO $ tensorFromHostBufferGC defaultDevice =<< newArray (realToFrac <$> g)
instance Neighborhood Word8 where
  neighborhood' _ _ = []
  neighborhood  _   = []

  realGradToTensor _ = 0
instance Neighborhood Bool where
  neighborhood' _ _ = []
  neighborhood  _   = []

  realGradToTensor _ = splat False

class NumericalMethod f where
  type NGradFunc g f
  type NGradReif f
  ngrad' :: ([[Double]] -> g) -> [[f]] -> f -> NGradFunc g f
  nreif  :: Annotated [[Double]] f -> NGradReif f

instance (T s t, T '[] t', Neighborhood t, Real t') => NumericalMethod (Tensor s t -> Tensor '[] t') where
  type NGradFunc g (Tensor s t -> Tensor '[] t') = Tensor s t -> g
  type NGradReif (Tensor s t -> Tensor '[] t') = Tensor s t
  ngrad' reifier fs f n = reifier grad
    where ns   = neighborhood n
          fs'  = fmap (<*> [n]) fs ++ [fmap f ns]
          f'   = f n
          grad = fmap (fmap (\x -> realToFrac (getScalar (x - f')) / delta)) fs'
  nreif (Annotated [grad]) = realGradToTensor grad
  nreif _                  = error "Not enough or too many results generated"


instance (T s t, Neighborhood t, NumericalMethod (a -> b)) => NumericalMethod (Tensor s t -> a -> b) where
  type NGradFunc g (Tensor s t -> a -> b) = Tensor s t -> NGradFunc g (a -> b)
  type NGradReif (Tensor s t -> a -> b) = Tensor s t <&> NGradReif (a -> b)
  ngrad' reifier fs f n = ngrad' reifier fs' f'
    where ns   = neighborhood n
          fs'  = fmap (<*> [n]) fs ++ [fmap f ns]
          f'   = f n
  nreif (Annotated (grad:gs)) = realGradToTensor grad :&: nreif (Annotated gs :: Annotated [[Double]] (a -> b))
  nreif _                     = error "Not enough value"

ngrad :: forall f. NumericalMethod f => f -> NGradFunc (NGradReif f) f
ngrad = ngrad' reifier []
  where reifier grad = nreif (Annotated grad :: Annotated [[Double]] f)
