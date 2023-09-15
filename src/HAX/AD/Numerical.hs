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


neighborhood :: forall s t. (T s t, Num t) => t -> Tensor s t -> [Tensor s t]
neighborhood _delta (tensorToPrimArray -> tensor) = [unsafePerformIO $ do 
  buffer <- mallocArray nelem
  copyPrimArrayToPtr buffer tensor 0 nelem
  pokeElemOff buffer i (_delta + indexPrimArray tensor i)
  tensorFromHostBufferGC defaultDevice buffer | i <- [0..nelem - 1]]
  where nelem = fromInteger $ product $ shapeVal (Proxy :: Proxy s)

delta :: Fractional t => t
delta = 0.004918684734
class NumericalMethod f where
  type NGradFunc g f
  type NGradReif f
  ngrad' :: ([[Double]] -> g) -> [[f]] -> f -> NGradFunc g f
  nreif  :: Annotated [[Double]] f -> NGradReif f

instance (T s t, T '[] t', Fractional t, Real t') => NumericalMethod (Tensor s t -> Tensor '[] t') where
  type NGradFunc g (Tensor s t -> Tensor '[] t') = Tensor s t -> g
  type NGradReif (Tensor s t -> Tensor '[] t') = Tensor s t
  ngrad' reifier fs f n = reifier grad
    where ns   = neighborhood delta n
          fs'  = fmap (<*> [n]) fs ++ [fmap f ns]
          f'   = f n
          grad = fmap (fmap (\x -> realToFrac (getScalar (x - f')) / delta)) fs'
  nreif (Annotated [grad]) = unsafePerformIO $ tensorFromHostBufferGC defaultDevice =<< newArray (realToFrac <$> grad)
  nreif _                  = error "Not enough or too many results generated"

instance (T s t, Fractional t, NumericalMethod (a -> b)) => NumericalMethod (Tensor s t -> a -> b) where
  type NGradFunc g (Tensor s t -> a -> b) = Tensor s t -> NGradFunc g (a -> b)
  type NGradReif (Tensor s t -> a -> b) = Tensor s t <+> NGradReif (a -> b)
  ngrad' reifier fs f n = ngrad' reifier fs' f'
    where ns   = neighborhood delta n
          fs'  = fmap (<*> [n]) fs ++ [fmap f ns]
          f'   = f n
  nreif (Annotated (grad:gs)) = unsafePerformIO (tensorFromHostBufferGC defaultDevice =<< newArray (realToFrac <$> grad)) :+: nreif (Annotated gs :: Annotated [[Double]] (a -> b))
  nreif _                     = error "Not enough value"

ngrad :: forall f. NumericalMethod f => f -> NGradFunc (NGradReif f) f
ngrad = ngrad' reifier []
  where reifier grad = nreif (Annotated grad :: Annotated [[Double]] f)













-- This sample gradient in random directions and check 
-- the error between numerical approximation and 
-- other methods
-- class Fractional p => RandomGradientTest p where
--   reduceGradient :: p -> Rational
--   rangradtest' :: (Fractional p, Fractional t, Tensorial t) => p -> (p -> Tensor '[] t) -> p -> Tensor '[] t -> [p] -> Rational -> t
--   rangradtest' center function gradient forward deviations stepsize =
--     case deviations of
--       []   -> 0
--       a:as -> 
--         let derivedSlope   = fromRational $ reduceGradient $ gradient * a
--             empiricalSlope = getScalar (function (center + a * fromRational stepsize) - forward) / fromRational stepsize
--             difference     = derivedSlope - empiricalSlope
--         in  difference * difference + rangradtest' center function gradient forward as stepsize
--   rangradtest :: (Fractional p, Fractional t, Tensorial t) => p -> (p -> Tensor '[] t, p -> p) -> [p] -> Rational -> t 
--   rangradtest center (forward, gradient) = 
--     rangradtest' center forward (gradient center) (forward center)
-- 
-- instance (T s t, Real t, Fractional t) => RandomGradientTest (Tensor s t) where
--   reduceGradient = toRational . getScalar . sigma'
-- 
-- instance (RandomGradientTest a, RandomGradientTest b) => RandomGradientTest (a <+> b) where
--   reduceGradient (a :+: b) = reduceGradient a + reduceGradient b










