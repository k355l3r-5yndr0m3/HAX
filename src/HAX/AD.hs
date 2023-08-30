{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilyDependencies #-}
{-# LANGUAGE ViewPatterns #-}
module HAX.AD where

import HAX.AD.Gradient
import HAX.AD.Reverse

import HAX.HList
import HAX.Utils

import Foreign.C
import Control.Exception (assert)



class ReverseMode f where
  type Rev g f
  type GradResult f
  rgrad' :: (Gradient -> g, CIntPtr) -> f -> Rev g f
  rgradReify :: Annotated CIntPtr f -> Gradient -> GradResult f

-- NOTE: This is somewhat undesireable since some part of the code overlap eachother
--       which could be solved by add another class for types that can be arguments to 
--       differentiable functions
instance (Cotangent t0, Num t) => ReverseMode (Reverse t0 -> Reverse t) where
  type Rev g (Reverse t0 -> Reverse t) = t0 -> g
  type GradResult (Reverse t0 -> Reverse t) = t0
  rgrad' (g, i) f t = g (cotangent (f $ Reverse t (independent i)) 1)
  rgradReify (Annotated idx) (reifyGrad idx -> (g, Gradient g')) = assert (null g') g

instance (ReverseMode (a -> f), Cotangent t) => ReverseMode (Reverse t -> a -> f) where
  type Rev g (Reverse t -> a -> f) = t -> Rev g (a -> f)
  type GradResult (Reverse t -> a -> f) = t <+> GradResult (a -> f)
  rgrad' (g, i) f t = rgrad' (g, i + 1) (f $ Reverse t (independent i))
  rgradReify (Annotated idx) (reifyGrad idx -> (g, g')) = g :+: rgradReify (Annotated (idx + 1) :: Annotated CIntPtr (a -> f)) g'

type RGrad f = Rev (GradResult f) f
rgrad :: forall f. ReverseMode f => f -> RGrad f
rgrad = rgrad' (rgradReify (Annotated 0 :: Annotated CIntPtr f), 0)
