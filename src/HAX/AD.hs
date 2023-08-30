{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilyDependencies #-}
{-# LANGUAGE ViewPatterns #-}
module HAX.AD where

import HAX.AD.Gradient
import HAX.AD.Reverse

import HAX.HList
import HAX.Utils

import Data.Proxy
import Foreign.C



class ReverseMode f where
  type Rev g f
  type GradResult f
  rgrad' :: (Gradient -> g, CIntPtr) -> f -> Rev g f
  rgradReify :: Annotated CIntPtr f -> Gradient -> GradResult f

-- TODO: Figure out how to do this without Proxy t
--       Make this a function or use overlaping instance
instance Num t => ReverseMode (Reverse t) where
  type Rev g (Reverse t) = g
  type GradResult (Reverse t) = Proxy t
  rgrad' (fst -> g) = g . (`cotangent` 1)
  rgradReify _ _ = Proxy

instance (ReverseMode f, Cotangent t) => ReverseMode (Reverse t -> f) where
  type Rev g (Reverse t -> f) = t -> Rev g f
  type GradResult (Reverse t -> f) = t <+> GradResult f
  rgrad' (g, i) f t = rgrad' (g, i + 1) (f $ Reverse t (independent i))
  rgradReify (Annotated idx) (reifyGrad idx -> (g, g')) = g :+: rgradReify (Annotated (idx + 1) :: Annotated CIntPtr f) g'

type RGrad f = Rev (GradResult f) f
rgrad :: forall f. ReverseMode f => f -> RGrad f
rgrad = rgrad' (rgradReify (Annotated 0 :: Annotated CIntPtr f), 0)
