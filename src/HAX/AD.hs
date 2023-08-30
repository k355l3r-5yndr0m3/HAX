{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilyDependencies #-}
{-# LANGUAGE ViewPatterns #-}
module HAX.AD where

import HAX.AD.Gradient
import HAX.AD.Reverse

-- import HAX.HList

import Data.Kind
import Data.Proxy
import Foreign.C


-- class ReverseDiff f where
--   type F f
--   type K f :: [Type]
--   grad' :: (Gradient -> HList g) -> CIntPtr -> f -> GradImpl (F f) (HList g)
--   takeg :: CIntPtr -> Proxy f -> Gradient -> HList (K f)
-- 
-- 
-- instance (Trace f, Num (f s t)) => ReverseDiff (Reverse f s t) where
--   type F (Reverse f s t) = Gradient
--   type K (Reverse f s t) = '[]
--   grad' h _ f = h $ partial f 1
--   takeg _ _ _ = (:@)
--   
-- 
-- instance (ReverseDiff h, PartialGradient f, Typeable (f s t), Num (f s t)) => ReverseDiff (Reverse f s t -> h) where
--   type F (Reverse f s t -> h) = f s t -> F h
--   type K (Reverse f s t -> h) = f s t ': K h
--   grad' h i f a = grad' h (i + 1) (f a')
--     where a' = Reverse {
--                  primal = a,
--                  partial = partialGradient i
--                }
--   takeg i _ g = a :+ takeg (i + 1) p g'
--     where (a, g') = gradientDecode i g
--           p :: Proxy h = Proxy
-- 
-- type family GradImpl f g where
--   GradImpl (a -> b) g = a -> GradImpl b g
--   GradImpl _ g = g
-- type Grad f = GradImpl (F f) (HList (K f))
-- 
-- grad :: forall f. ReverseDiff f => f -> Grad f
-- grad = grad' (takeg 0 (Proxy :: Proxy f)) 0

-- newtype Annotated a b = Annotated a
-- 
-- class ReverseMode f where
--   type Rev g f
--   type GradResult f :: [Type]
--   rgrad' :: (Gradient -> g, CIntPtr) -> f -> Rev g f
--   rgradReify :: Annotated CIntPtr f -> Gradient -> HList (GradResult f)
-- 
-- instance Num t => ReverseMode (Reverse t) where
--   type Rev g (Reverse t) = g
--   type GradResult _    = '[]
--   rgrad' (fst -> g) = g . (`cotangent` 1)
--   rgradReify _ _ = (:@)
-- 
-- instance (ReverseMode f, Cotangent t) => ReverseMode (Reverse t -> f) where
--   type Rev g (Reverse t -> f) = t -> Rev g f
--   type GradResult (Reverse t -> f) = t ': GradResult f
--   rgrad' (g, i) f t = rgrad' (g, i + 1) (f $ Reverse t (independent i))
--   rgradReify (Annotated idx) (reifyGrad idx -> (g, g')) = g :+ rgradReify (Annotated (idx + 1) :: Annotated CIntPtr f) g'
-- 
-- type RGrad f = Rev (HList (GradResult f)) f
-- rgrad :: forall f. ReverseMode f => f -> RGrad f
-- rgrad = rgrad' (rgradReify (Annotated 0 :: Annotated CIntPtr f), 0)


