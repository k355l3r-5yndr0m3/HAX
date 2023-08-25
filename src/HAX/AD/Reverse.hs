{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
module HAX.AD.Reverse where

import HAX.Tensor
import HAX.TList

import Data.Dynamic 
import Data.Kind
import Data.Data
import Data.List (partition)

import Foreign.C.Types

-- TODO: Optimize
data Gradient = Gradient [(CIntPtr, Dynamic)]

(<+>) :: Gradient -> Gradient -> Gradient
(Gradient lhs) <+> (Gradient rhs) = Gradient (lhs ++ rhs)

class (Trace f, Typeable f) => PartialGradient f where
  partialGradient :: Typeable (f s t) => CIntPtr -> f s t -> Gradient
  gradientDecode :: (Typeable (f s t), Num (f s t)) => CIntPtr -> Gradient -> (f s t, Gradient)

instance PartialGradient Tracer where
  partialGradient i t = Gradient [(i, toDyn t)]
  gradientDecode i (Gradient g) = (sum (fromDyn <$> a <*> [error "Wrong type"]), Gradient b)
    where (fmap snd -> a, b) = partition ((i ==) . fst) g

zero :: Gradient
zero = Gradient []

data RDual f s t = RDual { primal :: f s t, partial :: f s t -> Gradient }

instance (Trace t) => Trace (RDual t) where

instance (Num (f s t)) => Num (RDual f s t) where
  (RDual f f') + (RDual g g') = 
    RDual (f + g) (\ i -> f' i <+> g' i)

  (RDual f f') - (RDual g g') = 
    RDual (f - g) (\ i -> f' i <+> g' (negate i))

  (RDual f f') * (RDual g g') = 
    RDual (f * g) (\ i -> f' (i * g) <+> g' (i * f))

  negate (RDual f f') = 
    RDual (negate f) (\ i -> f' (negate i))

  abs    (RDual f f') = 
    RDual (abs f) (\ i -> f' (i * signum f))

  signum (RDual f f') = 
    RDual (signum f) (\ _ -> f' 0 )
  
  fromInteger a = 
    RDual (fromInteger a) (const zero)

instance (Fractional (f s t)) => Fractional (RDual f s t) where
  recip (RDual f f') = 
    RDual (recip f) (\ i -> f' ((negate i) / f / f))

  (RDual f f') / (RDual g g') = 
    RDual (f / g) (\ i -> f' (i / g) <+> g' (negate (f / (g * g))))

  fromRational r = 
    RDual (fromRational r) (const zero)

class ReverseDiff f where
  type F f
  type K f :: [Type]
  grad' :: (Gradient -> TList g) -> CIntPtr -> f -> GradImpl (F f) (TList g)
  takeg :: CIntPtr -> Proxy f -> Gradient -> TList (K f)


instance (Trace f, Num (f s t)) => ReverseDiff (RDual f s t) where
  type F (RDual f s t) = Gradient
  type K (RDual f s t) = '[]
  grad' h _ f = h $ partial f 1
  takeg _ _ _ = (:@)
  

instance (ReverseDiff h, PartialGradient f, Typeable (f s t), Num (f s t)) => ReverseDiff (RDual f s t -> h) where
  type F (RDual f s t -> h) = f s t -> F h
  type K (RDual f s t -> h) = f s t ': K h
  grad' h i f a = grad' h (i + 1) (f a')
    where a' = RDual {
                 primal = a,
                 partial = partialGradient i
               }
  takeg i _ g = a :+ takeg (i + 1) p g'
    where (a, g') = gradientDecode i g
          p :: Proxy h = Proxy

type family GradImpl f g where
  GradImpl (a -> b) g = a -> GradImpl b g
  GradImpl _ g = g
type Grad f = GradImpl (F f) (TList (K f))

grad :: forall f. ReverseDiff f => f -> Grad f
grad f = grad' (takeg 0 (Proxy :: Proxy f)) 0 f
