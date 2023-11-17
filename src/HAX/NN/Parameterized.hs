{-# LANGUAGE TypeFamilyDependencies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE DataKinds #-}
module HAX.NN.Parameterized where

import HAX.Tensor
import HAX.Target 
import HAX.AD.Reverse

import Data.Bifunctor (Bifunctor(first))

import GHC.TypeLits
import GHC.Generics

class Jitter p => Parameterized p where
  type Input  p 
  type Output p
  feed :: p -> Input p -> Output p

newtype Dense (r :: Z) (i :: Nat) (o :: Nat) t = Dense (r [i, o] t, r '[o] t) deriving Generic
instance (KnownNat i, KnownNat o, Tensorial t) => JitIn (Dense Tracer i o t) where
  type JitI (Dense Tracer i o t) = Dense Tensor i o t
instance JitIn (Dense r i o t) => JitIn (Dense (Reverse r) i o t) where
  type JitI (Dense (Reverse r) i o t) = JitI (Dense r i o t)
  jitIn i t = (i', Dense (R a undefined, R b undefined), t'')
    where (i', Dense (a, b), t'') = jitIn i t
instance JitIn (Dense r i o t) => JitIn (Dense (Target r) i o t) where
  type JitI (Dense (Target r) i o t) = JitI (Dense r i o t)
  jitIn i t = (i', Dense (Target [] a, Target [] b), t'')
    where (i', Dense (a, b), t'') = jitIn i t

instance (KnownNat i, KnownNat o, Tensorial t) => JitOut (Dense Tracer i o t) where
  type JitO (Dense Tracer i o t) = Dense Tensor i o t
  jitOut (Dense t) = (r, v, first Dense . g)
    where (r, v, g) = jitOut t
instance JitOut (Dense r i o t) => JitOut (Dense (Reverse r) i o t) where
  type JitO (Dense (Reverse r) i o t) = JitO (Dense r i o t)
  jitOut = jitOut
instance JitOut (Dense r i o t) => JitOut (Dense (Target r) i o t) where
  type JitO (Dense (Target r) i o t) = JitO (Dense r i o t)
  jitOut = jitOut
instance (TensorOp r, KnownNat i, KnownNat o, Tensorial t, Num t, Jitter (Dense r i o t)) => Parameterized (Dense r i o t) where
  type Input  (Dense r i o t) = r '[i] t
  type Output (Dense r i o t) = r '[o] t
  feed (Dense (weights, biases)) feats = biases `unsafePairwiseAdd` linearMap weights feats

data Sigmoid (r :: Z) (s :: Shape) t = Sigmoid deriving Generic
instance JitIn (Sigmoid Tracer s t) where
  type JitI (Sigmoid Tracer s t) = Sigmoid Tensor s t
instance JitIn (Sigmoid r s t) => JitIn (Sigmoid (Reverse r) s t) where
  type JitI (Sigmoid (Reverse r) s t) = JitI (Sigmoid r s t)
  jitIn i _ = (i, Sigmoid, []) 
instance JitIn (Sigmoid r s t) => JitIn (Sigmoid (Target r) s t) where
  type JitI (Sigmoid (Target r) s t) = JitI (Sigmoid r s t)
  jitIn i _ = (i, Sigmoid, []) 
instance JitOut (Sigmoid Tracer s t) where
  type JitO (Sigmoid Tracer s t) = Sigmoid Tensor s t
  jitOut _ = (pure . (, []), [] , (Sigmoid,))
instance JitOut (Sigmoid r s t) => JitOut (Sigmoid (Reverse r) s t) where
  type JitO (Sigmoid (Reverse r) s t) = JitO (Sigmoid r s t)
  jitOut = jitOut
instance JitOut (Sigmoid r s t) => JitOut (Sigmoid (Target r) s t) where
  type JitO (Sigmoid (Target r) s t) = JitO (Sigmoid r s t)
  jitOut = jitOut
instance (Floating (r s t), Jitter (Sigmoid r s t)) => Parameterized (Sigmoid r s t) where
  type Input  (Sigmoid r s t) = r s t
  type Output (Sigmoid r s t) = r s t
  feed _ = sigmoid

data ReLU (r :: Z) (s :: Shape) t = ReLU
instance (TensorOp r, T s t, Num (r s t), Jitter (ReLU r s t)) => Parameterized (ReLU r s t) where
  type Input  (ReLU r s t) = r s t
  type Output (ReLU r s t) = r s t
  feed _ = relu

newtype LeakyReLU (r :: Z) (s :: Shape) t = LeakyReLU Rational deriving Generic
instance JitIn (LeakyReLU Tracer s t) where
  type JitI (LeakyReLU Tracer s t) = LeakyReLU Tensor s t
instance JitIn (LeakyReLU r s t) => JitIn (LeakyReLU (Reverse r) s t) where
  type JitI (LeakyReLU (Reverse r) s t) = JitI (LeakyReLU r s t)
  jitIn i _ = (i, LeakyReLU 0, []) 
instance JitIn (LeakyReLU r s t) => JitIn (LeakyReLU (Target r) s t) where
  type JitI (LeakyReLU (Target r) s t) = JitI (LeakyReLU r s t)
  jitIn i _ = (i, LeakyReLU 0, []) 
instance JitOut (LeakyReLU Tracer s t) where
  type JitO (LeakyReLU Tracer s t) = LeakyReLU Tensor s t
  jitOut (LeakyReLU i) = (pure . (, []), [] , (LeakyReLU i,))
instance JitOut (LeakyReLU r s t) => JitOut (LeakyReLU (Reverse r) s t) where
  type JitO (LeakyReLU (Reverse r) s t) = JitO (LeakyReLU r s t)
  jitOut = jitOut
instance JitOut (LeakyReLU r s t) => JitOut (LeakyReLU (Target r) s t) where
  type JitO (LeakyReLU (Target r) s t) = JitO (LeakyReLU r s t)
  jitOut = jitOut
instance (T s t, TensorOp r, Fractional t, Num (r s t), Jitter (LeakyReLU r s t)) => Parameterized (LeakyReLU r s t) where
  type Input  (LeakyReLU r s t) = r s t
  type Output (LeakyReLU r s t) = r s t
  feed (LeakyReLU alpha) = leakyrelu $ splat $ fromRational alpha
