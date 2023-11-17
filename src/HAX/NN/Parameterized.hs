{-# LANGUAGE TypeFamilyDependencies #-}
{-# LANGUAGE DataKinds #-}
module HAX.NN.Parameterized where

import HAX.Tensor

import GHC.TypeLits
import GHC.Generics

class Jitter p => Parameterized p where
  type Input  p 
  type Output p
  feed :: p -> Input p -> Output p

newtype Dense (r :: Z) t (i :: Nat) (o :: Nat) = Dense (r [i, o] t, r '[o] t) deriving Generic
instance (JNT r, KnownNat i, KnownNat o, Tensorial t) => Jitter (Dense r t i o) where
  type JitI (Dense r t i o) = Dense Tensor t i o
  type JitO (Dense r t i o) = Dense Tensor t i o
instance (TensorOp r, KnownNat i, KnownNat o, Tensorial t, Num t, JNT r) => Parameterized (Dense r t i o) where
  type Input  (Dense r t i o) = r '[i] t
  type Output (Dense r t i o) = r '[o] t
  feed (Dense (weights, biases)) feats = biases `unsafePairwiseAdd` linearMap weights feats

data Sigmoid (r :: Z) t (s :: Shape) = Sigmoid deriving Generic
instance (JNT r, T s t) => Jitter (Sigmoid r t s) where
  type JitI (Sigmoid r t s) = Sigmoid Tensor t s
  type JitO (Sigmoid r t s) = Sigmoid Tensor t s
instance (T s t, Floating (r s t), JNT r) => Parameterized (Sigmoid r t s) where
  type Input  (Sigmoid r t s) = r s t
  type Output (Sigmoid r t s) = r s t
  feed _ = sigmoid

data ReLU (r :: Z) t (s :: Shape) = ReLU deriving Generic
instance (JNT r, T s t) => Jitter (ReLU r t s) where
  type JitI (ReLU r t s) = ReLU Tensor t s
  type JitO (ReLU r t s) = ReLU Tensor t s
instance (TensorOp r, T s t, Num (r s t), JNT r) => Parameterized (ReLU r t s) where
  type Input  (ReLU r t s) = r s t
  type Output (ReLU r t s) = r s t
  feed _ = relu

newtype LeakyReLU (r :: Z) t (s :: Shape) = LeakyReLU Rational deriving Generic
instance (JNT r, T s t) => Jitter (LeakyReLU r t s) where
  type JitI (LeakyReLU r t s) = LeakyReLU Tensor t s
  type JitO (LeakyReLU r t s) = LeakyReLU Tensor t s

instance (T s t, TensorOp r, Fractional t, Num (r s t), JNT r) => Parameterized (LeakyReLU r t s) where
  type Input  (LeakyReLU r t s) = r s t
  type Output (LeakyReLU r t s) = r s t
  feed (LeakyReLU alpha) = leakyrelu $ splat $ fromRational alpha


