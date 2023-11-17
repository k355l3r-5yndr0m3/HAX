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
  type JitT (Dense r t i o) = Dense Tensor t i o
instance (TensorOp r, KnownNat i, KnownNat o, Tensorial t, Num t, JNT r) => Parameterized (Dense r t i o) where
  type Input  (Dense r t i o) = r '[i] t
  type Output (Dense r t i o) = r '[o] t
  feed (Dense (weights, biases)) feats = biases `unsafePairwiseAdd` linearMap weights feats

data Sigmoid (r :: Z) t (s :: Shape) = Sigmoid deriving Generic
instance (JNT r, T s t) => Jitter (Sigmoid r t s) where
  type JitT (Sigmoid r t s) = Sigmoid Tensor t s
instance (T s t, Floating (r s t), JNT r) => Parameterized (Sigmoid r t s) where
  type Input  (Sigmoid r t s) = r s t
  type Output (Sigmoid r t s) = r s t
  feed _ = sigmoid

data ReLU (r :: Z) t (s :: Shape) = ReLU deriving Generic
instance (JNT r, T s t) => Jitter (ReLU r t s) where
  type JitT (ReLU r t s) = ReLU Tensor t s
instance (TensorOp r, T s t, Num (r s t), JNT r) => Parameterized (ReLU r t s) where
  type Input  (ReLU r t s) = r s t
  type Output (ReLU r t s) = r s t
  feed _ = relu

newtype LeakyReLU (r :: Z) t (s :: Shape) = LeakyReLU Rational deriving Generic
instance (JNT r, T s t) => Jitter (LeakyReLU r t s) where
  type JitT (LeakyReLU r t s) = LeakyReLU Tensor t s

instance (T s t, TensorOp r, Fractional t, Num (r s t), JNT r) => Parameterized (LeakyReLU r t s) where
  type Input  (LeakyReLU r t s) = r s t
  type Output (LeakyReLU r t s) = r s t
  feed (LeakyReLU alpha) = leakyrelu $ splat $ fromRational alpha

instance (Jitter a, Jitter b) => Jitter (a >> b) where
  type JitT (a >> b) = JitT a >> JitT b
instance (Parameterized a, Parameterized b, Output a ~ Input b) => Parameterized (a >> b) where
  type Input  (a >> b) = Input a
  type Output (a >> b) = Output b

data a >> b = a :>: b deriving Generic
data a !! b = a :!: b deriving Generic

