{-# LANGUAGE TypeFamilyDependencies #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE UndecidableInstances #-}
module HAX.NN.Parameterized where

import HAX.Tensor
import HAX.AD

import Data.Kind

import GHC.TypeLits
import GHC.Generics

class (GradIn p, Jitter p) => Parameterized p where
  type Input  p b
  type Output p a
  type Constr p a b :: Constraint
  feed :: (Constr p a b, a ~ Input p b, b ~ Output p a) => p -> a -> b

newtype Dense (r :: Z) t (i :: Nat) (o :: Nat) = Dense (r [i, o] t, r '[o] t) deriving Generic
instance (GNT r, KnownNat i, KnownNat o, Tensorial t) => GradIn (Dense r t i o) where
  type GradI (Dense r t i o) = Dense (Ins r) t i o
instance (JNT r, KnownNat i, KnownNat o, Tensorial t) => Jitter (Dense r t i o) where
  type JitT (Dense r t i o) = Dense Tensor t i o
instance (TensorOp r, KnownNat i, KnownNat o, Tensorial t, Num t, JNT r, GNT r) => Parameterized (Dense r t i o) where
  type Input  (Dense r t i o) _ = r '[i] t
  type Output (Dense r t i o) _ = r '[o] t
  type Constr (Dense r t i o) _ _ = ()
  feed (Dense (weights, biases)) feats = biases `unsafePairwiseAdd` linearMap weights feats

data Reshape (r :: Z) t (i :: Shape) (o :: Shape) = Reshape deriving Generic
instance (GNT r, Tensorial t, KnownShape i, KnownShape o) => GradIn (Reshape r t i o) where
  type GradI (Reshape r t i o) = Reshape (Ins r) t i o
instance (JNT r, Tensorial t, KnownShape i, KnownShape o) => Jitter (Reshape r t i o) where
  type JitT (Reshape r t i o) = Reshape Tensor t i o
instance (TensorOp r, Tensorial t, KnownShape i, KnownShape o, JNT r, GNT r, Reshapable i o) => Parameterized (Reshape r t i o) where 
  type Input  (Reshape r t i o) _ = r i t
  type Output (Reshape r t i o) _ = r o t
  type Constr (Reshape r t i o) _ _ = ()
  feed _ = reshape

data Sigmoid (r :: Z) t (s :: Shape) = Sigmoid deriving Generic
instance (JNT r, T s t) => Jitter (Sigmoid r t s) where
  type JitT (Sigmoid r t s) = Sigmoid Tensor t s
instance (GNT r, T s t) => GradIn (Sigmoid r t s) where
  type GradI (Sigmoid r t s) = Sigmoid (Ins r) t s 
instance (T s t, Floating (r s t), JNT r, GNT r) => Parameterized (Sigmoid r t s) where
  type Input  (Sigmoid r t s) _ = r s t
  type Output (Sigmoid r t s) _ = r s t
  type Constr (Sigmoid r t s) _ _ = ()
  feed _ = sigmoid

data ReLU (r :: Z) t (s :: Shape) = ReLU deriving Generic
instance (JNT r, T s t) => Jitter (ReLU r t s) where
  type JitT (ReLU r t s) = ReLU Tensor t s
instance (GNT r, T s t) => GradIn (ReLU r t s) where
  type GradI (ReLU r t s) = ReLU (Ins r) t s 
instance (TensorOp r, T s t, JNT r, GNT r) => Parameterized (ReLU r t s) where
  type Input  (ReLU r t s) b = b
  type Output (ReLU r t s) a = a
  type Constr (ReLU r t s) a b = (a ~ r s t, b ~ r s t, Num (r s t))
  feed _ = relu

newtype LeakyReLU (r :: Z) t (s :: Shape) = LeakyReLU Rational deriving Generic
instance (JNT r, T s t) => Jitter (LeakyReLU r t s) where
  type JitT (LeakyReLU r t s) = LeakyReLU Tensor t s
instance (GNT r, T s t) => GradIn (LeakyReLU r t s) where
  type GradI (LeakyReLU r t s) = LeakyReLU (Ins r) t s
instance (T s t, TensorOp r, Fractional t, Num (r s t), JNT r, GNT r) => Parameterized (LeakyReLU r t s) where
  type Input  (LeakyReLU r t s) _ = r s t
  type Output (LeakyReLU r t s) _ = r s t
  type Constr (LeakyReLU r t s) _ _ = ()
  feed (LeakyReLU alpha) = leakyrelu $ splat $ fromRational alpha

instance (Jitter a, Jitter b) => Jitter (a >> b) where
  type JitT (a >> b) = JitT a >> JitT b
instance (GradIn a, GradIn b) => GradIn (a >> b) where
  type GradI (a >> b) = GradI a >> GradI b
instance (Parameterized f, Parameterized g) => Parameterized (f >> g) where
  type Input  (f >> g) b = Input f (Input g b)
  type Output (f >> g) a = Output g (Output f a)
  type Constr (f >> g) a b = (Constr f a (Output f a), Output f a ~ Input g b, Constr g (Input g b) b)

  feed (f :>: g) = feed g . feed f
instance (Jitter a, Jitter b) => Jitter (a !! b) where
  type JitT (a !! b) = JitT a !! JitT b
instance (GradIn a, GradIn b) => GradIn (a !! b) where
  type GradI (a !! b) = GradI a !! GradI b
instance (Parameterized f, Parameterized g) => Parameterized (f !! g) where 
  type Input  (f !! g) b = Input f b
  type Output (f !! g) a = Output g a
  type Constr (f !! g) a b = (Constr f a b, Input g b ~ a, b ~ Output f a, Constr g a b, Num b)

  feed (f :!: g) x = feed f x + feed g x

data a >> b = a :>: b deriving Generic
infixr 9 >>, :>:
data a !! b = a :!: b deriving Generic
infixr 9 !!, :!:
