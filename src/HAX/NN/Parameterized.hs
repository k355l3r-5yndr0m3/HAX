{-# LANGUAGE TypeFamilyDependencies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE DataKinds #-}
module HAX.NN.Parameterized where
import HAX.Tensor.Tensorial

import GHC.TypeLits

class Parameterized p where
  type Input p 
  type Output p
  feed :: p -> Input p -> Output p

instance Parameterized (a -> b) where 
  type Input (a -> b) = a
  type Output (a -> b) = b
  feed = id

data Dense r (i :: Nat) (o :: Nat) t = Dense (r [i, o] t) (r '[o] t) 
instance (MathOp r, MathConstr r t, Num (r '[o] t), KnownNat i, KnownNat o, Tensorial t, Num t) => Parameterized (Dense r i o t) where
  type Input (Dense r i o t)  = r '[i] t
  type Output (Dense r i o t) = r '[o] t
  feed (Dense weights biases) feats = biases + linearMap weights feats

-- Activation functions
data Sigmoid r (s :: Shape) t = Sigmoid
instance Floating (r s t) => Parameterized (Sigmoid r s t) where
  type Input (Sigmoid r s t)  = r s t
  type Output (Sigmoid r s t) = r s t
  feed _ = sigmoid

data ReLU r (s :: Shape) t = Relu
instance (SelectOp r t, OrderOp r, T s t, Num (r s t)) => Parameterized (ReLU r s t) where
  type Input (ReLU r s t)  = r s t
  type Output (ReLU r s t) = r s t
  feed _ = relu

newtype LeakyReLU r (s :: Shape) t = LeakyReLU (r '[] t)
instance (Num (r s t), ShapeConstr r t, SelectOp r t, KnownShape s, OrderOp r, ShapeOp r) => Parameterized (LeakyReLU r s t) where
  type Input (LeakyReLU r s t)  = r s t
  type Output (LeakyReLU r s t) = r s t
  feed (LeakyReLU alpha) = leakyrelu alpha
