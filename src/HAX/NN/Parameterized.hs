{-# LANGUAGE TypeFamilyDependencies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE DataKinds #-}
module HAX.NN.Parameterized where
import HAX.Tensor.Tensorial

import GHC.TypeLits

class Parameterized p where
  type Input p 
  type Output p
  forward :: p -> Input p -> Output p

instance Parameterized (a -> b) where 
  type Input (a -> b) = a
  type Output (a -> b) = b
  forward = id

data Linear r (i :: Nat) (o :: Nat) t = Linear (r '[i, o] t) (r '[o] t) 
instance (TensorOp r t, Num (r '[o] t), KnownNat i, KnownNat o) => Parameterized (Linear r i o t) where
  type Input (Linear r i o t) = r '[i] t
  type Output (Linear r i o t) = r '[o] t
  forward (Linear weights biases) input = biases + linearMap weights input

data Sigmoid r (s :: Shape) t = Sigmoid
instance Floating (r s t) => Parameterized (Sigmoid r s t) where
  type Input (Sigmoid r s t) = r s t
  type Output (Sigmoid r s t) = r s t
  forward _ input = recip $ 1 + exp (negate input)

data Relu r (s :: Shape) t = Relu
