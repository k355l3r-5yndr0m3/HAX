{-# LANGUAGE TypeFamilyDependencies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE DataKinds #-}
module HAX.NN.Parameterized where
import Data.Kind

import HAX.Jit
import HAX.Tensor.Tensor
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
type instance JitTransform (Dense r i o t) = Dense Tensor i o t
instance (TraceableElement (r [i, o] t), TraceableElement (r '[o] t)) => TraceableElement (Dense r i o t) where
  constructTracer i = (i'', Dense weights biases, d1 ++ d2) 
    where (i', weights, d1) = constructTracer i
          (i'', biases, d2) = constructTracer i'
  deconstructTracer (Dense weights biases) = (\ t0 -> do 
    (t1, _a) <- a' t0
    (t2, _b) <- b' t1
    return (t2, _a ++ _b), join aSig bSig)
    where (a', aSig) = deconstructTracer weights
          (b', bSig) = deconstructTracer biases
          join :: ([a], [b]) -> ([a], [b]) -> ([a], [b])
          join (_a, _b) (_c, _d) = (_a ++ _c, _b ++ _d)

instance (TensorOp r, Num (r '[o] t), KnownNat i, KnownNat o, Tensorial t, Num t) => Parameterized (Dense r i o t) where
  type Input (Dense r i o t)  = r '[i] t
  type Output (Dense r i o t) = r '[o] t
  feed (Dense weights biases) feats = biases + linearMap weights feats

-- Activation functions
data Sigmoid (r :: Shape -> Type -> Type) (s :: Shape) t = Sigmoid
type instance JitTransform (Sigmoid r s t) = Sigmoid Tensor s t
instance TraceableElement (Sigmoid r s t) where
  constructTracer i = (i, Sigmoid, [])
  deconstructTracer Sigmoid = (\i -> return (i, []), ([], []))
instance Floating (r s t) => Parameterized (Sigmoid r s t) where
  type Input (Sigmoid r s t)  = r s t
  type Output (Sigmoid r s t) = r s t
  feed _ = sigmoid

data ReLU (r :: Shape -> Type -> Type) (s :: Shape) t = ReLU
type instance JitTransform (ReLU r s t) = ReLU Tensor s t
instance TraceableElement (ReLU r s t) where
  constructTracer i = (i, ReLU, [])
  deconstructTracer ReLU = (\i -> return (i, []), ([], []))
instance (TensorOp r, T s t, Num (r s t)) => Parameterized (ReLU r s t) where
  type Input (ReLU r s t)  = r s t
  type Output (ReLU r s t) = r s t
  feed _ = relu

newtype LeakyReLU r (s :: Shape) t = LeakyReLU (r '[] t)
type instance JitTransform (LeakyReLU r s t) = LeakyReLU Tensor s t
instance TraceableElement (r '[] t) => TraceableElement (LeakyReLU r s t) where
  constructTracer i = (i', LeakyReLU a, dt)
    where (i', a, dt) = constructTracer i
  deconstructTracer (LeakyReLU r) = deconstructTracer r
instance (Num (r s t), T s t, TensorOp r) => Parameterized (LeakyReLU r s t) where
  type Input (LeakyReLU r s t)  = r s t
  type Output (LeakyReLU r s t) = r s t
  feed (LeakyReLU alpha) = leakyrelu alpha
  

