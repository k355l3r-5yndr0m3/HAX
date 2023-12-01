{-# LANGUAGE TypeFamilyDependencies #-}
{-# LANGUAGE DefaultSignatures #-}
{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE UndecidableInstances #-}
module HAX.NN.Parameterized where
import Data.Proxy
import HAX.Utils

import HAX.NN.Model
import HAX.Tensor
import HAX.PjRt (defaultDevice)

import GHC.IO.Unsafe (unsafePerformIO)
import GHC.Generics
import GHC.TypeLits

import System.Random.Stateful
class RandParam p where
  randM :: StatefulGen g m => g -> m p
  default randM :: (Generic p, RandParam' (Rep p), StatefulGen g m) => g -> m p
  randM key = to <$> gRandM key

  rand :: RandomGen g => g -> p
  rand g = runStateGen_ g randM

class RandParam' f where 
  gRandM :: StatefulGen g m => g -> m (f x)
instance RandParam' V1 where
  gRandM _ = return undefined
instance RandParam' U1 where
  gRandM _ = return U1
instance TypeError (Text "Cannot handle RandParam type with multiple data constructors") => RandParam' (a :+: b) where
  gRandM = undefined
instance (RandParam' f, RandParam' g) => RandParam' (f :*: g) where 
  gRandM key = do 
    f <- gRandM key 
    g <- gRandM key 
    return $ f :*: g
instance RandParam c => RandParam' (K1 i c) where
  gRandM key = K1 <$> randM key
instance RandParam' f => RandParam' (M1 i t f) where
  gRandM key = M1 <$> gRandM key

instance (T s t) => RandParam (Tensor s t) where
  randM key = do
    createBuffer <- tensorialUniformM nelem key
    return $ unsafePerformIO $ do 
      buffer <- createBuffer 
      tensorFromHostBufferGC defaultDevice buffer
    where nelem = product $ fromInteger <$> shapeVal (Proxy :: Proxy s)

instance (RandParam a, RandParam b) => RandParam (a, b)
instance (RandParam a, RandParam b) => RandParam (a <&> b)
instance (Tensorial t, KnownNat i, KnownNat o) => RandParam (Dense Tensor t i o)
instance RandParam (Reshape a b)
instance RandParam Sigmoid
instance RandParam ReLU
instance RandParam Softmax
instance (Tensorial t, KnownShape s, KnownNat i, KnownNat o, KnownShape (s :+ o)) => RandParam (Convolute Tensor t i s o) where

instance (RandParam f, RandParam g) => RandParam (f >> g)
instance (RandParam f, RandParam g) => RandParam (f !! g)

class Parameter p where
  step' :: Double -> p -> p -> p
  default step' :: (Generic p, Parameter' (Rep p)) => Double -> p -> p -> p
  step' size (from -> p) (from -> p') = to (gStep' size p p')
class Parameter' p where
  gStep' :: Double -> p x -> p x -> p x
instance Parameter' V1 where
  gStep' _ x _ = x
instance Parameter' U1 where
  gStep' _ x _ = x
instance TypeError (Text "Cannot handle Parameter type with multiple data constructors") => Parameter' (a :+: b) where
  gStep' = undefined
instance (Parameter' f, Parameter' g) => Parameter' (f :*: g) where 
  gStep' size (f :*: g) (f' :*: g') = gStep' size f f' :*: gStep' size g g'
instance Parameter c => Parameter' (K1 i c) where
  gStep' size (K1 c) (K1 c') = K1 $ step' size c c'
instance Parameter' f => Parameter' (M1 i t f) where
  gStep' size (M1 f) (M1 f') = M1 $ gStep' size f f'

instance (TensorOp r, T s t) => Parameter (r s t) where
  step' = updateParameter

instance (Parameter a, Parameter b) => Parameter (a, b)
instance (Parameter a, Parameter b) => Parameter (a <&> b)

type instance ReverseJit (Dense Tensor t i o) = Dense Tracer t i o
instance (Tensorial t, KnownNat i, KnownNat o, TensorOp r) => Parameter (Dense r t i o)
type instance ReverseJit (Reshape a b) = Reshape a b
instance Parameter (Reshape a b)
type instance ReverseJit Sigmoid = Sigmoid
instance Parameter Sigmoid
type instance ReverseJit ReLU = ReLU
instance Parameter ReLU
type instance ReverseJit Softmax = Softmax
instance Parameter Softmax
type instance ReverseJit (Convolute Tensor t i s o) = Convolute Tracer t i s o
instance (Tensorial t, KnownShape s, KnownNat i, KnownNat o, KnownShape (s :+ o), TensorOp r) => Parameter (Convolute r t i s o) where
type instance ReverseJit (f >> g) = ReverseJit f >> ReverseJit g
instance (Parameter f, Parameter g) => Parameter (f >> g)
type instance ReverseJit (f !! g) = ReverseJit f !! ReverseJit g
instance (Parameter f, Parameter g) => Parameter (f !! g)

step :: (ReverseJit (JitF p) ~ p, JitT p ~ JitF p, Jit' p, Jit p, Parameter p) => Double -> JitT p -> JitT p -> JitF p
step d = jitT (step' d)
