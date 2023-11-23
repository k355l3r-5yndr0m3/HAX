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

class Parameter p where
  step :: Double -> p -> p -> p
  default step :: (Generic p, GParameter (Rep p)) => Double -> p -> p -> p
  step size (from -> p) (from -> p') = to (gStep size p p')
  randM :: StatefulGen g m => g -> m p
  default randM :: (Generic p, GParameter (Rep p), StatefulGen g m) => g -> m p
  randM key = to <$> gRandM key

  rand :: RandomGen g => g -> p
  rand g = runStateGen_ g randM

class GParameter f where 
  gStep :: Double -> f x -> f x -> f x
  gRandM :: StatefulGen g m => g -> m (f x)
instance GParameter V1 where
  gStep _ x _ = x
  gRandM _ = return undefined
instance GParameter U1 where
  gStep _ x _ = x
  gRandM _ = return U1
instance TypeError (Text "Cannot handle Parameter type with multiple data constructors") => GParameter (a :+: b) where
  gStep = undefined
  gRandM = undefined
instance (GParameter f, GParameter g) => GParameter (f :*: g) where 
  gStep size (f :*: g) (f' :*: g') = gStep size f f' :*: gStep size g g'
  gRandM key = do 
    f <- gRandM key 
    g <- gRandM key 
    return $ f :*: g
instance Parameter c => GParameter (K1 i c) where
  gStep size (K1 c) (K1 c') = K1 $ step size c c'
  gRandM key = K1 <$> randM key
instance GParameter f => GParameter (M1 i t f) where
  gStep size (M1 f) (M1 f') = M1 $ gStep size f f'
  gRandM key = M1 <$> gRandM key

instance T s t => Parameter (Tensor s t) where
  step = updateParameter
  randM key = do
    createBuffer <- tensorialUniformM nelem key
    return $ unsafePerformIO $ do 
      buffer <- createBuffer 
      tensorFromHostBufferGC defaultDevice buffer
    where nelem = product $ fromInteger <$> shapeVal (Proxy :: Proxy s)

instance (Parameter a, Parameter b) => Parameter (a, b)
instance (Parameter a, Parameter b) => Parameter (a <&> b)
instance (Tensorial t, KnownNat i, KnownNat o) => Parameter (Dense Tensor t i o)
instance Parameter (Reshape a b)
instance Parameter Sigmoid
instance Parameter ReLU
instance Parameter Softmax
instance (Tensorial t, KnownShape s, KnownNat i, KnownNat o, KnownShape (s :+ o)) => Parameter (Convolute Tensor t i s o) where

instance (Parameter f, Parameter g) => Parameter (f >> g)
instance (Parameter f, Parameter g) => Parameter (f !! g)
