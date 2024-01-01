{-# LANGUAGE DataKinds #-}
{-# LANGUAGE NoStarIsType #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
module Main where
import Criterion.Main

import Data.Word

import HAX.NN
import HAX.IO
import HAX.Tensor
import HAX.Utils
import HAX.AD.Reverse
import HAX.Target

import GHC.TypeLits
import System.Random


type FFW r t = Reshape [28, 28] '[28 * 28] >> Dense r t (28 * 28) 392 >> ReLU >> Dense r t 392 196 >> ReLU >> Dense r t 196 10 >> Softmax
type NN r t = FFW r t

type B = 100
type R = Target (Reverse Tracer)

model :: (r ~ R, t ~ Float, KnownNat b) => NN r t -> r [b, 28, 28] t -> r [b, 10] t
model params = vmap (feed params)

mloss :: (r ~ R, t ~ Float, KnownNat b) => NN r t -> r [b, 28, 28] t -> r [b, 10] t -> r '[] t
mloss params x = (`crossEntropy` model params x)

criterion :: forall n. KnownNat n => NN Tensor Float -> Tensor [n, 28, 28] Float -> Tensor [n, 10] Float -> NN Tensor Float
criterion = jit gradient'
  where gradient  = grad mloss
        gradient' params x y = 
          let g :&: _ :&: _ = gradient params x y 
          in  g

train :: forall n' n. (n ~ 60000, KnownNat n', Split [60000, 28, 28] [n', 28, 28], Split [60000, 10] [n', 10]) => 
          (NN Tensor Float -> Tensor [n', 28, 28] Float -> Tensor [n', 10] Float -> NN Tensor Float, 
          Tensor [n, 28, 28] Float, Tensor [n, 10] Float, NN Tensor Float) -> NN Tensor Float
train (crit, features, lables, params) = foldlSplit2 (\p x (y :: Tensor [n', 10] Float)  -> step 1.8e-2 p (crit p x y)) params features lables

main :: IO ()
main = do 
  trainImages :: Tensor [60000, 28, 28] Float <- (/255) . convert . (toTensor' :: Tensor' -> Tensor [60000, 28, 28] Word8) <$> readIDXFile' "test/data/MNIST/train-images.idx3-ubyte"
  trainLabels :: Tensor [60000, 10]     Float <- select 0 1 . onehot . convert . (toTensor' :: Tensor' -> Tensor '[60000] Word8) <$> readIDXFile' "test/data/MNIST/train-labels.idx1-ubyte"
  let initial = rand (mkStdGen 3512) :: NN Tensor Float
  defaultMain [bgroup "FFW-784-391-196-10" [ bench "batch_size=100" $ whnf (train @100) (criterion, trainImages, trainLabels, initial), 
                                             bench "batch_size=200" $ whnf (train @200) (criterion, trainImages, trainLabels, initial), 
                                             bench "batch_size=600" $ whnf (train @600) (criterion, trainImages, trainLabels, initial)]]
  
