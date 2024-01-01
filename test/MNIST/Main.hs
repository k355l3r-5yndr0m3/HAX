{-# LANGUAGE DataKinds #-}
{-# LANGUAGE NoStarIsType #-}
{-# LANGUAGE ViewPatterns #-}
module Main(main) where
import GHC.TypeLits

import HAX.Tensor
import HAX.Target
import HAX.IO
import HAX.NN

import HAX.Jit (echoNumCompilations)
import HAX.AD
import HAX.Utils

import Data.Int
import Data.Word
import Data.Bifunctor

import System.Random (mkStdGen)
import System.IO (hFlush, stdout)
import Data.Foldable (forM_)

type R = Target (Reverse Tracer)
type R' = Tracer
type B = 200
type FFW r t = Reshape [28, 28] '[28 * 28] >> Dense r t (28 * 28) 392 >> ReLU >> Dense r t 392 98  >> ReLU >> Dense r t 98 10 >> Softmax
type ConvNet r t = Reshape [28, 28] [28, 28, 1] >> Convolute r t 1 [5, 5] 2 >> ReLU >> Convolute r t 2 [5, 5] 4 >> ReLU >> Convolute r t 4 [5, 5] 8 >> ReLU >>
                   Convolute r t 8 [5, 5] 16 >> ReLU >> Convolute r t 16 [5, 5] 32 >> ReLU >> Convolute r t 32 [5, 5] 64 >> ReLU >> Convolute r t 64 [4, 4] 128 >> 
                   Reshape [1, 1, 128] '[128] >> ReLU >> Dense r t 128 64 >> ReLU >> Dense r t 64 10 >> Softmax
-- type Test r t = Reshape [28, 28] '[28 * 28] >> Dense r t (28 * 28) 10 
type NN r t = ConvNet r t

model :: (r ~ R, t ~ Float, KnownNat b) => NN r t -> r [b, 28, 28] t -> r [b, 10] t
model params = vmap (feed params)

mloss :: (r ~ R, t ~ Float, b ~ B) => NN r t -> r [b, 28, 28] t -> r [b, 10] t -> r '[] t
mloss params x = (`crossEntropy` model params x)

-- TODO: Make this less dumb
train :: (n ~ 60000, n' ~ B) => Int -> (NN Tensor Float -> Tensor [n', 28, 28] Float -> Tensor [n', 10] Float -> (Tensor '[] Float, NN Tensor Float)) -> 
                                      Tensor [n, 28, 28] Float -> Tensor [n, 10] Float -> NN Tensor Float -> ([Float], NN Tensor Float)
train epoch criterion features lables params
  | epoch > 0 =
    let (reverse -> losses, params') = foldlSplit2 (\(h, p) x y -> 
          let (getScalar -> loss, gradient) = criterion p x y 
              p' = step 1.8e-2 p gradient
          in  (loss:h, p')) ([], params) features lables
    in  first (losses ++) (train (epoch - 1) criterion features lables params')
  | otherwise = ([], params)

main :: IO ()
main = do 
  trainImages :: Tensor [60000, 28, 28] Float <- (/255) . convert . (toTensor' :: Tensor' -> Tensor [60000, 28, 28] Word8) <$> readIDXFile' "test/data/MNIST/train-images.idx3-ubyte"
  trainLabels :: Tensor [60000, 10]     Float <- select 0 1 . onehot . convert . (toTensor' :: Tensor' -> Tensor '[60000] Word8) <$> readIDXFile' "test/data/MNIST/train-labels.idx1-ubyte"
  
  testImages :: Tensor [10000, 28, 28] Float <- (/255) . convert . (toTensor' :: Tensor' -> Tensor [10000, 28, 28] Word8) <$> readIDXFile' "test/data/MNIST/t10k-images.idx3-ubyte"
  testLabels :: Tensor '[10000]        Int64 <- convert . (toTensor' :: Tensor' -> Tensor '[10000] Word8) <$> readIDXFile' "test/data/MNIST/t10k-labels.idx1-ubyte"
  
  let predict :: (r ~ Tensor, t ~ Float, KnownNat b) => NN r t -> r [b, 28, 28] t -> r [b, 10] t
      predict = jit model
      criterion = jit $ 
        let g = fgrad mloss
            x (a :&: _ :&: _) = a
        in  \p f l -> second x (g p f l)
      logging = fst $ train 10 criterion trainImages trainLabels (rand (mkStdGen 10))
  forM_ logging (\a -> print a >> hFlush stdout)
  
  echoNumCompilations

  return ()

