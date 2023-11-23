{-# LANGUAGE DataKinds #-}
{-# LANGUAGE NoStarIsType #-}
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
import Data.Proxy 

import System.Random (mkStdGen)
import System.IO (hFlush, stdout)

type R = Target (Reverse Tracer)
type R' = Tracer
type B = 125
-- type FFW r t = Reshape [28, 28] '[28 * 28] >> Dense r t (28 * 28) 392 >> ReLU >> Dense r t 392 98  >> ReLU >> Dense r t 98 10 >> Softmax
type ConvNet r t = Reshape [28, 28] [28, 28, 1] >> Convolute r t 1 [5, 5] 3 >> ReLU >> Convolute r t 3 [5, 5] 6 >> ReLU >> Convolute r t 6 [5, 5] 12 >> ReLU >>
                   Convolute r t 12 [5, 5] 24 >> ReLU >> Convolute r t 24 [5, 5] 48 >> ReLU >> Convolute r t 48 [5, 5] 96 >> ReLU >> Convolute r t 96 [4, 4] 192 >> 
                   Reshape [1, 1, 192] '[192] >> ReLU >> Dense r t 192 96 >> ReLU >> Dense r t 96 10 >> Softmax

model :: (r ~ R, t ~ Float, KnownNat b) => ConvNet r t -> r [b, 28, 28] t -> r [b, 10] t
model params = vmap (feed params)

mloss :: (r ~ R, t ~ Float, b ~ B) => ConvNet r t -> r [b, 28, 28] t -> r [b, 10] t -> r '[] t
mloss params x = (`crossEntropy` model params x)

mgrad :: (r ~ R', t ~ Float, b ~ B) => ConvNet r t -> r [b, 28, 28] t -> r [b, 10] t -> ConvNet r t
mgrad params x y = params'
  where params' :&: _ :&: _ = grad mloss params x y

-- TODO: Make this less dumb
train :: [([(Tensor [B, 28, 28] Float, Tensor [B, 10] Float)], [(Tensor [B, 28, 28] Float, Tensor '[B] Int64)])] ->
         (ConvNet Tensor Float -> Tensor [B, 28, 28] Float -> Tensor [B, 10] Float -> (Tensor '[] Float, ConvNet Tensor Float), 
          ConvNet Tensor Float -> Tensor [B, 28, 28] Float -> Tensor [B, 10] Float) ->
          ConvNet Tensor Float -> IO (ConvNet Tensor Float)
train []     _         params = return params
train ((trainSet, testSet):bs) (criterion, predict) params = do 
  params' <- epoch params trainSet
  let test = validate testSet params'
  putStrLn ("Epoch test: " ++ show test ++ "/10000")
  train bs (criterion, predict) params'
  where epoch :: ConvNet Tensor Float -> [(Tensor [B, 28, 28] Float, Tensor [B, 10] Float)] -> IO (ConvNet Tensor Float)
        epoch p []         = return p
        epoch p ((i, l):d) = do 
          let (loss, gradient) = criterion p i l 
          print loss
          hFlush stdout
          epoch (step 1.5e-2 p gradient) d
        validate :: [(Tensor [B, 28, 28] Float, Tensor '[B] Int64)] -> ConvNet Tensor Float -> Int64
        validate []          _ = 0
        validate ((i, l):as) p =
          let l' = argmax (predict p i) (Proxy :: Proxy 1)
              correct :: Int64 = getScalar $ reduceAdd' $ select 0 1 $ l' `isEQ` l
          in  correct + validate as p

main :: IO ()
main = do 
  trainImages :: Tensor [60000, 28, 28] Float <- (/255) . convert . (toTensor' :: Tensor' -> Tensor [60000, 28, 28] Word8) <$> readIDXFile' "test/data/MNIST/train-images.idx3-ubyte"
  trainLabels :: Tensor [60000, 10]     Float <- select 0 1 . onehot . convert . (toTensor' :: Tensor' -> Tensor '[60000] Word8) <$> readIDXFile' "test/data/MNIST/train-labels.idx1-ubyte"
  
  testImages :: Tensor [10000, 28, 28] Float <- (/255) . convert . (toTensor' :: Tensor' -> Tensor [10000, 28, 28] Word8) <$> readIDXFile' "test/data/MNIST/t10k-images.idx3-ubyte"
  testLabels :: Tensor '[10000]        Int64 <- convert . (toTensor' :: Tensor' -> Tensor '[10000] Word8) <$> readIDXFile' "test/data/MNIST/t10k-labels.idx1-ubyte"

  let epoch = 16
      trainImageBatches :: [Tensor [B, 28, 28] Float] = split trainImages
      trainLabelBatches :: [Tensor [B, 10] Float]     = split trainLabels
      testImageBatches  :: [Tensor [B, 28, 28] Float] = split testImages
      testLabelBatches  :: [Tensor '[B] Int64]        = split testLabels

      params :: ConvNet Tensor Float = rand (mkStdGen 35)
      criterion = jit $ 
        let g = fgrad mloss
            x (a :&: _ :&: _) = a
        in  \p f l -> second x (g p f l)
      predict :: (r ~ Tensor, t ~ Float, KnownNat b) => ConvNet r t -> r [b, 28, 28] t -> r [b, 10] t
      predict = jit model
  _ <- train (replicate epoch (zip trainImageBatches trainLabelBatches, zip testImageBatches testLabelBatches)) (criterion, predict) params
  echoNumCompilations
  return ()
