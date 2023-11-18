{-# LANGUAGE DataKinds #-}
{-# LANGUAGE NoStarIsType #-}
module Main(main) where
import GHC.TypeLits

import HAX.Tensor
import HAX.Target
import HAX.IO
import HAX.NN

import Data.Word
import HAX.Jit (echoNumCompilations)
import HAX.AD

type R s t = Target (Reverse Tracer) s t
type Model r t = Reshape r t [28, 28] '[28 * 28] >> Dense r t (28 * 28) 392 >> ReLU r t '[392] >> Dense r t 392 196 >> ReLU r t '[196] >> Dense r t 196 98 >> ReLU r t '[98]

main :: IO ()
main = do 
  trainImages :: Tensor [60000, 28, 28] Float <- convert . (toTensor' :: AnyTsr -> Tensor [60000, 28, 28] Word8) <$> readIDXFile' "test/data/MNIST/train-images.idx3-ubyte"
  trainLabels :: Tensor [60000, 10]     Bool  <- onehot . convert . (toTensor' :: AnyTsr -> Tensor '[60000] Word8) <$> readIDXFile' "test/data/MNIST/train-labels.idx1-ubyte"
  
  let trainImageBatches :: [Tensor [2500, 28, 28] Float] = split trainImages
      trainLabelBatches :: [Tensor [2500, 10] Bool]      = split trainLabels


  echoNumCompilations
  return ()
