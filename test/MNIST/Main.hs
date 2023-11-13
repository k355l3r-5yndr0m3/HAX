{-# LANGUAGE DataKinds #-}
module Main(main) where

import HAX.Tensor
import HAX.IO

import Data.Word

main :: IO ()
main = do 
  trainImages :: Tensor [60000, 28, 28] Float <- toTensor' <$> readIDXFile' "test/data/MNIST/train-images.idx3-ubyte"
  trainLabels :: Tensor [60000, 10]     Bool  <- onehot . convert . (toTensor' :: AnyTsr -> Tensor '[60000] Word8) <$> readIDXFile' "test/data/MNIST/train-labels.idx1-ubyte"
  
  let trainImageBatches :: [Tensor [2500, 28, 28] Float] = split trainImages
      trainLabelBatches :: [Tensor [2500, 10] Bool]      = split trainLabels
 
  return ()
