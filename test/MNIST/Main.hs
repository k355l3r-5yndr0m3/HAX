{-# LANGUAGE DataKinds #-}
module Main(main) where

import HAX.Tensor
import HAX.IO

import Data.Word

main :: IO ()
main = do 
  trainImages :: Tensor [60000, 28, 28] Float <- toTensor' <$> readIDXFile' "test/data/MNIST/train-images.idx3-ubyte"
  trainLabels :: Tensor [60000, 10]     Bool  <- unsafeOnehot . toTensor' <$> readIDXFile' "test/data/MNIST/train-labels.idx1-ubyte"
  

  return ()
