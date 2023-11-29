{-# LANGUAGE DataKinds #-}
module Main where
import HAX
import HAX.IO
import HAX.Tensor

import Data.Word
import Data.Int

main :: IO ()
main = do
  trainImages :: Tensor [60000, 28, 28] Float <- (/255) . convert . (toTensor' :: Tensor' -> Tensor [60000, 28, 28] Word8) <$> readIDXFile' "test/data/MNIST/train-images.idx3-ubyte"
  trainLabels :: Tensor [60000, 10]     Float <- select 0 1 . onehot . convert . (toTensor' :: Tensor' -> Tensor '[60000] Word8) <$> readIDXFile' "test/data/MNIST/train-labels.idx1-ubyte"

  testImages :: Tensor [10000, 28, 28] Float <- (/255) . convert . (toTensor' :: Tensor' -> Tensor [10000, 28, 28] Word8) <$> readIDXFile' "test/data/MNIST/t10k-images.idx3-ubyte"
  testLabels :: Tensor '[10000]        Int64 <- convert . (toTensor' :: Tensor' -> Tensor '[10000] Word8) <$> readIDXFile' "test/data/MNIST/t10k-labels.idx1-ubyte"

  sequence_ (fmap print (split trainImages :: [Tensor [60, 28, 28] Float]))

  return ()
