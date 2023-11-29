{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
module Main (main) where

import Codec.Picture
import HAX.IO 
import HAX.Tensor
import Data.Word
import Data.Maybe (fromJust)

main :: IO ()
main = do
  either putStrLn (\(tensorFromImage . convertRGB8 -> image :: Tensor [256, 256, 3] Word8) -> do
    writePng "test/data/output-image.png" (imageFromTensor image)) =<< readImage "test/data/image.jpg"  
  maybe (putStrLn "Failed to read idx") (\(fromJust . toTensor -> img :: Tensor [60000,28,28] Word8) -> 
    let img' :: Tensor [28, 28] Word8 = reshape $ img @% 59999
    in  writePng "test/data/mnist-test.png" (imageFromTensor img')) =<< readIDXFile "test/data/MNIST/train-images.idx3-ubyte"
