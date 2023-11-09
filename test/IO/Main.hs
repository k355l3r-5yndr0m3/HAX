{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE OverloadedLists #-}
module Main (main) where

import Codec.Picture
import HAX.IO 
import HAX.Tensor
import Data.Word

main :: IO ()
main = do
  either putStrLn (\(tensorFromImage . convertRGB8 -> image :: Tensor [256, 256, 3] Word8) -> do
    print image
    writePng "test/data/output-image.png" (imageFromTensor image)) =<< readImage "test/data/image.jpg"  
