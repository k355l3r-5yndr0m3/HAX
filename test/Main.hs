{-# LANGUAGE DataKinds #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
module Main (main) where
import Foreign

import HAX.Tensor 
import HAX.PjRt
import HAX.Jit

import HAX.AD
import HAX.AD.Reverse
import HAX.AD.Gradient

import Data.Dynamic

testing :: Num a => a -> a -> a
testing x y = x * y + x

main :: IO ()
main = do 
  device <- head <$> clientAddressableDevices client
  a :: Tensor '[12, 12] Float <- withArray (replicate (12 * 12) 5)  $ tensorFromHostBuffer device
  b :: Tensor '[12, 12] Float <- withArray (replicate (12 * 12) 10)  $ tensorFromHostBuffer device
  
  let tested :: Jit' (Tracer '[12, 12] Float -> Tracer '[12, 12] Float -> Tracer '[12, 12] Float)
      tested = jit testing
      -- test2 :: Jit' (Tracer '[12, 12] Float -> Tracer '[12, 12] Float -> Tracer '[12, 12] Float)
      -- test2  = jit tested
--      gradtest = jit $ rgrad (testing :: (a ~ Tracer '[12, 12] Float) => Reverse a -> Reverse a -> Reverse a)

  
  print $ tested a b
--  print $  a b

  clientDestroy client
  return ()
