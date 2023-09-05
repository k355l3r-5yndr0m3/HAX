{-# LANGUAGE DataKinds #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
module Main (main) where

import HAX.Tensor
import Data.Reflection

main :: IO ()
main = do 
  reify [12, 4] (print . reflect)
  -- let device = head $ clientAddressableDevices client
  -- a :: Tensor TestDim Float <- withArray (replicate (12 * 12) 5)  $ tensorFromHostBuffer device
  -- b :: Tensor TestDim Float <- withArray (replicate (12 * 12) 10)  $ tensorFromHostBuffer device
  -- 
  -- let tested :: Jit' (Tracer TestDim Float -> Tracer TestDim Float -> Tracer TestDim Float)
  --     tested = jit testing
  --     grad   = rgrad (testing :: (a ~ Tracer TestDim Float) => Reverse a -> Reverse a -> Reverse a)
  --     gradtest = jit grad
  --     
  -- 
  -- traceDebug grad
  -- 
  -- let c = tested a b
  --     d = tested a c
  --     e = tested c d
  --     g :: Tensor '[2, 2, 2, 2] Float = broadcast c (Proxy :: Proxy '[1])
  --     j :: Tensor '[2, 2, 2, 4, 2] Float = broadcast g (Proxy :: Proxy '[0, 1, 2, 4])
  -- 
  -- print $ gradtest a b
  -- print $ gradtest a c
  -- print $ gradtest e d
  -- print g
  -- print j
  -- print $ a * b
  -- traceDebug (\ (x :: Tracer '[2, 3] Float) (y :: Tracer '[4, 7] Float) -> x |#| y)
  -- print $ a |#| b
  -- 
  -- clientDestroy client
  return ()
