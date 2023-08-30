{-# LANGUAGE DataKinds #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
module Main (main) where
import Foreign

import HAX.Tensor 
import HAX.PjRt
import HAX.Jit

import HAX.AD
import HAX.AD.Reverse

type TestDim = '[2]

testing :: Num a => a -> a -> a
testing x y = x * y + x

main :: IO ()
main = do 
  device <- head <$> clientAddressableDevices client
  a :: Tensor TestDim Float <- withArray (replicate (12 * 12) 5)  $ tensorFromHostBuffer device
  b :: Tensor TestDim Float <- withArray (replicate (12 * 12) 10)  $ tensorFromHostBuffer device
  
  let tested :: Jit' (Tracer TestDim Float -> Tracer TestDim Float -> Tracer TestDim Float)
      tested = jit testing
      grad   = rgrad (testing :: (a ~ Tracer TestDim Float) => Reverse a -> Reverse a -> Reverse a)
      gradtest = jit grad

  traceDebug grad
  
  let c = tested a b
      d = tested a c
      e = tested c d

  print $ gradtest a b
  print $ gradtest a c
  print $ gradtest e d

  clientDestroy client
  return ()
