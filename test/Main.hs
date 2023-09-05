{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
module Main (main) where

import HAX.Tensor 
import HAX.PjRt
import Data.Data
import HAX (compile)
import HAX.AD.Reverse (Reverse)
import HAX.AD (rgrad)

-- Sharing test
test0 :: Tracer '[3, 5, 2] Float -> Tracer '[3, 5, 2] Float -> Tracer '[3, 5, 2] Float
test0 x y = d
  where a = x + y
        b = a + a
        c = b + b
        d = c + c

test1 :: Tracer '[5, 5] Float -> Tracer '[5, 5] Float -> Tracer '[5, 5] Float -> Tracer '[5, 5] Float
test1 x y z = e
  where a = x + y
        b = z * x
        c = a * z
        d = b - y
        e = c / d
-- broadcast test
test2 :: Tracer '[4] Float -> Tracer '[7, 4] Float -> Tracer '[7, 4] Float
test2 x y = broadcast' x + y

-- Vmap test
test3 :: Tracer '[7, 5] Float -> Tracer '[7, 8] Float -> Tracer '[7, 5, 8] Float
test3 x y = signum $ vmap (\ a b -> negate (prod a b)) x y

test4 :: Tracer '[7, 3] Float -> Tracer '[7, 3] Float -> Tracer '[3] Float -> Tracer '[7, 3] Float
test4 x y z = vmap (\ a b -> a * b + z) x y

test5 i j k = vmap (\ a (b :: Tracer '[3] Float) -> vmap (\ c d -> c + d - k) a b) (i :: Tracer '[5, 3] Float) (j :: Tracer '[5, 3] Float)
test8 (i :: Tracer '[] Float) = broadcast' k  :: Tracer '[7, 5] Float
  where k = broadcast' i :: Tracer '[5] Float

test6 :: Tracer '[] Float -> Tracer '[] Float
test6 = id

test7 :: Tracer '[] Float -> Tracer '[5, 5] Float
test7 = broadcast'

test9 :: Tracer '[2, 5] Float -> Tracer '[2, 5] Float -> Tracer '[5] Float -> Tracer '[5] Float -> Tracer '[2, 5] Float
test9 a b c d = broadcast' (c + d) + vmap (+) a b

-- Reduction test
test10 :: Tracer '[5, 2, 7, 2] Float -> Tracer '[5, 7] Float
test10 x = reduceAdditive x (Proxy :: Proxy '[1, 3])

-- Gradient 
test11 :: Reverse Tracer '[5] Float -> Reverse Tracer '[5] Float -> Reverse Tracer '[5] Float
test11 = (+)



main :: IO ()
main = do 
  let t0 :: Tensor '[2, 5, 3] Float 
      t0 = [[[1, 4, 6], [2, 6, -1], [9, 4, 1], [5, -4, -5], [3, -2, -9]], [[1, 4, 6], [2, 6, -1], [9], [0, -4, -5], [3, -2 ]]]
  
  print t0 
  
  traceDebug test0
  traceDebug test1

  traceDebug test2
  
  traceDebug test3
  traceDebug test4
  traceDebug test5
  traceDebug test8

  traceDebug test6
  traceDebug test7
  traceDebug test9

  traceDebug test10

  traceDebug $ rgrad test11
  
  clientDestroy client
  return ()
