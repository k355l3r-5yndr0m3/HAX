{-# LANGUAGE DataKinds #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE OverloadedLists #-}
module Main (main) where

import HAX.Tensor
import HAX.PjRt 
import HAX.AD
import HAX.AD.Reverse

import Data.Proxy
import GHC.IsList

test1 :: Target '[5] Float -> Target '[5] Float -> Target '[5] Float 
test1 = (+)
test2 :: Target '[5] Float -> Target '[5] Float -> Target '[5] Float 
test2 = (-)
test3 :: Target '[5] Float -> Target '[5] Float -> Target '[5] Float 
test3 = (*)

-- test5 :: Target '[5] Float -> Target '[7] Float -> Target '[5, 7] Float
-- test5 = prod
-- 
-- test6 :: Target '[8, 4, 7] Float -> Target '[4] Float
-- test6 x = reduceAdd x (Proxy :: Proxy '[0, 2])

test5 :: Target '[5, 3] Float -> Target '[5, 3] Float -> Target '[5, 3] Float
test5 = vmap (\ a b -> 
  let c = a + b
      d = c + a
      e = d + c
      f = d + d
      g = f + e
  in  g)

test6 :: Target '[2, 3, 4] Float -> Target '[2, 3, 4] Float -> Target '[2, 3, 4] Float
test6 = vmap (\ a b -> a - vmap (+) a b)

test7 :: Target '[2, 3] Float -> Target '[3] Float -> Target '[2, 3] Float
test7 x y = vmap (+ y) x

test8 :: Target '[4, 5] Float -> Target '[7, 4, 3, 5] Float 
test8 = (`broadcast` (Proxy :: Proxy '[1, 3]))

test9 :: Target '[5] Float -> Target '[4, 2, 5] Float
test9 = broadcast'

test10 :: Target '[5, 2] Float -> Target '[5, 4, 2, 7] Float
test10 = vmap (`broadcast` (Proxy :: Proxy '[1]))

test11 :: Target '[2, 5, 3] Float -> Target '[3, 7] Float -> Target '[2, 5, 7] Float 
test11 x y = vmap (`matmul` y) x

test12 :: Target [5, 2] Float -> Target [5, 2] Float -> Target '[2] Float -> Target [5, 2] Float
test12 x y z = 
  vmap (\ a b -> 
    a + vmap (+) b z) x y

test13 :: Target [5, 2] Float -> Target '[2] Float -> Target [5, 2] Float
test13 x y = vmap (const y) x

test14 :: Target [5, 2] Float -> Target [2, 5] Float
test14 operand = broadcast operand (Proxy :: Proxy '[1, 0])

-- differentiation test
type RTracer = Reverse Tracer
test15 :: RTracer '[4, 3, 6] Float -> RTracer '[4, 3, 6] Float 
test15 = negate

test16 :: RTracer '[4] Float -> RTracer '[3, 4, 5] Float 
test16 = (`broadcast` (Proxy :: Proxy '[1]))

test17 :: RTracer '[4] Float -> RTracer '[3] Float -> RTracer '[4, 3] Float 
test17 = prod



traceDebugGrad :: (Rev (GradResult f) f ~ (a -> b), Traceable (a -> b), ReverseMode f) => f -> IO () 
traceDebugGrad x = traceDebug $ rgrad x
main :: IO ()
main = do 
  traceDebug test1
  traceDebug test2
  traceDebug test3
  -- traceDebug test5
  -- traceDebug test6
  traceDebug test5
  traceDebug test6
  traceDebug test7
  traceDebug test8
  traceDebug test9
  traceDebug test10
  traceDebug test11

  traceDebug test12
  traceDebug test13

  traceDebug test14

--  let tensor :: Tensor '[5, 5] Float = fromList [[i..i+4] | i <- [0..4]]
--  print tensor

  traceDebugGrad test15
  traceDebugGrad test16
  traceDebugGrad test17

  clientDestroy client
  return ()
