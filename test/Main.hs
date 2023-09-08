{-# LANGUAGE DataKinds #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE OverloadedLists #-}
module Main (main) where

import HAX.Tensor
import HAX.PjRt 
import HAX.AD
import HAX.AD.Reverse
import HAX.Target

import Data.Proxy

test1 :: Target Tracer '[5] Float -> Target Tracer '[5] Float -> Target Tracer '[5] Float 
test1 = (+)
test2 :: Target Tracer '[5] Float -> Target Tracer '[5] Float -> Target Tracer '[5] Float 
test2 = (-)
test3 :: Target Tracer '[5] Float -> Target Tracer '[5] Float -> Target Tracer '[5] Float 
test3 = (*)

test4 :: Target Tracer '[5] Float -> Target Tracer '[5] Float -> Target Tracer '[5] Float 
test4 = vmap (+)
-- test5 :: Target Tracer '[5] Float -> Target Tracer '[7] Float -> Target Tracer '[5, 7] Float
-- test5 = prod
-- 
-- test6 :: Target Tracer '[8, 4, 7] Float -> Target Tracer '[4] Float
-- test6 x = reduceAdd x (Proxy :: Proxy '[0, 2])

test5 :: Target Tracer '[5, 3] Float -> Target Tracer '[5, 3] Float -> Target Tracer '[5, 3] Float
test5 = vmap (\ a b -> 
  let c = a + b
      d = c + a
      e = d + c
      f = d + d
      g = f + e
  in  g)

test6 :: Target Tracer '[2, 3, 4] Float -> Target Tracer '[2, 3, 4] Float -> Target Tracer '[2, 3, 4] Float
test6 = vmap (\ a b -> a - vmap (+) a b)

test7 :: Target Tracer '[2, 3] Float -> Target Tracer '[3] Float -> Target Tracer '[2, 3] Float
test7 x y = vmap (+ y) x

test8 :: Target Tracer '[4, 5] Float -> Target Tracer '[7, 4, 3, 5] Float 
test8 = (`broadcast` (Proxy :: Proxy '[1, 3]))

test9 :: Target Tracer '[5] Float -> Target Tracer '[4, 2, 5] Float
test9 = broadcast'

test10 :: Target Tracer '[5, 2] Float -> Target Tracer '[5, 4, 2, 7] Float
test10 = vmap (`broadcast` (Proxy :: Proxy '[1]))

test11 :: Target Tracer '[2, 5, 3] Float -> Target Tracer '[3, 7] Float -> Target Tracer '[2, 5, 7] Float 
test11 x y = vmap (`matmul` y) x

test12 :: Target Tracer [5, 2] Float -> Target Tracer [5, 2] Float -> Target Tracer '[2] Float -> Target Tracer [5, 2] Float
test12 x y z = 
  vmap (\ a b -> 
    a + vmap (+) b z) x y

test13 :: Target Tracer [5, 2] Float -> Target Tracer '[2] Float -> Target Tracer [5, 2] Float
test13 x y = vmap (const y) x

test14 :: Target Tracer [5, 2] Float -> Target Tracer [2, 5] Float
test14 operand = broadcast operand (Proxy :: Proxy '[1, 0])

-- differentiation test
type RTracer = Reverse Tracer
test15 :: RTracer '[4, 3, 6] Float -> RTracer '[4, 3, 6] Float 
test15 = negate

test16 :: RTracer '[4] Float -> RTracer '[3, 4, 5] Float 
test16 = (`broadcast` (Proxy :: Proxy '[1]))

test17 :: RTracer '[4] Float -> RTracer '[3] Float -> RTracer '[4, 3] Float 
test17 = prod

-- vmap differentiablity
test18 :: Target (Reverse Tracer) '[5, 3] Float -> Target (Reverse Tracer) '[5, 3] Float -> Target (Reverse Tracer) '[5, 3] Float
test18 = vmap (+)

test19 :: Target (Reverse Tracer) '[5, 3] Float -> Target (Reverse Tracer) '[5, 3] Float -> Target (Reverse Tracer) '[5, 3] Float
test19 = (+)

test20 :: Target (Reverse Tracer) [10, 5, 6] Float
               -> Target (Reverse Tracer) [10, 5, 2] Float
               -> Target (Reverse Tracer) [6, 2] Float
               -> Target (Reverse Tracer) [10, 5, 2] Float 
test20 x y z = 
  vmap (\ (a :: Target (Reverse Tracer) '[5, 6] Float) (b :: Target (Reverse Tracer) '[5, 2] Float) -> 
    matmul a z + b) 
      x (y :: Target (Reverse Tracer) '[10, 5, 2] Float)


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

  traceDebugGrad test18
  traceDebugGrad test19

  traceDebug     test4
  traceDebugGrad test20

  clientDestroy client
  return ()
