{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedLists #-}
module Main where
import HAX.Target
import HAX.Tensor
import HAX.Utils

import HAX.AD.Reverse
import HAX.AD.Numerical
import HAX.Jit

import Data.Int
import GHC.TypeLits
-- import HAX.AD (traceDebugGrad)


type R = Target (Reverse Tracer)
test1 :: R '[2, 4] Float -> R '[2, 4] Float -> R '[2, 4] Float -> R '[] Float
test1 x y z = reduceAdd' $ y * x - z

test2 :: R '[] Bool -> R '[2, 4] Float -> R '[2, 4] Float -> R '[] Float 
test2 k x y = reduceAdd' $ branch (x - y) (x * y) k

test3 :: R '[2, 2] Float -> R '[2, 2] Bool -> R '[] Float
test3 x = reduceAdd' . select 0 x

test4 :: R '[4, 2] Float -> R '[4] Bool -> R '[] Float 
test4 x = reduceAdd' . vmap (branch 0) x

type K = Reverse Tracer
test5 :: K [5, 5] Float -> K '[5] Float
test5 matrix = unsafeGather matrix starts [] [0, 1] [0, 1] 1 [1, 1]
  where starts :: K '[5, 2] Int64 = unsafeIota 0

test6 :: K [5, 7, 5] Float -> K '[2, 5] Float
test6 operand = unsafeGather operand starts [0] [0, 2] [0, 2] 1 [0, 2, 0]
  where starts :: K '[5, 2] Int64 = unsafeIota 0

test7 :: K '[5] Float -> K [5, 5] Float
test7 diag = unsafeIota 0 * unsafeScatter 0 starts diag [] [0, 1] [0, 1] 1
  where starts :: K [5, 2] Int64 = unsafeIota 0

test8 :: R [5, 2, 2] Float -> R [5, 2] Float
test8 = vmap (unsafeDiagonal 0 1 :: R '[2, 2] Float -> R '[2] Float)

test9 :: Tensorial t => R [10, 5, 2] t -> R '[10, 2] t
test9 x = unsafeGather x starts [1] [0, 1] [0, 1] 1 [1, 1, 2]
  where starts :: R '[10, 2] Int64 = [[0, 1], [5, 3], [0, 3], [9, 2], [0, 0], [9, 4], [5, 1], [7, 3], [1, 1], [4, 3]]

test10 :: forall n m. (KnownNat n, KnownNat m) => R [n, m, 2] Float
test10 = unsafeConcat 2 n m
  where n :: R [n, m, 1] Float = unsafeIota 0
        m :: R [n, m, 1] Float = unsafeIota 1

startIdx0 :: R [3, 2] Int64
startIdx0 = [[0, 0], [3, 2], [1, 4]]

startIdx1 :: R [3, 2] Int64
startIdx1 = [[3, 4], [2, 0], [2, 1]]

test11input :: R [2, 4, 5, 2] Float
test11input = broadcast' (test10 :: R [4, 5, 2] Float)

test11 :: R [2, 4, 5, 2] Float -> R [2, 3, 2] Float
test11 input = unsafeConcat 0 a' b'
  where a  :: R [1, 4, 5, 2] Float = unsafeSlice input [(0, 1, 1), (0, 4, 1), (0, 5, 1), (0, 2, 1)]
        b  :: R [1, 4, 5, 2] Float = unsafeSlice input [(1, 2, 1), (0, 4, 1), (0, 5, 1), (0, 2, 1)]
        a' :: R [1, 3, 2]    Float = unsafeGather a startIdx0 [0, 2] [1, 2] [1, 2] 1 [1, 1, 1, 2]
        b' :: R [1, 3, 2]    Float = unsafeGather b startIdx1 [0, 2] [1, 2] [1, 2] 1 [1, 1, 1, 2]

test12 :: R [2, 4, 5, 2] Float -> R [2, 3, 2] Float
test12 = vmap (\input -> 
  unsafeGather input startIdx0 [1] [0, 1] [0, 1] 1 [1, 1, 2])


test13 :: R [2, 4, 5, 2] Float -> R [2, 3, 2] Float
test13 = vmap (\start input -> 
  unsafeGather input start [1] [0, 1] [0, 1] 1 [1, 1, 2]) s
  where s :: R [2, 3, 2] Int64 = unsafeConcat 0 (reshape startIdx0 :: R [1, 3, 2] Int64) (reshape startIdx1 :: R [1, 3, 2] Int64)

test14 :: R [4, 5, 2] Float -> R [2, 3, 2] Float
test14 input = vmap (\start -> 
  unsafeGather input start [1] [0, 1] [0, 1] 1 [1, 1, 2]) s
  where s :: R [2, 3, 2] Int64 = unsafeConcat 0 (reshape startIdx0 :: R [1, 3, 2] Int64) (reshape startIdx1 :: R [1, 3, 2] Int64)



-- TODO: Make more test
-- TODO: Make these test proper


main :: IO ()
main = do
  -- test 1
  let t1a = jit $ rgrad test1
      t1b = ngrad $ jit test1
    
  print $ t1a [[2, 1, 4, 3], [4, 2, 1, 6]] [[2, -8, 7, -5], [0, -4, 8, 0]] [[5, 5, 2, 7, 1], [3, 5, 6, 2, 1]] -
          t1b [[2, 1, 4, 3], [4, 2, 1, 6]] [[2, -8, 7, -5], [0, -4, 8, 0]] [[5, 5, 2, 7, 1], [3, 5, 6, 2, 1]]
  -- test 2
  let t2a = jit $ rgrad test2
      t2b = ngrad $ jit test2
  print $ t2a (splat True) [[2, -1, -4, 5], [5, -2, -5, 2]] [[0, 0, 0, 0], [5, 2, -1, -5]]
  print $ t2b (splat True) [[2, -1, -4, 5], [5, -2, -5, 2]] [[0, 0, 0, 0], [5, 2, -1, -5]]

  -- test 3
  let t3a = jit $ rgrad test3 
  print $ t3a [[2, 4], [-1, 6]] [[True, False], [False, True]]

  -- test 4
  let t4a = jit $ rgrad test4
  print $ t4a [[2, 1], [5, 2], [-4, 1], [-5, -6]] [False, True, False, True]

  -- test 5
  putStrLn $ replicate 16 '='
  print $ (jit test5 :: Tensor [5, 5] Float -> Tensor '[5] Float) [[0, 1, 2, 3, 4], [1, 0, 1, 2, 3], [2, 1, 0, 1, 2], [3, 2, 1, 0, 1], [4, 3, 2, 1, 0]]
  print $ (jit . rgrad $ reduceAdd' . test5) [[0, 1, 2, 3, 4], [1, 0, 1, 2, 3], [2, 1, 0, 1, 2], [3, 2, 1, 0, 1], [4, 3, 2, 1, 0]]

  -- test6
  print $ jit test6 [[[0, 1], [3, 1], [4, 5]], 
                     [[-1,3], [4, 1], [0, 0]], 
                     [[3, 1], [-6,4], [3, 0]], 
                     [[0,-3], [4,-1], [3, 3]],
                     [[0, 0], [9, 9], [4, 4]]]
  print $ (jit . rgrad $ reduceAdd' . test6) [[[0, 1], [3, 1], [4, 5]], 
                                              [[-1,3], [4, 1], [0, 0]], 
                                              [[3, 1], [-6,4], [3, 0]], 
                                              [[0,-3], [4,-1], [3, 3]],
                                              [[0, 0], [9, 9], [4, 4]]]

  print $ (ngrad . jit $ reduceAdd' . test6) [[[0, 1], [3, 1], [4, 5]], 
                                              [[-1,3], [4, 1], [0, 0]], 
                                              [[3, 1], [-6,4], [3, 0]], 
                                              [[0,-3], [4,-1], [3, 3]],
                                              [[0, 0], [9, 9], [4, 4]]]
  -- test 7
  putStrLn $ replicate 16 '='
  print $ jit test7 [0, 1, 2, 3, 4]
  print $ (jit . rgrad $ reduceAdd' . test7) [0, 1, 2, 3, 4]
  print $ (ngrad . jit $ reduceAdd' . test7) [0, 1, 2, 3, 4]
  
  print $ jit (unsafeDiagonal 0 1 :: Tracer [3, 3] Float -> Tracer '[3] Float) ([[3, 4, 1], [6, 3, 1], [2, 9, 3]] :: Tensor [3, 3] Float)
  
  print $ jit test8 [[[0, 1], 
                      [4, 0]], 
                     [[6, 4],
                      [8, 6]],
                     [[-3,1],
                      [0,-3]],
                     [[5, 5],
                      [0, 0]],
                     [[9, 9],
                      [0, 0]]]
  
  
  putStrLn $ replicate 16 '='
  print (jit test11 $ jit test11input)
  print (jit test12 $ jit test11input)
  print (jit test13 $ jit test11input)
  print (jit test14 $ jit test10)
  -- 
  putStrLn $ replicate 16 '='
  -- 
  -- traceDebugGrad test11
  let a = (jit . rgrad) test11 $ jit test11input
      b = (ngrad . jit) (reduceAdd' . test11) $ jit test11input
  -- traceDebug ((`unsafeReduceAdd` [0, 1, 2, 3]) :: Tracer '[2, 4, 5, 2] Float -> Tracer '[] Float)
  print $ l2Loss (a - b)

  echoNumCompilations 
