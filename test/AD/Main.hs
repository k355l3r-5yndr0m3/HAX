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


type R = Target (Reverse Tracer)
test1 :: R '[2, 4] Float -> R '[2, 4] Float -> R '[2, 4] Float -> R '[] Float
test1 x y z = reduceAdd' $ y * x - z

test2 :: R '[2, 4] Float -> R '[2, 4] Float -> R '[] Pred -> R '[] Float 
test2 x y = reduceAdd' . branch (x - y) (x * y)

test3 :: R '[2, 2] Float -> R '[2, 2] Pred -> R '[] Float
test3 x = reduceAdd' . select 0 x

test4 :: R '[4, 2] Float -> R '[4] Pred -> R '[] Float 
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


test8 :: R '[5, 2, 2] Float -> R [5, 2] Float
test8 = vmap (unsafeDiagonal 0 1 :: R '[2, 2] Float -> R '[2] Float)



main :: IO ()
main = do
  let t1a = jit $ rgrad test1
      t1b = ngrad $ jit test1
  print $ t1a [[2, 1, 4, 3], [4, 2, 1, 6]] [[2, -8, 7, -5], [0, -4, 8, 0]] [[5, 5, 2, 7, 1], [3, 5, 6, 2, 1]] - 
          t1b [[2, 1, 4, 3], [4, 2, 1, 6]] [[2, -8, 7, -5], [0, -4, 8, 0]] [[5, 5, 2, 7, 1], [3, 5, 6, 2, 1]]
  let t2a = jit $ rgrad test2
      t2b = ngrad $ jit test2
  print $ t2a [[2, -1, -4, 5], [5, -2, -5, 2]] [[0, 0, 0, 0], [5, 2, -1, -5]] 1
  print $ t2b [[2, -1, -4, 5], [5, -2, -5, 2]] [[0, 0, 0, 0], [5, 2, -1, -5]] 1
  let t3a = jit $ rgrad test3 
  print $ t3a [[2, 4], [-1, 6]] [[1, 0], [0, 1]]
  let t4a = jit $ rgrad test4
  print $ t4a [[2, 1], [5, 2], [-4, 1], [-5, -6]] [1, 0, 1, 0]

  putStrLn "========================"
  print $ (jit test5 :: Tensor [5, 5] Float -> Tensor '[5] Float) [[0, 1, 2, 3, 4], [1, 0, 1, 2, 3], [2, 1, 0, 1, 2], [3, 2, 1, 0, 1], [4, 3, 2, 1, 0]]
  print $ (jit . rgrad $ reduceAdd' . test5) [[0, 1, 2, 3, 4], [1, 0, 1, 2, 3], [2, 1, 0, 1, 2], [3, 2, 1, 0, 1], [4, 3, 2, 1, 0]]

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
