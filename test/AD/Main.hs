{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedLists #-}
module Main where
import HAX.Target
import HAX.Tensor
import HAX.Utils

import HAX.AD.Reverse
import HAX.AD.Numerical
import HAX.Jit

import Control.Monad

import System.Random
import System.Random.Stateful

import Data.Word

type R = Target (Reverse Tracer)
test1 :: R '[2, 4] Float -> R '[2, 4] Float -> R '[2, 4] Float -> R '[] Float
test1 x y z = sigma' $ y * x - z

test2 :: R '[2, 4] Float -> R '[2, 4] Float -> R '[] Pred -> R '[] Float 
test2 x y = sigma' . branch (x - y) (x * y)

test3 :: R '[2, 2] Float -> R '[2, 2] Pred -> R '[] Float
test3 x = sigma' . select 0 x

test4 :: R '[4, 2] Float -> R '[4] Pred -> R '[] Float 
test4 x = sigma' . vmap (branch 0) x

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


