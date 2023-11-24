{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedLists #-}
module Main where
import HAX.Target
import HAX.Tensor
import HAX.Utils

import HAX.AD.Reverse
import HAX.AD.Numerical
import HAX.Jit
import HAX.NN

import Data.Int
import GHC.TypeLits

import System.Random

type R = Target (Reverse Tracer)

type Test1 r t = Convolute r t 2 [5, 5] 4 
test1 :: (r ~ R, t ~ Float, n ~ 10) => Test1 R t -> r [n, 6, 6, 2] t -> r '[n, 2, 2, 4] t
test1 params = vmap (feed params)
-- 
test1rgf = jit $ grad test1
test1ngf = ngrad $ jit test1


main :: IO ()
main = do
  let x = rand (mkStdGen 6234)
      y = rand (mkStdGen 1245)
      a = test1rgf x y
      b = test1ngf x y
  print (compareGrad a b)
  echoNumCompilations 
