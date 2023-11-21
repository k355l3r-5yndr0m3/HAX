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

type Test1 r t = Dense r t 64 32 >> ReLU >> Softmax
test1 :: (r ~ R, t ~ Float) => Test1 R t -> r '[64] t -> r '[32] t
test1 = feed

test1rgf = jit $ grad test1
test1ngf = ngrad $ jit test1

main :: IO ()
main = do
  let x = rand (mkStdGen 6234)
      y = rand (mkStdGen 7837)
      a = test1rgf x y
      b = test1ngf x y
  print $ compareGrad a b
  print $ softmax y
  echoNumCompilations 
