{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE DataKinds #-}
module Main where
import HAX.Target
import HAX.Tensor
import HAX.Jit

import HAX.AD.Reverse
import HAX.AD
import HAX.Random

import Control.Monad

import System.Random
import System.Random.Stateful

type Te = Target (Reverse Tracer)

test1 :: Te [5, 5] Float -> Te [5, 5] Float
test1 = negate

grad1 = jit $ rgrad test1

test2 :: Te [2, 8] Float -> Te [2, 8] Float -> Te [2, 8] Float
test2 = (+)

grad2 = jit $ rgrad test2

test3 :: Te [6, 2] Float -> Te [2, 4] Float -> Te [6, 4] Float 
test3 = matmul

grad3 = jit $ rgrad test3

main :: IO ()
main = do
  _ <- runStateGenT (mkStdGen 53) $ \ g -> do 
    t1_1 :: Tensor [5, 5] Float   <- tensorUniformRM (-1, 1) g
    d1_1 :: [Tensor [5, 5] Float] <- replicateM 8 (tensorUniformRM (-1, 1) g)

    undefined

  putStrLn "Hello, world!"

