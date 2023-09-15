{-# LANGUAGE DataKinds #-}
module Main where
import HAX.Tensor
import HAX.Target

import HAX.AD
import HAX.Jit
import HAX.Utils
import HAX.Random

import System.Random

-- Example linear regression
model :: Target (Reverse Tracer) '[128] Float -> Target (Reverse Tracer) '[] Float -> Target (Reverse Tracer) '[] Float -> Target (Reverse Tracer) '[128] Float
model x a b = vmap (\x' -> x' * a + b) x

loss :: Target (Reverse Tracer) '[128] Float -> Target (Reverse Tracer) '[] Float -> Target (Reverse Tracer) '[] Float -> Target (Reverse Tracer) '[128] Float -> Target (Reverse Tracer) '[] Float
loss x a b y = l2Loss (y - model x a b)

loss2 :: Tensor '[128] Float
              -> Tensor '[] Float
              -> Tensor '[] Float
              -> Tensor '[128] Float
              -> Tensor '[128] Float
                 <+> Tensor '[] Float
                      <+> Tensor '[] Float 
                           <+> Tensor '[128] Float
loss2 = jit $ rgrad loss

lr :: Fractional f => f
lr = 0.0001

training :: [(Tensor '[128] Float, Tensor '[128] Float)]
                 -> (Tensor '[] Float, Tensor '[] Float)
                 -> IO (Tensor '[] Float, Tensor '[] Float)
training []          (a, b) = return (a, b)
training ((x, y):ds) (a, b) = do 
  print (a, b)
  training ds (a - a' * lr, b - b' * lr)
  where _ :+: a' :+: b' :+: _ = loss2 x a b y 

main :: IO ()
main = do 
  let x :: Tensor '[128] Float = linspace (-5, 5) 
      y = -4 * x + 5 + fst (tensorUniformR (-0.2, 0.2) (mkStdGen 52))
      a :: Tensor '[] Float = 5
      b :: Tensor '[] Float = -4
  (a', b') <- training (replicate 256 (x, y)) (a, b)
  print (a', b')
