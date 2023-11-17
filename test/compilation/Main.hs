{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedLists #-}
module Main (main) where
import HAX.Tensor
import HAX.Target

import HAX.AD.Reverse
import HAX.AD.Numerical

import GHC.TypeLits
import Data.Int
import HAX.Jit (echoNumCompilations)

type R = Target (Reverse Tracer)

test10 :: forall n m. (KnownNat n, KnownNat m) => R [n, m, 2] Float
test10 = unsafeConcat 2 n m
  where n :: R [n, m, 1] Float = unsafeIota 0
        m :: R [n, m, 1] Float = unsafeIota 1

test11input :: R [2, 4, 5, 2] Float
test11input = broadcast' (test10 :: R [4, 5, 2] Float)

startIdx0 :: R [3, 2] Int64
startIdx0 = [[0, 0], [3, 2], [1, 4]]

startIdx1 :: R [3, 2] Int64
startIdx1 = [[3, 4], [2, 0], [2, 1]]

test11 :: R [2, 4, 5, 2] Float -> R '[] Float
test11 input = reduceAdd' (unsafeConcat 0 a' b' :: R [2, 3, 2] Float)
  where a  :: R [1, 4, 5, 2] Float = unsafeSlice input [(0, 1, 1), (0, 4, 1), (0, 5, 1), (0, 2, 1)]
        b  :: R [1, 4, 5, 2] Float = unsafeSlice input [(1, 2, 1), (0, 4, 1), (0, 5, 1), (0, 2, 1)]
        a' :: R [1, 3, 2]    Float = unsafeGather a startIdx0 [0, 2] [1, 2] [1, 2] 1 [1, 1, 1, 2]
        b' :: R [1, 3, 2]    Float = unsafeGather b startIdx1 [0, 2] [1, 2] [1, 2] 1 [1, 1, 1, 2]

main :: IO ()
main = do
  let a = (jit . rgrad) test11 $ jit test11input
--      b = (ngrad . jit) test11 $ jit test11input
  -- traceDebug ((`unsafeReduceAdd` [0, 1, 2, 3]) :: Tracer '[2, 4, 5, 2] Float -> Tracer '[] Float)
  -- print (unsafeReduceAdd ([[0, 1], [3, 4]] :: Tensor '[2, 2] Float) [0, 1] :: Tensor '[] Float)
--  print $ l2Loss (b - a)
  echoNumCompilations
