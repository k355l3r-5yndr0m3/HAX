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

type R = Target (Reverse Tracer)
test1 :: R '[2, 4] Float -> R '[2, 4] Float -> R '[2, 4] Float -> R '[] Float
test1 x y z = sigma' $ y * x - z


main :: IO ()
main = do
  let t1a = jit $ rgrad test1
      t1b = ngrad $ jit test1
  print $ t1a [[2, 1, 4, 3], [4, 2, 1, 6]] [[2, -8, 7, -5], [0, -4, 8, 0]] [[5, 5, 2, 7, 1], [3, 5, 6, 2, 1]] - 
          t1b [[2, 1, 4, 3], [4, 2, 1, 6]] [[2, -8, 7, -5], [0, -4, 8, 0]] [[5, 5, 2, 7, 1], [3, 5, 6, 2, 1]]
  -- let r = test1' <$> neighborhood [[4, 3, 5, 5, 2], [2, 4, 5, 1, 3]] <*> neighborhood [[44, 2, 4, 6, 2], [3, 1, 8, 3, 2]]
  -- forM_ r print
  -- let (center, deviation) = fst $ runStateGen (mkStdGen 21) (\ g -> do 
  --       _center    :: (Tensor [5, 5] Float <+> Tensor [5, 5] Float <+> Tensor [5, 2] Float) <- (\ a b c -> a :+: b :+: c) <$> tensorUniformRM (-1, 1) g <*> tensorUniformRM (-1, 1) g <*> tensorUniformRM (-1, 1) g
  --       _deviation :: [Tensor [5, 5] Float <+> Tensor [5, 5] Float <+> Tensor [5, 2] Float] <- replicateM 32 $ (\ a b c -> a :+: b :+: c) <$> tensorUniformRM (-1, 1) g <*> tensorUniformRM (-1, 1) g <*> tensorUniformRM (-1, 1) g
  --       return (_center, _deviation))
  --     result3 = rangradtest center (jrgrad test3) deviation 0.0001
  -- print result3



