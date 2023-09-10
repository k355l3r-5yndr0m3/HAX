{-# LANGUAGE DataKinds #-}
module Main where
import HAX.Target
import HAX.Tensor
import HAX.Jit
import HAX.Utils

import HAX.AD.Reverse
import HAX.AD
import HAX.Random

import Control.Monad

import System.Random
import System.Random.Stateful

type Tar = Target (Reverse Tracer)

grad f = (jit f, jit $ rgrad f)

test1 :: Tar [5, 5] Float -> Tar [5, 5] Float -> Tar '[] Float 
test1 x1 x2 = sigma' $ x1 + x2 / x1

(forw1, grad1) = grad test1





main :: IO ()
main = do
  print . fst =<< runStateGenT (mkStdGen 3134) (\ g -> do 
    center    :: (Tensor [5, 5] Float, Tensor [5, 5] Float) <- (,) <$> tensorUniformRM (-1, 1) g <*> tensorUniformRM (-1, 1) g
    deviation ::[(Tensor [5, 5] Float, Tensor [5, 5] Float)]<- replicateM 16 $ (,) <$> tensorUniformRM (-1, 1) g <*> tensorUniformRM (-1, 1) g
    let forward  = uncurry forw1 center
        gradient = uncurry grad1 center
        graderr  = 
          let (x0, x1) = center
          in  [ let dy' = forw1 (x0 + dx0) (x1 + dx1) - forward 
                    dy = 
                      let (dy_dx0 :+: dy_dx1) = gradient * (dx0 :+: dx1)
                      in  sigma' dy_dx0 + sigma' dy_dx1
                in  dy' - dy | (dx0, dx1) <- deviation ]
    return graderr)

