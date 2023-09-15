{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedLists #-}
module Main where
import HAX.Tensor
import HAX.Target
import HAX.Jit
import HAX.AD

type R = Target (Reverse Tracer) 

test1 :: R '[5, 5] Float -> R '[] Float 
test1 = l2Loss

test1'  = jit test1
test1g' = jit $ rgrad test1

gd :: Word -> Tensor '[5, 5] Float -> IO (Tensor '[5, 5] Float)
gd 0 x = return x
gd i x = do 
  putStrLn $ "Loss: " ++ show l
  putStrLn $ "   x: " ++ show x
  putStrLn $ "  x': " ++ show x'
  putStrLn $ "   d: " ++ show d
  putStrLn ""
  gd (i - 1) (x - d)
  where x' = test1g' x
        l  = test1'  x
        d  = x' * 0.01

main :: IO ()
main = do 
  print =<< gd 64 [[4, 6, 2, 8, 2], [-5, -6, -1, -5, 0], [5, -1, 6, -8, 2], [5, -7, -7, 1, 5], [-5, 2, 7, -6, -2]]
