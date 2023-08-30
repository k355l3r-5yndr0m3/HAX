{-# LANGUAGE DataKinds #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
module Main (main) where
import Foreign

import HAX.Tensor 
import HAX.PjRt
import HAX.TList
import HAX.Jit
import HAX.AD.Reverse


testing :: Tracer '[12, 12] Float -> Tracer '[12, 12] Float -> Tracer '[12, 12] Float
testing x y = x * y + x

main :: IO ()
main = do 
  device <- head <$> clientAddressableDevices client
  a :: Tensor '[12, 12] Float <- withArray (replicate (12 * 12) 5)  $ tensorFromHostBuffer device
  b :: Tensor '[12, 12] Float <- withArray (replicate (12 * 12) 10)  $ tensorFromHostBuffer device
  
  let tested :: forall f t. (f ~ (Tracer '[12, 12] Float -> Tracer '[12, 12] Float -> Tracer '[12, 12] Float), Jit t f) => JitResult t f
      tested = jit testing
      test2 :: forall f t. (f ~ (Tracer '[12, 12] Float -> Tracer '[12, 12] Float -> Tracer '[12, 12] Float), Jit t f) => JitResult t f
      test2  = jit tested
  
  print $ tested a b
  print $ test2 a b

  clientDestroy client
  return ()
