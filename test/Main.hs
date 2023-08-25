{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
module Main (main) where
import Foreign

import HAX.Tensor 
import HAX.PjRt
import HAX.TList
import HAX.Jit
import HAX.AD.Reverse


test :: (Trace t, Num (t '[12] Float), Fractional (t '[12] Float)) => t '[12] Float -> t '[12] Float -> t '[12] Float
test a b = (s / s) * (s + s)
  where s = a + b

test1 :: (Trace t, Num (t '[12] Float)) => t '[12] Float -> TList '[t '[12] Float]
test1 x = (x + x) :+ (:@)


main :: IO ()
main = do 
  device <- head <$> clientAddressableDevices client
  a :: Tensor '[12] Float <- withArray [0..11]  $ tensorFromHostBuffer device
  b :: Tensor '[12] Float <- withArray [1..12]  $ tensorFromHostBuffer device
  
  let testjit :: (Trace t, Jit t (Tracer '[12] Float -> Tracer '[12] Float -> Tracer '[12] Float) f) => f
      testjit = jit (test :: Tracer '[12] Float -> Tracer '[12] Float -> Tracer '[12] Float)
      testgrad = grad (test :: (t ~ Tracer, Num (t '[12] Float)) => RDual t '[12] Float -> RDual t '[12] Float -> RDual t '[12] Float)
      gradient :: Tensor '[12] Float -> Tensor '[12] Float -> TList '[Tensor '[12] Float, Tensor '[12] Float]
      gradient = jit testgrad
      test1jit :: Tensor '[12] Float -> TList '[Tensor '[12] Float] = jit (test1 :: Tracer '[12] Float -> TList '[Tracer '[12] Float])

  print $ testjit a b
  print $ test a b
  print $ recip b
  print $ gradient a b
  traceDebug (testjit :: Tracer '[12] Float -> Tracer '[12] Float -> Tracer '[12] Float)

  putStrLn $ take 12 $ repeat '='
  print $ test1jit a


  clientDestroy client
  return ()
