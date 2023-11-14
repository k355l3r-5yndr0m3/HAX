{-# LANGUAGE DataKinds #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE ViewPatterns #-}
module Main (main) where

import Codec.Picture

import HAX.Tensor
import HAX.PjRt 
import HAX.AD
import HAX.Target
import HAX.Jit
import HAX.Utils
import HAX.IO

import Data.Proxy
import Data.Int
import Data.Word



test1 :: Target Tracer '[5] Float -> Target Tracer '[5] Float -> Target Tracer '[5] Float 
test1 = (+)
test2 :: Target Tracer '[5] Float -> Target Tracer '[5] Float -> Target Tracer '[5] Float 
test2 = (-)
test3 :: Target Tracer '[5] Float -> Target Tracer '[5] Float -> Target Tracer '[5] Float 
test3 = (*)

test4 :: Target Tracer '[5] Float -> Target Tracer '[5] Float -> Target Tracer '[5] Float 
test4 = vmap (+)

test5 :: Target Tracer '[5, 3] Float -> Target Tracer '[5, 3] Float -> Target Tracer '[5, 3] Float
test5 = vmap (\ a b -> 
  let c = a + b
      d = c + a
      e = d + c
      f = d + d
      g = f + e
  in  g)

test6 :: Target Tracer '[2, 3, 4] Float -> Target Tracer '[2, 3, 4] Float -> Target Tracer '[2, 3, 4] Float
test6 = vmap (\ a b -> a - vmap (+) a b)

test7 :: Target Tracer '[2, 3] Float -> Target Tracer '[3] Float -> Target Tracer '[2, 3] Float
test7 x y = vmap (+ y) x

test8 :: Target Tracer '[4, 5] Float -> Target Tracer '[7, 4, 3, 5] Float 
test8 = (`broadcast` (Proxy :: Proxy '[1, 3]))

test9 :: Target Tracer '[5] Float -> Target Tracer '[4, 2, 5] Float
test9 = broadcast'

test10 :: Target Tracer '[5, 2] Float -> Target Tracer '[5, 4, 2, 7] Float
test10 = vmap (`broadcast` (Proxy :: Proxy '[1]))

test11 :: Target Tracer '[2, 5, 3] Float -> Target Tracer '[3, 7] Float -> Target Tracer '[2, 5, 7] Float 
test11 x y = vmap (`matmul` y) x

test12 :: Target Tracer [5, 2] Float -> Target Tracer [5, 2] Float -> Target Tracer '[2] Float -> Target Tracer [5, 2] Float
test12 x y z = 
  vmap (\ a b -> 
    a + vmap (+) b z) x y

test13 :: Target Tracer [5, 2] Float -> Target Tracer '[2] Float -> Target Tracer [5, 2] Float
test13 x y = vmap (const y) x

test14 :: Target Tracer [5, 2] Float -> Target Tracer [2, 5] Float
test14 operand = broadcast operand (Proxy :: Proxy '[1, 0])

-- differentiation test
type RTracer = Reverse Tracer
test15 :: RTracer '[4, 3, 6] Float -> RTracer '[4, 3, 6] Float 
test15 = negate

test16 :: RTracer '[4] Float -> RTracer '[3, 4, 5] Float 
test16 = (`broadcast` (Proxy :: Proxy '[1]))

test17 :: RTracer '[4] Float -> RTracer '[3] Float -> RTracer '[4, 3] Float 
test17 = prod

-- vmap differentiablity
test18 :: Target (Reverse Tracer) '[5, 3] Float -> Target (Reverse Tracer) '[5, 3] Float -> Target (Reverse Tracer) '[5, 3] Float
test18 = vmap (+)

test19 :: Target (Reverse Tracer) '[5, 3] Float -> Target (Reverse Tracer) '[5, 3] Float -> Target (Reverse Tracer) '[5, 3] Float
test19 = (+)

test20 :: Target (Reverse Tracer) [10, 5, 6] Float
               -> Target (Reverse Tracer) [10, 5, 2] Float
               -> Target (Reverse Tracer) [6, 2] Float
               -> Target (Reverse Tracer) [10, 5, 2] Float 
test20 x y z = 
  vmap (\ (a :: Target (Reverse Tracer) '[5, 6] Float) (b :: Target (Reverse Tracer) '[5, 2] Float) -> 
    matmul a z + b) 
      x (y :: Target (Reverse Tracer) '[10, 5, 2] Float)



-- a <&> b as input
test21 :: (Reverse Tracer [4, 5] Float <&> Reverse Tracer [5, 7] Float) -> Reverse Tracer [4, 7] Float
test21 (a :&: b) = matmul a b

test22 :: (Reverse Tracer [4, 5] Float <&> Reverse Tracer [5, 7] Float <&> Reverse Tracer [4, 7] Float) -> Reverse Tracer [4, 7] Float
test22 (a :&: b :&: c) = (a `matmul` b) + c

test23 :: Tracer '[5] Float -> Tracer '[5] Float -> Tracer '[5] Float -> Tracer '[5] Float
test23 x y z = x + z - y

-- branching and predicate
test24 :: Target (Reverse Tracer) '[2, 4] Float -> Target (Reverse Tracer) '[2, 4] Float -> Target (Reverse Tracer) '[] Bool -> Target (Reverse Tracer) '[2, 4] Float
test24 x y = branch (x - y) (x * y)

test25 :: Tracer '[5, 5] Float -> Tracer '[5, 5] Float -> Tracer '[5, 5] Bool
test25 = isGT

test26 :: Tracer '[5, 5] Float -> Tracer '[5, 5] Float
test26 x = select (x + 2) x (x `isGT` 0)


-- Recursion test
-- TODO: Implement
test27 :: Target Tracer [5, 5] Float -> Target Tracer '[5] Float -> Target Tracer [5, 5] Float
test27 x y = unsafeScatter x indices y [] [0, 1] [0, 1] 1
  where indices :: Target Tracer [5, 2] Int64 = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]]


test28 :: Target Tracer [3, 5, 5] Float -> Target Tracer [3, 5] Float -> Target Tracer [3, 5, 5] Float
test28 = vmap test27

test29 :: Target Tracer [3, 5, 5] Float -> Target Tracer '[5] Float -> Target Tracer [3, 5, 5] Float
test29 x y = vmap (`test27` y) x

-- TODO: Implement more type inference
test30 :: Target Tracer [256, 256, 3] Float -> Target Tracer [248, 248, 3] Float
test30 image = convolution' image kernel
  where kernel  :: Target Tracer [3, 9, 9, 3] Float = 
          broadcast kernel' (Proxy :: Proxy [1, 2]) / 3
        kernel' :: Target Tracer [9, 9] Float = 
          [[0.0000, 0.0000, 0.0000, 0.0001, 0.0001, 0.0001, 0.0000, 0.0000, 0.0000], 
           [0.0000, 0.0000, 0.0004, 0.0014, 0.0023, 0.0014, 0.0004, 0.0000, 0.0000],
           [0.0000, 0.0004, 0.0037, 0.0146, 0.0232, 0.0146, 0.0037, 0.0004, 0.0000],
           [0.0001, 0.0014, 0.0146, 0.0584, 0.0926, 0.0584, 0.0146, 0.0014, 0.0001],
           [0.0001, 0.0023, 0.0232, 0.0926, 0.1466, 0.0926, 0.0232, 0.0023, 0.0001],
           [0.0001, 0.0014, 0.0146, 0.0584, 0.0926, 0.0584, 0.0146, 0.0014, 0.0001],
           [0.0000, 0.0004, 0.0037, 0.0146, 0.0232, 0.0146, 0.0037, 0.0004, 0.0000],
           [0.0000, 0.0000, 0.0004, 0.0014, 0.0023, 0.0014, 0.0004, 0.0000, 0.0000],
           [0.0000, 0.0000, 0.0000, 0.0001, 0.0001, 0.0001, 0.0000, 0.0000, 0.0000]]

test31 :: Target Tracer [256, 256, 3] Float -> Target Tracer [2, 3, 9, 9, 3] Float -> Target Tracer [2, 248, 248, 3] Float
test31 image = vmap (convolution' image)

gaussianKernel :: Tensor [9, 9] Float 
gaussianKernel = [[0.0000, 0.0000, 0.0000, 0.0001, 0.0001, 0.0001, 0.0000, 0.0000, 0.0000], 
                  [0.0000, 0.0000, 0.0004, 0.0014, 0.0023, 0.0014, 0.0004, 0.0000, 0.0000],
                  [0.0000, 0.0004, 0.0037, 0.0146, 0.0232, 0.0146, 0.0037, 0.0004, 0.0000],
                  [0.0001, 0.0014, 0.0146, 0.0584, 0.0926, 0.0584, 0.0146, 0.0014, 0.0001],
                  [0.0001, 0.0023, 0.0232, 0.0926, 0.1466, 0.0926, 0.0232, 0.0023, 0.0001],
                  [0.0001, 0.0014, 0.0146, 0.0584, 0.0926, 0.0584, 0.0146, 0.0014, 0.0001],
                  [0.0000, 0.0004, 0.0037, 0.0146, 0.0232, 0.0146, 0.0037, 0.0004, 0.0000],
                  [0.0000, 0.0000, 0.0004, 0.0014, 0.0023, 0.0014, 0.0004, 0.0000, 0.0000],
                  [0.0000, 0.0000, 0.0000, 0.0001, 0.0001, 0.0001, 0.0000, 0.0000, 0.0000]]


boxBlurKernel :: Tensor [9, 9] Float
boxBlurKernel = 1 / 81

test32input :: Tensor [2, 3, 9, 9, 3] Float
test32input = unsafeConcat 0 g b
  where g :: Tensor [1, 3, 9, 9, 3] Float = broadcast gaussianKernel (Proxy :: Proxy [2, 3])
        b :: Tensor [1, 3, 9, 9, 3] Float = broadcast boxBlurKernel  (Proxy :: Proxy [2, 3])

main :: IO ()
main = do 
  -- --traceDebug test1
  -- --traceDebug test2
  -- --traceDebug test3
  -- --traceDebug test5
  -- --traceDebug test6
  -- --traceDebug test5
  -- --traceDebug test6
  -- --traceDebug test7
  -- --traceDebug test8
  -- --traceDebug test9
  -- --traceDebug test10
  -- --traceDebug test11

  --traceDebug test12
  --traceDebug test13

  --traceDebug test14


  --traceDebugGrad test15
  --traceDebugGrad test16
  --traceDebugGrad test17

  --traceDebugGrad test18
  --traceDebugGrad test19

  --traceDebug     test4
  --traceDebugGrad test20

  let a :: Tensor '[4, 5] Float = 
        [[ 1, 2, 3, 4, 5], 
         [-1,-2,-3,-4,-5],
         [ 6,-2, 5, 9,-1],
         [-7, 3, 2, 8, 6]]
      b :: Tensor '[5, 1] Float = 
        [[1],[2],[3],[4],[5]]
      c = matmul a b

  -- print =<< debugTensorShape c
  -- print c
  -- let d :: Tensor '[4, 1] Float = [[1], [2], [3], [4]]
  -- print d
  print c


  let e = jit test10
  print $ e 1


  print ([fromRational i + fromRational j | i <- [0.0..6.0], j <- [-5.0..9.0]] :: [Tensor '[12] Float])

  --traceDebug test21
  --traceDebugGrad test21
  --traceDebug test22
  --traceDebugGrad test22

  --traceDebug test23

  --traceDebug test24
  --traceDebugGrad test24

  print ([[True, False], [False, True]] :: Tensor '[2, 2] Bool)
  let t24 = jit test24
  print $ t24 [[0, 4, 2, 5], [1, -3, -4, -5]] [[0, -1, -5, -3], [4, 2, 5, 6]] (splat False)

  --traceDebug test25
  --traceDebug test26



  print (linspace (Proxy :: Proxy 0) (0, 1) :: Tensor '[5] Float)
  
  print (jit test27 [[0, 4, 3, 1, 5],
                     [4, 1, 3, 7, 1],
                     [9, 3, 5, 2, 3],
                     [3, 3, 7, 3, 5],
                     [3, 5, 7, 2, 5]] 
                     [9, 9, 9, 9, 9])

  print (jit test28 0 (1 + unsafeIota 0))
  print (jit test29 0 (1 + unsafeIota 0))
  
  either putStrLn (\(convert . tensorFromImage . convertRGB8 -> image :: Tensor [256, 256, 3] Float) -> do
    let image' = jit (test30 . (/256.0)) image
        t2     = jit test31 (image / 256) test32input / 3
    writePng "test/data/image-conv.png" (convertRGB8 $ ImageRGBF $ imageFromTensor image')
    writePng "test/data/image-conv0.png" (convertRGB8 $ ImageRGBF $ imageFromTensor (t2 @% 0))
    writePng "test/data/image-conv1.png" (convertRGB8 $ ImageRGBF $ imageFromTensor (t2 @% 1))) =<< readImage "test/data/image.jpg"





  clientDestroy client
  echoNumCompilations
  return ()
