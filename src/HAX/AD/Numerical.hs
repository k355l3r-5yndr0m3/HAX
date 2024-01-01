{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilyDependencies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE DefaultSignatures #-}
{-# LANGUAGE LambdaCase #-}
module HAX.AD.Numerical where -- This module is intended only for debugging of differentiating algorithms

import HAX.Tensor.Tensorial
import HAX.Tensor.Tensor
import HAX.PjRt
import HAX.Utils

import Data.Primitive hiding (newArray)
import Data.Data
import Foreign

import GHC.IO.Unsafe
import GHC.TypeError
import GHC.Generics
import GHC.Real (infinity)

import Control.Exception (assert)
import Data.Bifunctor (Bifunctor(second, first))

delta :: Fractional t => t
delta = 1.40379e-5

class NGrad f where
  type NGradF' f g
  type NGradF  f

  ngrad' :: ([[Float]] -> (g, [[Float]])) -> (f, [[f]]) -> NGradF' f g
  ngrad  :: f -> NGradF f

instance (T s t, Real t) => NGrad (Tensor s t) where
  type NGradF' _ g = g
  type NGradF  _   = TypeError (Text "ngrad must be applied to a function")

  ngrad' recover (reduceAdd' -> y, fmap (fmap reduceAdd') -> ys) = assert (null l) g
    where gradients = [[(/delta) . realToFrac . getScalar $ (y' - y) | y' <- ys'] | ys' <- ys]
          (g, l) = recover gradients

  ngrad = undefined

instance (NGradIn t, NGrad f) => NGrad (t -> f) where
  type NGradF' (t -> f) g = t -> NGradF' f (g <&> t)
  type NGradF  (t -> f)   = t -> NGradF' f t

  ngrad' recover ffs t = ngrad' recover' ffs'
    where (ffs', rec) = ngradIn t ffs
          recover' g = let (t', c ) = rec g
                           (g', c') = recover c
                       in  (g' :&: t', c')
  ngrad f t = ngrad' recover (f', fs')
    where ((f', init -> fs'), recover) = ngradIn t (f, [[f]]) 

class NGradIn t where
  -- NOTE: The order of [(CIntPtr, [t -> a])] (or [(CIntPtr, [a])]) is in the opposite to the order of the arguments
  --       i.e the first element is corrilated to the last tensor, the second is the second to last tensor, ..., and the last element is corrilated to the first tensor 
  --       This behavour might make it unnessisary to tag each element with an id, since their order is known
  ngradIn :: t -> (t -> a, [[t -> a]]) -> ((a, [[a]]), [[Float]] -> (t, [[Float]]))
  default ngradIn :: (Generic t, GNGradIn (Rep t)) => t -> (t -> a, [[t -> a]]) -> ((a, [[a]]), [[Float]] -> (t, [[Float]]))
  ngradIn (from -> x) (f, fs) = second (first to .) $ gNgradIn x (f', fs')
    where f'  = f . to
          fs' = [[j . to | j <- is] | is <- fs]

  compareGrad' :: t -> t -> (Float, Int)
  default compareGrad' :: (Generic t, GNGradIn (Rep t)) => t -> t -> (Float, Int)
  compareGrad' (from -> a) (from -> b) = gCompareGrad' a b

  compareGrad :: t -> t -> Float
  compareGrad a b = total / fromIntegral nfree
    where (total, nfree) = compareGrad' a b

class GNGradIn f where
  gNgradIn :: f x -> (f x -> a, [[f x -> a]]) -> ((a, [[a]]), [[Float]] -> (f x, [[Float]]))
  gCompareGrad' :: f x -> f x -> (Float, Int)
instance GNGradIn V1 where
  gNgradIn v (f, fs) = ((f v, [[g v | g <- gs] | gs <- fs]), (v,))
  gCompareGrad' _ _ = (0, 0)
instance GNGradIn U1 where
  gNgradIn v (f, fs) = ((f v, [[g v | g <- gs] | gs <- fs]), (v,))
  gCompareGrad' _ _ = (0, 0)
instance (GNGradIn f, GNGradIn g) => GNGradIn (f :+: g) where
  gNgradIn (L1 x) (f, fs) = second (first L1 .) $ gNgradIn x (f', fs')
    where f'  = f . L1
          fs' = [[g . L1 | g <- gs] | gs <- fs]
  gNgradIn (R1 x) (f, fs) = second (first R1 .) $ gNgradIn x (f', fs')
    where f'  = f . R1
          fs' = [[g . R1 | g <- gs] | gs <- fs]
  gCompareGrad' (L1 a) (L1 b) = gCompareGrad' a b
  gCompareGrad' (R1 a) (R1 b) = gCompareGrad' a b
  gCompareGrad' _      _      = (fromRational infinity, 0)

instance (GNGradIn f, GNGradIn g) => GNGradIn (f :*: g) where
  gNgradIn (f :*: g) (j, js) = (s', \x -> 
    let (b', x')  = c' x
        (a', x'') = c x'
    in  (a' :*: b', x''))
    where (s , c ) = gNgradIn f (j', js')
          (s', c') = gNgradIn g s
          j' a b = j (a :*: b)
          js'    = [[\a b -> y (a :*: b) | y <- ys] | ys <- js] 
  gCompareGrad' (a :*: b) (c :*: d) = gCompareGrad' a c `add` gCompareGrad' b d
    where (x, y) `add` (z, w) = (x + z, y + w)

instance NGradIn c => GNGradIn (K1 i c) where
  gNgradIn (K1 c) (f, fs) = second (first K1 .) $ ngradIn c (f', fs')
    where f'  = f . K1
          fs' = [[g . K1 | g <- gs] | gs <- fs]
  gCompareGrad' (K1 a) (K1 b) = compareGrad' a b

instance GNGradIn f => GNGradIn (M1 i t f) where
  gNgradIn (M1 c) (f, fs) = second (first M1 .) $ gNgradIn c (f', fs')
    where f'  = f . M1
          fs' = [[g . M1 | g <- gs] | gs <- fs]
  gCompareGrad' (M1 a) (M1 b) = gCompareGrad' a b

instance NGradIn t => NGradIn [t] where
instance (NGradIn a, NGradIn b) => NGradIn (a, b) where
instance (NGradIn a, NGradIn b) => NGradIn (a <&> b) where

instance KnownShape s => NGradIn (Tensor s Float) where
  ngradIn t@(tensorToPrimArray -> x) (f, fs) = ((f t, (f <$> xs):[g <*> [t] | g <- fs]) , \case 
    []   -> error "Not enough output has been produced!"
    a:as -> (unsafePerformIO $ do 
      buffer <- mallocArray sz
      pokeArray buffer [realToFrac i | i <- a]
      tensorFromHostBufferGC defaultDevice buffer, as))
    where dx = delta
          xs = [unsafePerformIO $ do 
            buffer <- mallocArray sz
            copyPrimArrayToPtr buffer x 0 sz
            pokeElemOff buffer i $ dx + indexPrimArray x i
            tensorFromHostBufferGC defaultDevice buffer| i <- [0..sz - 1]]
          sz = sizeofPrimArray x
  compareGrad' a b = (getScalar $ reduceAdd' $ let x = a - b in x * x, fromIntegral $ product $ shapeVal (Proxy :: Proxy s))

instance KnownShape s => NGradIn (Tensor s Int64) where
  ngradIn t (f, fs) = ((f t, fmap (<*> [t]) fs), (splat 0, ))
  compareGrad' _ _ = (0, 0)
instance KnownShape s => NGradIn (Tensor s Word8) where
  ngradIn t (f, fs) = ((f t, fmap (<*> [t]) fs), (splat 0, ))
  compareGrad' _ _ = (0, 0)
instance KnownShape s => NGradIn (Tensor s Bool) where
  ngradIn t (f, fs) = ((f t, fmap (<*> [t]) fs), (splat False, ))
  compareGrad' _ _ = (0, 0)
