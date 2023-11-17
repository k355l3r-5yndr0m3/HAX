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
import Foreign

import GHC.IO.Unsafe
import GHC.TypeError

import Control.Exception (assert)

delta :: Fractional t => t
delta = 0.004918684734

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
  default ngradIn :: (T s u, Fractional (StorageType u), t ~ Tensor s u) => t -> (t -> a, [[t -> a]]) -> ((a, [[a]]), [[Float]] -> (t, [[Float]]))
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

instance NGradIn t => NGradIn [t] where
  ngradIn []     (f, fs) = ((f [], [g <*> [[]] | g <- fs]), ([], ))
  ngradIn (t:ts) (f, fs) = (s', \g -> 
    let (ts', g') = c' g
        (t', g'') = c g'
    in  (t':ts', g''))
    where modf b a as = b (a:as)
          f'  = modf f
          fs' = [[modf j | j <- i] | i <- fs]
          (s , c ) = ngradIn t  (f', fs')
          (s', c') = ngradIn ts s

instance (NGradIn a, NGradIn b) => NGradIn (a, b) where
  ngradIn (a, b) (curry -> f, fmap (fmap curry) -> fs) = (s', \g -> 
    let (b', g')  = c' g
        (a', g'') = c g'
    in  ((a', b'), g''))
    where (s , c ) = ngradIn a (f, fs)
          (s', c') = ngradIn b s

instance (NGradIn a, NGradIn b) => NGradIn (a <&> b) where
  ngradIn (a :&: b) (f, fs) = (s', \g -> 
    let (b', g')  = c' g
        (a', g'') = c g'
    in  (a' :&: b', g''))
    where modf g x y = g (x :&: y)
          f'  = modf f
          fs' = [[modf j | j <- i] | i <- fs]
          (s , c ) = ngradIn a (f', fs')
          (s', c') = ngradIn b s 

instance KnownShape s => NGradIn (Tensor s Float)

instance KnownShape s => NGradIn (Tensor s Int64) where
  ngradIn t (f, fs) = ((f t, fmap (<*> [t]) fs), (splat 0, ))
instance KnownShape s => NGradIn (Tensor s Word8) where
  ngradIn t (f, fs) = ((f t, fmap (<*> [t]) fs), (splat 0, ))
instance KnownShape s => NGradIn (Tensor s Bool) where
  ngradIn t (f, fs) = ((f t, fmap (<*> [t]) fs), (splat False, ))
