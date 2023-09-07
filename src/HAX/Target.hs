{-# LANGUAGE DataKinds #-}
{-# LANGUAGE QuantifiedConstraints #-}
{-# LANGUAGE TypeFamilyDependencies #-}
{-# LANGUAGE UndecidableInstances #-}
module HAX.Target where
import HAX.Tensor.Tracer
import HAX.Tensor.Tensorial

import Control.Exception

import Data.Coerce
import Data.Proxy 
import Data.List
import Data.Functor
import Data.Bifunctor
import Data.IntMap.Strict

import MLIR
import Stablehlo.Dialect.Stablehlo.Attributes

import GHC.TypeLits

-- Target: represent the target of vmap transformation
--         Target dims r 
--         dims ++ shapeVal r is the true shape

-- Target dim tracer
-- NOTE: The true shape of Tracer is 
--   dim ++ s
-- So, what happen when a binary function encountered two different dims
--   f (x :: [5, 3]) (y :: [5, 3]) (z :: [3]) =         (A)
--     vmap (\ a b -> a + b + z) x y
-- Is equivalent to 
--   f (x :: [5, 3]) (y :: [5, 3]) (z :: [3]) =         (B)
--     vmap (\ a b c -> a + b + c) x y (broadcast z) 
--
-- It is obvious, all parameters of the function that is being vmapped (not correct for now)
--    has the same dims
-- To convert from (A) to (B) is possible by broadcasting so that the dims match up
-- Because of how variable scope work, all the dims that come into contact with eachother have a common suffix
--    and that suffix is the shortest dim, and all dims are suffix of the longest dims. That is to say
--    given a function (f) of arrity n 
--      f :: t1 -> t2 -> .. -> tn -> out
--    the longest dim is dim_max
--        shortest dim is dim_min
--    for all i in [1..n]
--        dim_i isSuffixOf dim_max
--        dim_min isSuffixOf dim_i
-- To ensure correctness, it is enough to broadcast all t_es so that they have the same dims as dim_max. The proof is obvious (I think)
--
-- The `capture` function allow this broadcasting to be done inside function, and the `binary` function allow this to be done automatically whenever two 
--  targets is fed into a binary function, they are automatically broadcasted
-- But this is not done at the site of vmap, which assumes all the inputs feeding it have the same dims and the output has the expected dim, this can easily be solved.

data Target r s t = Target [Integer] (r s t)
type Transformable r t = forall s s'. Coercible (r s t) (r s' t)

capture :: forall r s t. (T s t, TensorOp r t, Transformable r t) => Target r s t -> [Integer] -> Target r s t 
capture (Target di u) dmax = assert (di `isSuffixOf` dmax) $ 
  Target dmax $
    reifyShape src $ \ s0 -> reifyShape dst $ \ s1 -> 
      result s0 s1
  where src = di   ++ shapeVal (Proxy :: Proxy s)
        dst = dmax ++ shapeVal (Proxy :: Proxy s)
        result :: forall s0 s1. (KnownShape s0, KnownShape s1) => Proxy s0 -> Proxy s1 -> r s t
        result _ _ = 
          let t0 :: r s0 t = coerce u
              t1 :: r s1 t = unsafeBroadcast t0 mapping
          in  coerce $! t1
        mapping = fromIntegral <$> take (length src) [length dst - length src..]

binary :: (T s0 t0, T s1 t1, Transformable r t0, TensorOp r t0, TensorOp r t1, Transformable r t1) => Target r s0 t0 -> Target r s1 t1 -> ([Integer], r s0 t0, r s1 t1)
binary l@(Target ld lhs) r@(Target rd rhs) 
  | ld == rd           = (ld, lhs, rhs)
  | ld `isSuffixOf` rd = 
    let Target _ lhs' = capture l rd
    in  (rd, lhs', rhs)
  | rd `isSuffixOf` ld = 
    let Target _ rhs' = capture r ld 
    in  (ld, lhs, rhs')
  | otherwise = error $ "The impossible has happen: " ++ show ld ++ " vs " ++ show rd 

instance (T s t, forall s'. KnownShape s' => Num (r s' t), TensorOp r t, Transformable r t) => Num (Target r s t) where
  lhs + rhs = Target dim $ reifyShape (dim ++ shapeVal (Proxy :: Proxy s)) result
    where (dim, _lhs, _rhs) = binary lhs rhs
          result :: forall s'. KnownShape s' => Proxy s' -> r s t 
          result _ = coerce $! (coerce _lhs + coerce _rhs :: r s' t)

  lhs - rhs = Target dim $ reifyShape (dim ++ shapeVal (Proxy :: Proxy s)) result
    where (dim, _lhs, _rhs) = binary lhs rhs
          result :: forall s'. KnownShape s' => Proxy s' -> r s t 
          result _ = coerce $! (coerce _lhs - coerce _rhs :: r s' t)

  lhs * rhs = Target dim $ reifyShape (dim ++ shapeVal (Proxy :: Proxy s)) result
    where (dim, _lhs, _rhs) = binary lhs rhs
          result :: forall s'. KnownShape s' => Proxy s' -> r s t 
          result _ = coerce $! (coerce _lhs * coerce _rhs :: r s' t)

  abs (Target d operand) = Target d $ reifyShape (d ++ shapeVal (Proxy :: Proxy s)) result
    where result :: forall s'. KnownShape s' => Proxy s' -> r s t
          result _ = coerce $! (abs $ coerce operand :: r s' t)

  signum (Target d operand) = Target d $ reifyShape (d ++ shapeVal (Proxy :: Proxy s)) result
    where result :: forall s'. KnownShape s' => Proxy s' -> r s t
          result _ = coerce $! (signum $ coerce operand :: r s' t)

  negate (Target d operand) = Target d $ reifyShape (d ++ shapeVal (Proxy :: Proxy s)) result
    where result :: forall s'. KnownShape s' => Proxy s' -> r s t
          result _ = coerce $! (negate $ coerce operand :: r s' t)

  fromInteger i = Target [] (fromInteger i)

instance (T s t, forall s'. KnownShape s' => Fractional (r s' t), TensorOp r t, Transformable r t) => Fractional (Target r s t) where
  lhs / rhs = Target dim $ reifyShape (dim ++ shapeVal (Proxy :: Proxy s)) result
    where (dim, _lhs, _rhs) = binary lhs rhs
          result :: forall s'. KnownShape s' => Proxy s' -> r s t
          result _ = coerce $! (coerce _lhs / coerce _rhs :: r s' t)
  
  recip (Target d operand) = Target d $ reifyShape (d ++ shapeVal (Proxy :: Proxy s)) result
    where result :: forall s'. KnownShape s' => Proxy s' -> r s t
          result _ = coerce $! (recip $ coerce operand :: r s' t)

  fromRational r = Target [] (fromRational r)


instance (Tensorial t, TensorOp r t, Transformable r t) => TensorOp (Target r) t where
  unsafeBroadcast :: forall s0 s1. (T s0 t, T s1 t) => Target r s0 t -> [Integer] -> Target r s1 t
  unsafeBroadcast (Target dim operand) _map = Target dim $ 
    reifyShape (dim ++ shapeVal (Proxy :: Proxy s0)) $ \ s0' -> 
      reifyShape (dim ++ shapeVal (Proxy :: Proxy s1)) $ \ s1' -> 
        result s0' s1'
    where _map' = take (length dim) [0..] ++ fmap (+ (fromIntegral $ length dim)) _map
          result :: forall s0' s1'. (KnownShape s0', KnownShape s1') => Proxy s0' -> Proxy s1' -> r s1 t
          result _ _ =  
            let t0 :: r s0' t = coerce operand
                t1 :: r s1' t = unsafeBroadcast t0 _map' 
            in  coerce $! t1

  unsafeReduce :: forall s0 s1. (T s0 t, T s1 t) => Target r s0 t -> (Value -> Value -> AnyType -> BlockM Value) -> t -> [Integer] -> Target r s1 t
  unsafeReduce (Target dims operand) body initvalue redims = Target dims $ 
    reifyShape s0 $ \ s0' -> reifyShape s1 $ \ s1' -> 
      result s0' s1'
    where redims' = fmap (+ (fromIntegral $ length dims)) redims
          s0 = dims ++ shapeVal (Proxy :: Proxy s0)
          s1 = dims ++ shapeVal (Proxy :: Proxy s1)
          result :: forall s0' s1'. (KnownShape s0', KnownShape s1') => Proxy s0' -> Proxy s1' -> r s1 t
          result _ _ = 
            let t0 :: r s0' t = coerce operand
                t1 :: r s1' t = unsafeReduce t0 body initvalue redims'
            in  coerce $! t1

  unsafeDotGeneral :: forall s0 s1 s2. (T s0 t, T s1 t, T s2 t) => Target r s0 t -> Target r s1 t -> DotDimensionNumbersAttr -> Target r s2 t
  unsafeDotGeneral lhs rhs attr = Target dims $ 
    reifyShape s0 $ \ s0' -> 
      reifyShape s1 $ \ s1' -> 
        reifyShape s2 $ \ s2' -> 
          result s0' s1' s2'
    where (dims, _lhs, _rhs) = binary lhs rhs
          s0 = dims ++ shapeVal (Proxy :: Proxy s0)
          s1 = dims ++ shapeVal (Proxy :: Proxy s1)
          s2 = dims ++ shapeVal (Proxy :: Proxy s2)
          adder = (+) (fromIntegral $ length dims)
          attr' = DotDimensionNumbersAttr {
            getBatchingDims    = 
              let prefix = take (length dims) [0..] <&> \ a -> (a,a)
                  suffix = bimap adder adder <$> getBatchingDims attr
              in  prefix ++ suffix,
            getContractingDims = bimap adder adder <$> getContractingDims attr
          }
          result :: forall s0' s1' s2'. (KnownShape s0', KnownShape s1', KnownShape s2') => Proxy s0' -> Proxy s1' -> Proxy s2' -> r s2 t
          result _ _ _ = 
            let t0 :: r s0' t = coerce _lhs
                t1 :: r s1' t = coerce _rhs
                t2 :: r s2' t = unsafeDotGeneral t0 t1 attr'
            in  coerce $! t2

  splat = Target [] . splat


class Vectorizable f where
  type Vectorized (i :: Nat) f = r | r -> i f
  vmap' :: KnownNat i => [Integer] -> ([Integer] -> f) -> Vectorized i f

instance (T s t, TensorOp r t, Transformable r t) => Vectorizable (Target r s t) where
  type Vectorized i (Target r s t) = Target r (i ': s) t
  vmap' :: forall i. KnownNat i => [Integer] -> ([Integer] -> Target r s t) -> Vectorized i (Target r s t)
  vmap' dimmax f = Target dimmax $ coerce t
    where i = natVal (Proxy :: Proxy i)
          Target _ t = capture (f dimmax) (dimmax ++ [i])

instance (T s t, TensorOp r t, Transformable r t, Vectorizable f) => Vectorizable (Target r s t -> f) where
  type Vectorized i (Target r s t -> f) = Target r (i ': s) t -> Vectorized i f
  vmap' :: forall i. KnownNat i => [Integer] -> ([Integer] -> Target r s t -> f) -> Vectorized i (Target r s t -> f)
  vmap' dimmax f arg@(Target ds _) = vmap' dimmax' f'
    where i = natVal (Proxy :: Proxy i)
          f' dim = 
            let Target _ u = capture arg dim
            in  f dim $ Target (dim ++ [i]) (coerce u)
          dimmax' = if length dimmax < length ds then ds else dimmax

vmap :: (KnownNat i, Vectorizable (a -> b)) => (a -> b) -> Vectorized i (a -> b) 
vmap f = vmap' [] (const f)


instance T s t => Traceable (Target Tracer s t) where
  trace' i (Target d u) = reifyShape (d ++ shapeVal (Proxy :: Proxy s)) result
    where result :: forall s'. KnownShape s' => Proxy s' -> (IntMap Value -> BlockM (IntMap Value, [Value]), ([AnyType], [AnyType]))
          result _ = trace' i $! (coerce u :: Tracer s' t)

instance (T s t, Traceable f) => Traceable (Target Tracer s t -> f) where
  trace' i f = first (_type :) <$> trace' (i + 1) (f argn)
    where argn = Target [] $ Tracer (\ a -> (a, ) <$> blockArg i)
          _type = tensorType' (Proxy :: Proxy (Tracer s t))