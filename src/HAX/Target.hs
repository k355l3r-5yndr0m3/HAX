{-# LANGUAGE DataKinds #-}
{-# LANGUAGE QuantifiedConstraints #-}
{-# LANGUAGE TypeFamilyDependencies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE ViewPatterns #-}
module HAX.Target where
-- import HAX.Tensor.Tracer
import HAX.Tensor.Tensorial
import HAX.Tensor.Tensor (Tensor)

import HAX.AD.Gradient
import HAX.AD.Reverse

import HAX.Jit
import HAX.Utils

import Control.Exception

import Data.Coerce
import Data.Proxy 
import Data.List
import Data.Functor
import Data.Bifunctor

import Stablehlo.Dialect.Stablehlo.Attributes

import GHC.TypeLits
import Data.Int (Int64)
import GHC.IsList

-- Target: represent the target of vmap transformation
--         Target dims r 
--         dims ++ shapeVal r is the true shape
-- TODO: Implement an entire unsafe version of Target without dimension,
--       simplify the impementation

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

instance IsList (r s t) => IsList (Target r s t) where
  type Item (Target r s t) = Item (r s t)

  fromList = Target [] . fromList

capture :: forall r s t. (T s t, ShapeOp r t, Transformable r t) => Target r s t -> [Integer] -> r s t 
capture (Target di u) dmax 
  | di == dmax = u
  | otherwise  = assert (di `isSuffixOf` dmax) $
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

binary :: (T s0 t0, T s1 t1, Transformable r t0, ShapeOp r t0, ShapeOp r t1, Transformable r t1) => Target r s0 t0 -> Target r s1 t1 -> ([Integer], r s0 t0, r s1 t1)
binary l@(Target ld lhs) r@(Target rd rhs) 
  | ld == rd           = (ld, lhs, rhs)
  | ld `isSuffixOf` rd = 
    let lhs' = capture l rd
    in  (rd, lhs', rhs)
  | rd `isSuffixOf` ld = 
    let rhs' = capture r ld 
    in  (ld, lhs, rhs')
  | otherwise = error $ "The impossible has happen: " ++ show ld ++ " vs " ++ show rd 

tertiary :: (T s0 t0, T s1 t1, T s2 t2, Transformable r t0, Transformable r t1, Transformable r t2, ShapeOp r t0, ShapeOp r t1, ShapeOp r t2) => Target r s0 t0 -> Target r s1 t1 -> Target r s2 t2 -> ([Integer], r s0 t0, r s1 t1, r s2 t2)
tertiary a@(Target ad _) b@(Target bd _) c@(Target cd _) = (dims, capture a dims, capture b dims, capture c dims)
  where dims
          | length ad > length bd = if length ad > length cd then ad else cd
          | length bd > length cd = bd
          | otherwise             = cd

instance (T s t, forall s'. KnownShape s' => Num (r s' t), ShapeOp r t, Transformable r t) => Num (Target r s t) where
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

instance (T s t, forall s'. KnownShape s' => Fractional (r s' t), ShapeOp r t, Transformable r t) => Fractional (Target r s t) where
  lhs / rhs = Target dim $ reifyShape (dim ++ shapeVal (Proxy :: Proxy s)) result
    where (dim, _lhs, _rhs) = binary lhs rhs
          result :: forall s'. KnownShape s' => Proxy s' -> r s t
          result _ = coerce $! (coerce _lhs / coerce _rhs :: r s' t)
  
  recip (Target d operand) = Target d $ reifyShape (d ++ shapeVal (Proxy :: Proxy s)) result
    where result :: forall s'. KnownShape s' => Proxy s' -> r s t
          result _ = coerce $! (recip $ coerce operand :: r s' t)

  fromRational r = Target [] (fromRational r)

instance (T s t, forall s'. KnownShape s' => Floating (r s' t), ShapeOp r t, Transformable r t) => Floating (Target r s t) where
  pi = Target [] pi

  sin (Target d operand) = Target d $ reifyShape (d ++ shapeVal (Proxy :: Proxy s)) result
    where result :: forall s'. KnownShape s' => Proxy s' -> r s t
          result _ = coerce $! (sin $ coerce operand :: r s' t)
  cos (Target d operand) = Target d $ reifyShape (d ++ shapeVal (Proxy :: Proxy s)) result
    where result :: forall s'. KnownShape s' => Proxy s' -> r s t
          result _ = coerce $! (cos $ coerce operand :: r s' t)
  tanh (Target d operand) = Target d $ reifyShape (d ++ shapeVal (Proxy :: Proxy s)) result
    where result :: forall s'. KnownShape s' => Proxy s' -> r s t
          result _ = coerce $! (tanh $ coerce operand :: r s' t)

  exp (Target d operand) = Target d $ reifyShape (d ++ shapeVal (Proxy :: Proxy s)) result
    where result :: forall s'. KnownShape s' => Proxy s' -> r s t
          result _ = coerce $! (exp $ coerce operand :: r s' t)
  log (Target d operand) = Target d $ reifyShape (d ++ shapeVal (Proxy :: Proxy s)) result
    where result :: forall s'. KnownShape s' => Proxy s' -> r s t
          result _ = coerce $! (log $ coerce operand :: r s' t)

instance (ConvertOp r, forall t. Transformable r t) => ConvertOp (Target r) where
  convert :: forall s f g. (T s f, T s g) => Target r s f -> Target r s g
  convert (Target dims operand) = Target dims $ reifyShape (dims ++ shapeVal (Proxy :: Proxy s)) $ \shape ->
    let operand' = same (scoerce operand) shape
        result'  = same (convert operand') shape
    in  scoerce result'
    where same :: KnownShape s' => r s' t -> Proxy s' -> r s' t
          same i _ = i
          scoerce :: Coercible (r a b) (r c b) => r a b -> r c b
          scoerce = coerce 

instance (Tensorial t, ShapeOp r t, Transformable r t, MathOp r Int64, Transformable r Int64) => ShapeOp (Target r) t where
  -- NOTE: haskell cannot determine the write method to call so this is a fix
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

  unsafeTranspose :: forall s0 s1. (KnownShape s0, KnownShape s1) => Target r s0 t -> [Integer] -> Target r s1 t
  unsafeTranspose (Target dim operand) perm = Target dim $ 
    reifyShape (dim ++ shapeVal (Proxy :: Proxy s0)) $ \s -> 
      reifyShape (dim ++ shapeVal (Proxy :: Proxy s1)) $ \s' ->
        result s s'
    where result :: forall s s'. (KnownShape s, KnownShape s') => Proxy s -> Proxy s' -> r s1 t 
          result _ _ = 
            let t0 :: r s  t = coerce operand
                t1 :: r s' t = unsafeTranspose t0 perm'
            in  coerce $! t1
          perm' = [0..fromIntegral (length dim - 1)] ++ map (+ (fromIntegral $ length dim)) perm

  unsafeReshape :: forall s0 s1. (ShapeOp r t, KnownShape s0, KnownShape s1) => Target r s0 t -> Target r s1 t
  unsafeReshape (Target dim operand) = Target dim $
    reifyShape (dim ++ shapeVal (Proxy :: Proxy s0)) $ \s0 ->
      reifyShape (dim ++ shapeVal (Proxy :: Proxy s1)) $ \s1 ->
        result s0 s1
    where result :: forall z0 z1. (KnownShape z0, KnownShape z1) => Proxy z0 -> Proxy z1 -> r s1 t
          result _ _ =
            let operand' :: r z0 t = coerce operand
            in  coerce $! (unsafeReshape operand' :: r z1 t)

  unsafeSlice :: forall s0 s1. (KnownShape s0, KnownShape s1) => Target r s0 t -> [(Integer, Integer, Integer)] -> Target r s1 t
  unsafeSlice (Target dims operand) slicing = Target dims $ 
    reifyShape (dims ++ shapeVal (Proxy :: Proxy s0)) $ 
      reifyShape (dims ++ shapeVal (Proxy :: Proxy s1))
        result 
    where result :: forall s0' s1'. (KnownShape s0', KnownShape s1') => Proxy s1' -> Proxy s0' -> r s1 t
          result _ _ = 
            let _operand :: r s0' t = coerce operand
                _result  :: r s1' t = unsafeSlice _operand slicing'
            in  coerce $! _result
          slicing' = fmap (0, , 1) dims ++ slicing

  unsafePad :: forall s0 s1. (KnownShape s0, KnownShape s1) => t -> Target r s0 t -> [(Integer, Integer, Integer)] -> Target r s1 t
  unsafePad padval (Target dims operand) padding = Target dims $ 
    reifyShape (dims ++ shapeVal (Proxy :: Proxy s0)) $ 
      reifyShape (dims ++ shapeVal (Proxy :: Proxy s1))
        result 
    where result :: forall s0' s1'. (KnownShape s0', KnownShape s1') => Proxy s1' -> Proxy s0' -> r s1 t
          result _ _ = 
            let _operand :: r s0' t = coerce operand
                _result  :: r s1' t = unsafePad padval _operand padding'
            in  coerce $! _result
          padding' = ((0, 0, 0) <$ dims) ++ padding

  unsafeReverse :: forall s0. (KnownShape s0) => Target r s0 t -> [Integer] -> Target r s0 t
  unsafeReverse (Target dims operand) reverseDims = Target dims $ 
    reifyShape (dims ++ shapeVal (Proxy :: Proxy s0)) result
    where result :: forall s0'. KnownShape s0' => Proxy s0' -> r s0 t
          result _ =
            let _operand :: r s0' t = coerce operand
                _result  :: r s0' t = unsafeReverse _operand ((+offset) <$> reverseDims)
            in  undefined
          offset = fromIntegral $ length dims
  
  unsafeGather :: forall s0 s1 s2. (T s0 t, T s1 t, T s2 t) => Target r s0 t -> Target r s1 Int64 -> [Integer] -> [Integer] -> [Integer] -> Integer -> [Integer] -> Target r s2 t
  unsafeGather (Target [] operand) (Target [] start) offsetAxes collapsedAxes startAxesMap idxVectorAxis sliceSizes = Target [] $ 
    unsafeGather operand start offsetAxes collapsedAxes startAxesMap idxVectorAxis sliceSizes
  unsafeGather (Target batching operand) (Target [] start) offsetAxes collapsedAxes startAxesMap idxVectorAxis sliceSizes = Target batching $ 
    reifyShape (batching ++ shapeVal (Proxy :: Proxy s0)) $ \(same (coerce operand) -> operand') ->  
      reifyShape (batching ++ shapeVal (Proxy :: Proxy s2)) $ \(same (unsafeGather operand' start offsetAxes' collapsedAxes' startAxesMap' idxVectorAxis sliceSizes') -> output) ->
        coerce $! output
    where batchRank      = fromIntegral $ length batching
          offsetAxes'    = [0..batchRank - 1] ++ ((+batchRank) <$> offsetAxes)
          collapsedAxes' = (+batchRank) <$> collapsedAxes
          startAxesMap'  = (+batchRank) <$> startAxesMap
          sliceSizes'    = batching ++ sliceSizes
          same :: r s t -> Proxy s -> r s t
          same = const
  unsafeGather (Target [] operand) (Target batching start) offsetAxes collapsedAxes startAxesMap idxVectorAxis sliceSizes = Target batching $ 
    reifyShape (batching ++ shapeVal (Proxy :: Proxy s1)) $ \(same (coerce start) -> start') ->
      reifyShape (batching ++ shapeVal (Proxy :: Proxy s2)) $ \(same' (unsafeGather operand start' offsetAxes' collapsedAxes startAxesMap idxVectorAxis' sliceSizes) -> output') ->
        coerce $! output'
    where batchRank      = fromIntegral $ length batching
          offsetAxes'    = (+batchRank) <$> offsetAxes
          idxVectorAxis' = batchRank + idxVectorAxis
          same :: r s Int64 -> Proxy s -> r s Int64
          same = const
          same' :: r s t -> Proxy s -> r s t
          same' = const
  unsafeGather (Target opBatch operand) (Target stBatch start) offsetAxes collapsedAxes startAxesMap idxVectorAxis sliceSizes = 
    reifyShape (batchSize:shapeVal (Proxy :: Proxy s0)) $ \(same (coerce operand) -> operand') -> 
      reifyShape (batchSize:shapeVal (Proxy :: Proxy s1)) $ \(same' (coerce start) -> start') -> 
        reifyShape iotaShape $ \(sameT (unsafeIota idxVectorAxis') -> iota') ->
          let oper = Target opBatch' operand'
              star = Target stBatch' start'
              outshape = batchSize:shapeVal (Proxy :: Proxy s2)
          in  reifyShape concatedShape $ \(sameT (unsafeConcat idxVectorAxis' iota' star) -> star') ->
                 reifyShape outshape $ \(sameR (unsafeGather oper star' offsetAxes' collapsedAxes' startAxesMap' idxVectorAxis' sliceSizes') -> Target d u) ->
                   Target (d ++ [batchSize]) $ coerce $! u
    where batchSize = assert (last opBatch == last stBatch) $ last stBatch
          opBatch'  = init opBatch
          stBatch'  = init stBatch
          offsetAxes'    = (+1) <$> offsetAxes
          collapsedAxes' = 0:((+1) <$> collapsedAxes)
          startAxesMap'  = 0:((+1) <$> startAxesMap)
          idxVectorAxis' = idxVectorAxis + 1
          sliceSizes'    = 0:sliceSizes
          iotaShape      = batchSize:(take (fromIntegral idxVectorAxis) (shapeVal (Proxy :: Proxy s1)) ++ 1:drop (fromIntegral idxVectorAxis + 1) (shapeVal (Proxy :: Proxy s1)))
          concatedShape  = batchSize:(take (fromIntegral idxVectorAxis) (shapeVal (Proxy :: Proxy s1)) ++ (1 + shapeVal (Proxy :: Proxy s1) !! fromIntegral idxVectorAxis):drop (fromIntegral idxVectorAxis + 1) (shapeVal (Proxy :: Proxy s1)))
          same :: r s t -> Proxy s -> r s t
          same = const
          same' :: r s Int64 -> Proxy s -> r s Int64
          same' = const
          sameT :: Target r s Int64 -> Proxy s -> Target r s Int64 
          sameT = const
          sameR :: Target r s t -> Proxy s -> Target r s t
          sameR = const
  
  unsafeScatter :: forall s0 s1 s2 .(T s0 t, T s1 t, T s2 t) => Target r s0 t -> Target r s1 Int64 -> Target r s2 t -> [Integer] -> [Integer] -> [Integer] -> Integer -> Target r s0 t
  unsafeScatter (Target [] input) (Target [] indices) (Target [] update) uwd iwd sdtod ivd = Target [] $ unsafeScatter input indices update uwd iwd sdtod ivd
  unsafeScatter input (Target [] indices) update uwd iwd sdtod ivd = 
    reifyShape (batching ++ shapeVal (Proxy :: Proxy s0)) $ \inputShapeProxy ->
      reifyShape (batching ++ shapeVal (Proxy :: Proxy s2)) $ \(same (scoerce _update) -> update') ->
        let input'  = same (scoerce _input) inputShapeProxy
            output' = same (unsafeScatter input' indices update' uwd' iwd' sdtod' ivd') inputShapeProxy
        in  Target batching (coerce $! output')
    where (batching, _input, _update) = binary input update
          batchRank = fromIntegral $ length batching
          sdtod' = (+batchRank) <$> sdtod
          uwd' = [0..batchRank - 1] ++ ((+batchRank) <$> uwd)
          iwd' = (+batchRank) <$> iwd
          ivd' = ivd
          same :: KnownShape g => k g h -> Proxy g -> k g h 
          same = const
          scoerce :: Coercible (k g h) (k g' h) => k g h -> k g' h
          scoerce = coerce
  unsafeScatter input indices update uwd iwd sdtod ivd = Target batching $
    reifyShape (batching ++ shapeVal (Proxy :: Proxy s0)) $ \s0proxy -> 
      reifyShape indicesShape $ \(same (scoerce _indices) -> indices') ->
        reifyShape (batching ++ shapeVal (Proxy :: Proxy s2)) $ \(same (scoerce _update) -> update') ->
          reifyShape iotaShape $ \(same (unsafeMultiIota [0..batchRank - 1] ivd') -> iotaIdx) -> 
            reifyShape newIndxShape $ \(same (unsafeConcat ivd' iotaIdx indices') -> newIndices) ->
              let input'  = same (scoerce _input) s0proxy
                  output' = same (unsafeScatter input' newIndices update' uwd' iwd' sdtod' ivd') s0proxy
              in  coerce $! output'
    where (batching, _input, _indices, _update) = tertiary input indices update
          batchRank = fromIntegral $ length batching
          ivd' = batchRank + ivd
          indicesShape = batching ++ shapeVal (Proxy :: Proxy s1)
          iotaShape = changeAt (fromInteger ivd') (const batchRank) indicesShape
          newIndxShape = changeAt (fromInteger ivd') (+batchRank) indicesShape
          uwd' = (+batchRank) <$> uwd
          iwd' = [0..batchRank - 1] ++ ((+batchRank) <$> iwd)
          sdtod' = [0..batchRank - 1] ++ ((+batchRank) <$> sdtod)
          same :: KnownShape g => k g h -> Proxy g -> k g h 
          same = const
          scoerce :: Coercible (k g h) (k g' h) => k g h -> k g' h
          scoerce = coerce
          changeAt :: Int -> (a -> a) -> [a] -> [a]
          changeAt i f n
            | i >= 0    = 
              let changeAt' _ []     = []
                  changeAt' j (b:bs) = if j == 0 then f b:bs else b:changeAt' (j - 1) bs
              in  changeAt' i n
            | otherwise = error "Negative index"
      
          

  
          
  
  unsafeConcat :: forall s0 s1 s2. (KnownShape s0, KnownShape s1, KnownShape s2) => Integer -> Target r s0 t -> Target r s1 t -> Target r s2 t
  unsafeConcat d lhs rhs = Target b $
    reifyShape (b ++ shapeVal (Proxy :: Proxy s0)) $ \(same (coerce _lhs) -> lhs') ->
      reifyShape (b ++ shapeVal (Proxy :: Proxy s1)) $ \(same (coerce _rhs) -> rhs') -> 
        reifyShape (b ++ shapeVal (Proxy :: Proxy s2)) $ \(same (unsafeConcat (d + fromIntegral (length b)) lhs' rhs') -> result') ->
          coerce $! result'
    where (b, _lhs, _rhs) = binary lhs rhs
          same :: r h t -> Proxy h -> r h t
          same i _ = i

  splat = Target [] . splat

instance (MathOp r t, ShapeOp r t, Transformable r t, Transformable r Int64, MathOp r Int64) => MathOp (Target r) t where
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

  unsafeReduceAdd :: forall s0 s1. (T s0 t, T s1 t) => Target r s0 t -> [Integer] -> Target r s1 t
  unsafeReduceAdd (Target dims operand) axies = Target dims $ 
    reifyShape s0 $ \ s0' -> reifyShape s1 $ \ s1' -> 
      result s0' s1'
    where axies' = fmap (+ (fromIntegral $ length dims)) axies
          s0 = dims ++ shapeVal (Proxy :: Proxy s0)
          s1 = dims ++ shapeVal (Proxy :: Proxy s1)
          result :: forall s0' s1'. (KnownShape s0', KnownShape s1') => Proxy s0' -> Proxy s1' -> r s1 t
          result _ _ = 
            let t0 :: r s0' t = coerce operand
                t1 :: r s1' t = unsafeReduceAdd t0 axies'
            in  coerce $! t1

  unsafeReduceMul :: forall s0 s1. (T s0 t, T s1 t) => Target r s0 t -> [Integer] -> Target r s1 t
  unsafeReduceMul (Target dims operand) axies = Target dims $ 
    reifyShape s0 $ \ s0' -> reifyShape s1 $ \ s1' -> 
      result s0' s1'
    where axies' = fmap (+ (fromIntegral $ length dims)) axies
          s0 = dims ++ shapeVal (Proxy :: Proxy s0)
          s1 = dims ++ shapeVal (Proxy :: Proxy s1)
          result :: forall s0' s1'. (KnownShape s0', KnownShape s1') => Proxy s0' -> Proxy s1' -> r s1 t
          result _ _ = 
            let t0 :: r s0' t = coerce operand
                t1 :: r s1' t = unsafeReduceMul t0 axies'
            in  coerce $! t1

  unsafeConvolution :: forall s0 s1 s2. (T s0 t, T s1 t, T s2 t) => Target r s0 t -> Target r s1 t -> Target r s2 t
  unsafeConvolution (Target dims input) (Target [] kernel) = Target dims $ -- degenerate case
    reifyShape (dims ++ inputShape) $
      reifyShape convInShape $ 
        reifyShape (dims ++ resultShape) $ 
          reifyShape convOutShape
            result
    where inputShape   = shapeVal (Proxy :: Proxy s0)
          resultShape  = shapeVal (Proxy :: Proxy s2)
          batchSize    = head inputShape * product dims
          convInShape  = batchSize:tail inputShape
          convOutShape = batchSize:tail resultShape
          result :: forall inS reS convInS convOutS. (KnownShape inS, KnownShape reS, KnownShape convInS, KnownShape convOutS) => 
                      Proxy convOutS -> Proxy reS -> Proxy convInS -> Proxy inS -> r s2 t
          result _ _ _ _ = 
            let _input   :: r inS t      = coerce input
                _input'  :: r convInS t  = unsafeReshape _input
                _result  :: r convOutS t = unsafeConvolution _input' kernel
                _result' :: r reS t      = unsafeReshape _result
            in  coerce $! _result'
  unsafeConvolution input kernel = Target dims $
    reifyShape inputShape $ \(same (coerce _input) -> input') -> 
      reifyShape kernelShape $ \(same (coerce _kernel) -> kernel') ->
        reifyShape (totalBatchSize : tail (shapeVal (Proxy :: Proxy s0))) $ \(same (unsafeReshape input') -> reshapedInput) ->
          reifyShape [kernelShape !! fromInteger i | i <- kernelPerm] $ \(same (unsafeTranspose kernel' kernelPerm) -> transposedKernel) ->
            reifyShape (init (shapeVal (Proxy :: Proxy s1)) ++ [outFeat * longBatchSize]) $ \(same (unsafeReshape transposedKernel) -> reshapedKernel) ->
              reifyShape (totalBatchSize:outputSize ++ [outFeat * longBatchSize]) $ \(same (unsafeConvolution reshapedInput reshapedKernel) -> output) ->
                reifyShape reshapedOutputShape $ \(same (unsafeReshape output) -> reshapedOutput) ->
                  reifyShape diagedOutputShape $ \(same (unsafeDiagonal 0 outputRemoveDim reshapedOutput) -> diagedOutput) ->
                    reifyShape outputFinalShape $ \(same (unsafeReshape diagedOutput) -> outputFinal) ->
                      coerce $! outputFinal
    where (dims, _input, _kernel) = binary input kernel
          extraBatchDim  = fromIntegral $ length dims
          longBatchSize  = product dims
          shortBatchSize = head $ shapeVal (Proxy :: Proxy s0)
          totalBatchSize = longBatchSize * shortBatchSize
          inputShape  = dims ++ shapeVal (Proxy :: Proxy s0)
          kernelShape = dims ++ shapeVal (Proxy :: Proxy s1)
          same :: r h t -> Proxy h -> r h t
          same i _ = i
          kernelPerm = [i + extraBatchDim | i <- [0..shapeRank (Proxy :: Proxy s1) - 1]] ++ [0..extraBatchDim - 1]
          middle = init . tail
          windowSize = middle $ shapeVal (Proxy :: Proxy s0)
          kernelSize = middle $ shapeVal (Proxy :: Proxy s1)
          outFeat    = last   $ shapeVal (Proxy :: Proxy s1)
          outputSize = [w - k + 1 | (w, k) <- zip windowSize kernelSize]
          reshapedOutputShape = longBatchSize:shortBatchSize:outputSize ++ [outFeat, longBatchSize]
          diagedOutputShape   = longBatchSize:shortBatchSize:outputSize ++ [outFeat]
          outputRemoveDim     = fromIntegral $ length diagedOutputShape
          outputFinalShape    = dims ++ shortBatchSize:outputSize ++ [outFeat]
          

  unsafeIota dims = Target [] $ unsafeIota dims
  linspace = Target [] . linspace

instance (SelectOp r t, ShapeOp r t, Transformable r t, ShapeOp r Bool, Transformable r Bool, MathOp r Int64, Transformable r Int64) => SelectOp (Target r) t where
  branch :: forall s. KnownShape s => Target r s t -> Target r s t -> Target r '[] Bool -> Target r s t
  branch false true (Target [] cond) = Target dims $ 
    reifyShape (dims ++ shape) $ \s ->
      result s
    where shape = shapeVal (Proxy :: Proxy s)
          (dims, false', true') = binary false true
          result :: forall s'. KnownShape s' => Proxy s' -> r s t
          result _ =
            let f :: r s' t = coerce false'
                t :: r s' t = coerce true'
            in  coerce $! branch f t cond
  branch false true (broadcast' -> cond) = select false true cond

  select :: forall s. KnownShape s => Target r s t -> Target r s t -> Target r s Bool -> Target r s t
  select false true cond = Target dims $
    reifyShape (dims ++ shape) $ \s -> 
      result s
    where (dims, false', true', cond') = tertiary false true cond
          shape = shapeVal (Proxy :: Proxy s)
          result :: forall s'. KnownShape s' => Proxy s' -> r s t
          result _ = 
            let f :: r s' t     = coerce false'
                t :: r s' t     = coerce true'
            in  coerce $! select f t (coerce cond')
instance (EqualOp r t, ShapeOp r t, Transformable r t, Transformable r Bool) => EqualOp (Target r) t where
  isEQ :: forall s. KnownShape s => (Target r) s t -> (Target r) s t -> (Target r) s Bool
  isEQ lhs rhs = Target dims $ 
    reifyShape (dims ++ shape) $ \s ->
      result s
    where (dims, _lhs, _rhs) = binary lhs rhs
          shape = shapeVal (Proxy :: Proxy s)
          result :: forall _s. KnownShape _s => Proxy _s -> r s Bool
          result _ =
            let lhs' :: r _s t = coerce _lhs
                rhs' :: r _s t = coerce _rhs
            in  coerce $! isEQ lhs' rhs'
  isNE :: forall s. KnownShape s => (Target r) s t -> (Target r) s t -> (Target r) s Bool
  isNE lhs rhs = Target dims $ 
    reifyShape (dims ++ shape) $ \s ->
      result s
    where (dims, _lhs, _rhs) = binary lhs rhs
          shape = shapeVal (Proxy :: Proxy s)
          result :: forall _s. KnownShape _s => Proxy _s -> r s Bool
          result _ =
            let lhs' :: r _s t = coerce _lhs
                rhs' :: r _s t = coerce _rhs
            in  coerce $! isNE lhs' rhs'
instance (OrderOp r t, ShapeOp r t, Transformable r t, Transformable r Bool) => OrderOp (Target r) t where
  isGT :: forall s. KnownShape s => (Target r) s t -> (Target r) s t -> (Target r) s Bool
  isGT lhs rhs = Target dims $ 
    reifyShape (dims ++ shape) $ \s ->
      result s
    where (dims, _lhs, _rhs) = binary lhs rhs
          shape = shapeVal (Proxy :: Proxy s)
          result :: forall _s. KnownShape _s => Proxy _s -> r s Bool
          result _ =
            let lhs' :: r _s t = coerce _lhs
                rhs' :: r _s t = coerce _rhs
            in  coerce $! isGT lhs' rhs'
  isGE :: forall s. KnownShape s => (Target r) s t -> (Target r) s t -> (Target r) s Bool
  isGE lhs rhs = Target dims $ 
    reifyShape (dims ++ shape) $ \s ->
      result s
    where (dims, _lhs, _rhs) = binary lhs rhs
          shape = shapeVal (Proxy :: Proxy s)
          result :: forall _s. KnownShape _s => Proxy _s -> r s Bool
          result _ =
            let lhs' :: r _s t = coerce _lhs
                rhs' :: r _s t = coerce _rhs
            in  coerce $! isGE lhs' rhs'
  isLT :: forall s. KnownShape s => (Target r) s t -> (Target r) s t -> (Target r) s Bool
  isLT lhs rhs = Target dims $ 
    reifyShape (dims ++ shape) $ \s ->
      result s
    where (dims, _lhs, _rhs) = binary lhs rhs
          shape = shapeVal (Proxy :: Proxy s)
          result :: forall _s. KnownShape _s => Proxy _s -> r s Bool
          result _ =
            let lhs' :: r _s t = coerce _lhs
                rhs' :: r _s t = coerce _rhs
            in  coerce $! isLT lhs' rhs'
  isLE :: forall s. KnownShape s => (Target r) s t -> (Target r) s t -> (Target r) s Bool
  isLE lhs rhs = Target dims $ 
    reifyShape (dims ++ shape) $ \s ->
      result s
    where (dims, _lhs, _rhs) = binary lhs rhs
          shape = shapeVal (Proxy :: Proxy s)
          result :: forall _s. KnownShape _s => Proxy _s -> r s Bool
          result _ =
            let lhs' :: r _s t = coerce _lhs
                rhs' :: r _s t = coerce _rhs
            in  coerce $! isLE lhs' rhs'

-- VMap transform
class Vectorizable f where
  type Vectorized (i :: Nat) f = r | r -> i f
  vmap' :: KnownNat i => [Integer] -> ([Integer] -> f) -> Vectorized i f

instance (T s t, ShapeOp r t, Transformable r t) => Vectorizable (Target r s t) where
  type Vectorized i (Target r s t) = Target r (i ': s) t
  vmap' :: forall i. KnownNat i => [Integer] -> ([Integer] -> Target r s t) -> Vectorized i (Target r s t)
  vmap' dimmax f = Target dimmax $ coerce t
    where i = natVal (Proxy :: Proxy i)
          t = capture (f dimmax) (dimmax ++ [i])

instance (T s t, ShapeOp r t, Transformable r t, Vectorizable f) => Vectorizable (Target r s t -> f) where
  type Vectorized i (Target r s t -> f) = Target r (i ': s) t -> Vectorized i f
  vmap' :: forall i. KnownNat i => [Integer] -> ([Integer] -> Target r s t -> f) -> Vectorized i (Target r s t -> f)
  vmap' dimmax f arg@(Target ds _) = vmap' dimmax' f'
    where i = natVal (Proxy :: Proxy i)
          f' dim = 
            let u = capture arg dim
            in  f dim $ Target (dim ++ [i]) (coerce u)
          dimmax' = if length dimmax < length ds then ds else dimmax
vmap :: (KnownNat i, Vectorizable (a -> b)) => (a -> b) -> Vectorized i (a -> b) 
vmap f = vmap' [] (const f)

-- So that Target work for other transformations
-- TODO: Implement a feature for vmaping gradient function
instance (T s t, TraceableElement (r s t), Transformable r t) => TraceableElement (Target r s t) where
  constructTracer i = (i', Target [] t, tt)
    where (i', t, tt) = constructTracer i
  
  deconstructTracer (Target [] t) = 
    deconstructTracer t
  deconstructTracer _ = error "deconstructTracer received an invalid target."

instance Reversable (Reverse r s t) => Reversable (Target (Reverse r) s t) where
  type ReversedType (Target (Reverse r) s t) = ReversedType (Reverse r s t)
  constructReverse i t = (i', Target [] r)
    where (i', r) = constructReverse i t
  gradReify _ = gradReify (Proxy :: Proxy (Reverse r s t))

instance (Reversable j, Num (r s t)) => ReverseMode (j -> Target (Reverse r) s t) where
  type Rev g (j -> Target (Reverse r) s t)      = ReversedType j -> g
  type GradResult (j -> Target (Reverse r) s t) = ReversedType j

  rgrad' (reifier, i) f t = assert (null dims) reifier $ cotangent r 1
    where Target dims r = f $ snd $ constructReverse i t
  rgradReify (Annotated i) (gradReify (Proxy :: Proxy j) i  -> (_, g, Gradient g')) = assert (null g') g

type instance JitTransform (Target r s t) = Tensor s t
