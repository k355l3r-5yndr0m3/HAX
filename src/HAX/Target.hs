{-# LANGUAGE DataKinds #-}
{-# LANGUAGE QuantifiedConstraints #-}
{-# LANGUAGE TypeFamilyDependencies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE PatternSynonyms #-}
module HAX.Target where
import HAX.Tensor

import HAX.AD.Reverse

import Control.Exception hiding (TypeError)

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

-- TODO: Implement dynamic type 
newtype Target r s t = Tgt ([Integer], r s t)

pattern Target :: [Integer] -> r s t -> Target r s t
pattern Target d v = Tgt (d, v)
{-# COMPLETE Target #-}

instance IsList (r s t) => IsList (Target r s t) where
  type Item (Target r s t) = Item (r s t)

  fromList = Target [] . fromList
  toList   = undefined

capture :: forall r s t a. (TensorOp r, T s t) => Target r s t -> [Integer] -> (forall s'. KnownShape s' => r s' t -> a) -> a
capture (Target di u) dm fn
  | di == dm  = forceShape j (coerce u) fn
  | otherwise = assert (di `isSuffixOf` dm) $
    forceShape j (coerce u) $ \u' -> 
      forceShape k (unsafeBroadcast u' m) fn
  where m = fromIntegral <$> take (length j) [length k - length j..]
        s = shapeVal (Proxy :: Proxy s)
        j = di ++ s
        k = dm ++ s

unitary :: forall r s t a. (TensorOp r, T s t) => Target r s t -> (forall s'. KnownShape s' => [Integer] -> r s' t -> a) -> a
unitary u@(Target d _) func = capture u d (func d)

binary :: (TensorOp r, T s0 t0, T s1 t1) => Target r s0 t0 -> Target r s1 t1 -> 
          (forall s0' s1'. (KnownShape s0', KnownShape s1') => [Integer] -> r s0' t0 -> r s1' t1 -> a) -> a
binary a@(Target da _) b@(Target db _) func = 
  capture b dm (capture a dm (func dm))
  where dm | length da > length db = da
           | otherwise             = db
tertiary :: (TensorOp r, T s0 t0, T s1 t1, T s2 t2) => Target r s0 t0 -> Target r s1 t1 -> Target r s2 t2 -> 
            (forall s0' s1' s2'. (KnownShape s0', KnownShape s1', KnownShape s2') => [Integer] -> r s0' t0 -> r s1' t1 -> r s2' t2 -> a) -> a
tertiary a@(Target da _) b@(Target db _) c@(Target dc _) func = 
  capture c dm (capture b dm (capture a dm (func dm)))
  where dm | length da > length db = if length da > length dc then da else dc
           | length db > length dc = db 
           | otherwise             = dc

instance (ConvertOp r, TensorOp r) => ConvertOp (Target r) where
  convert :: forall s f g. (T s f, T s g) => Target r s f -> Target r s g
  convert operand = unitary operand (\dims -> Target dims . coerceShape . convert)

instance TensorOp r => TensorOp (Target r) where
  unsafeBroadcast :: forall s0 s1 t. (T s0 t, T s1 t) => Target r s0 t -> [Integer] -> Target r s1 t
  unsafeBroadcast operand axes = unitary operand (\dims operand' -> Target dims $ 
    let axes' = take (length dims) [0..] ++ fmap (+ (fromIntegral $ length dims)) axes 
    in  forceShape (dims ++ shapeVal s1) (unsafeBroadcast operand' axes') coerce)
    where s1    = Proxy :: Proxy s1

  unsafeTranspose :: forall s0 s1 t. (T s0 t, T s1 t) => Target r s0 t -> [Integer] -> Target r s1 t
  unsafeTranspose operand axes = unitary operand (\dims operand' -> Target dims $ 
    let axes' = [0..fromIntegral (length dims - 1)] ++ map (+ (fromIntegral $ length dims)) axes
    in  forceShape (dims ++ shapeVal s1) (unsafeBroadcast operand' axes') coerce)
    where s1 = Proxy :: Proxy s1

  unsafeReshape :: forall s0 s1 t. (TensorOp r, T s0 t, T s1 t) => Target r s0 t -> Target r s1 t
  unsafeReshape operand = unitary operand (\dims operand' -> Target dims $ 
    forceShape (dims ++ shapeVal s1) (unsafeReshape operand') coerce)
    where s1 = Proxy :: Proxy s1

  unsafeSlice :: forall s0 s1 t. (T s0 t, T s1 t) => Target r s0 t -> [(Integer, Integer, Integer)] -> Target r s1 t
  unsafeSlice operand slicing = unitary operand (\dims operand' -> Target dims $
    let slicing' = fmap (0, , 1) dims ++ slicing
    in  forceShape (dims ++ shapeVal s1) (unsafeSlice operand' slicing') coerce)
    where s1 = Proxy :: Proxy s1

  unsafePad :: forall s0 s1 t. (T s0 t, T s1 t) => t -> Target r s0 t -> [(Integer, Integer, Integer)] -> Target r s1 t
  unsafePad padval operand padding = unitary operand (\dims operand' -> Target dims $ 
    let padding' = ((0, 0, 0) <$ dims) ++ padding
    in  forceShape (dims ++ shapeVal s1) (unsafePad padval operand' padding') coerce)
    where s1 = Proxy :: Proxy s1

  unsafeReverse :: forall s t. (T s t) => Target r s t -> [Integer] -> Target r s t
  unsafeReverse operand axes = unitary operand (\dims operand' -> Target dims $ 
    let axes' = fmap (+(fromIntegral $ length dims)) axes in coerce (unsafeReverse operand' axes'))

  unsafeGather :: forall s0 s1 s2 t. (T s0 t, T s1 t, T s2 t) => Target r s0 t -> Target r s1 Int64 -> [Integer] -> [Integer] -> [Integer] -> Integer -> [Integer] -> Target r s2 t
  unsafeGather (Target [] operand) (Target [] start) offsetAxes collapsedAxes startAxesMap idxVectorAxis sliceSizes = Target [] $ 
    unsafeGather operand start offsetAxes collapsedAxes startAxesMap idxVectorAxis sliceSizes
  unsafeGather operand (Target [] start) offsetAxes collapsedAxes startAxesMap idxVectorAxis sliceSizes = unitary operand (\batching operand' -> Target batching $ 
    let batchRank      = fromIntegral $ length batching
        sliceSizes'    = batching ++ sliceSizes
        offsetAxes'    = [0..batchRank - 1] ++ ((+batchRank) <$> offsetAxes)
        collapsedAxes' = (+batchRank) <$> collapsedAxes
        startAxesMap'  = (+batchRank) <$> startAxesMap
    in  forceShape (batching ++ shapeVal (Proxy :: Proxy s2)) (unsafeGather operand' start offsetAxes' collapsedAxes' startAxesMap' idxVectorAxis sliceSizes') coerce)
  unsafeGather (Target [] operand) start offsetAxes collapsedAxes startAxesMap idxVectorAxis sliceSizes = unitary start (\batching start' -> Target batching $ 
    let batchRank      = fromIntegral $ length batching
        offsetAxes'    = (+batchRank) <$> offsetAxes
        idxVectorAxis' = batchRank + idxVectorAxis
    in  forceShape (batching ++ shapeVal (Proxy :: Proxy s2)) (unsafeGather operand start' offsetAxes' collapsedAxes startAxesMap idxVectorAxis' sliceSizes) coerce)
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

  unsafeScatter :: forall s0 s1 s2 t. (T s0 t, T s1 t, T s2 t) => Target r s0 t -> Target r s1 Int64 -> Target r s2 t -> [Integer] -> [Integer] -> [Integer] -> Integer -> Target r s0 t
  unsafeScatter (Target [] input) (Target [] indices) (Target [] update) uwd iwd sdtod ivd = Target [] $ unsafeScatter input indices update uwd iwd sdtod ivd
  unsafeScatter input (Target [] indices) update uwd iwd sdtod ivd = binary input update (\batching input' update' ->
    let batchRank = fromIntegral $ length batching
        sdtod' = (+batchRank) <$> sdtod
        uwd' = [0..batchRank - 1] ++ ((+batchRank) <$> uwd)
        iwd' = (+batchRank) <$> iwd
        ivd' = ivd
    in  Target batching $ coerce (unsafeScatter input' indices update' uwd' iwd' sdtod' ivd'))
  unsafeScatter input indices update uwd iwd sdtod ivd = tertiary input indices update (\batching input' indices' update' -> 
    let batchRank = fromIntegral $ length batching
        ivd' = batchRank + ivd
        indicesShape = batching ++ shapeVal (Proxy :: Proxy s1)
        iotaShape = changeAt (fromInteger ivd') (const batchRank) indicesShape
        newIndxShape = changeAt (fromInteger ivd') (+batchRank) indicesShape
        uwd' = (+batchRank) <$> uwd
        iwd' = [0..batchRank - 1] ++ ((+batchRank) <$> iwd)
        sdtod' = [0..batchRank - 1] ++ ((+batchRank) <$> sdtod)
    in  Target batching $
          forceShape iotaShape (unsafeMultiIota [0..batchRank - 1] ivd') (\iotaIdx -> 
          forceShape newIndxShape (unsafeConcat ivd' iotaIdx indices') (\newIndices ->
          coerce (unsafeScatter input' newIndices update' uwd' iwd' sdtod' ivd'))))
    where changeAt :: Int -> (a -> a) -> [a] -> [a]
          changeAt i f n
            | i >= 0    = 
              let changeAt' _ []     = []
                  changeAt' j (b:bs) = if j == 0 then f b:bs else b:changeAt' (j - 1) bs
              in  changeAt' i n
            | otherwise = error "Negative index"




  unsafeConcat :: forall s0 s1 s2 t. (T s0 t, T s1 t, T s2 t) => Integer -> Target r s0 t -> Target r s1 t -> Target r s2 t
  unsafeConcat axis low high = binary low high (\dims low' high' -> Target dims $ 
    let axis' = axis + fromIntegral (length dims)
    in  forceShape (dims ++ shapeVal s2) (unsafeConcat axis' low' high') coerce)
    where s2 = Proxy :: Proxy s2

  splat = Target [] . splat
  unsafeDotGeneral :: forall s0 s1 s2 t. (T s0 t, T s1 t, T s2 t, Num t) => Target r s0 t -> Target r s1 t -> DotDimensionNumbersAttr -> Target r s2 t
  unsafeDotGeneral lhs rhs attr = binary lhs rhs (\dims lhs' rhs' -> Target dims $ 
    let attr' = DotDimensionNumbersAttr {
                  getBatchingDims    = 
                    let prefix = take (length dims) [0..] <&> \ a -> (a,a)
                        suffix = bimap adder adder <$> getBatchingDims attr
                    in  prefix ++ suffix,
                  getContractingDims = bimap adder adder <$> getContractingDims attr
                }
        adder = (+) (fromIntegral $ length dims)
    in  forceShape (dims ++ shapeVal s2) (unsafeDotGeneral lhs' rhs' attr') coerce)
    where s2 = Proxy :: Proxy s2

  unsafeReduceAdd :: forall s0 s1 t. (T s0 t, T s1 t, Num t) => Target r s0 t -> [Integer] -> Target r s1 t
  unsafeReduceAdd operand axis = unitary operand (\dims operand' -> Target dims $
    let axis' = fmap (+(fromIntegral $ length dims)) axis
    in  forceShape (dims ++ shapeVal s1) (unsafeReduceAdd operand' axis') coerce)
    where s1 = Proxy :: Proxy s1
  unsafeReduceMul :: forall s0 s1 t. (T s0 t, T s1 t, Num t) => Target r s0 t -> [Integer] -> Target r s1 t
  unsafeReduceMul operand axis = unitary operand (\dims operand' -> Target dims $
    let axis' = fmap (+(fromIntegral $ length dims)) axis
    in  forceShape (dims ++ shapeVal s1) (unsafeReduceMul operand' axis') coerce)
    where s1 = Proxy :: Proxy s1

  -- TODO: Since this does not automatically perform inversion and padding, it is actually not convolution (mathematically, maybe change that)
  unsafeConvolution :: forall s0 s1 s2 t. (T s0 t, T s1 t, T s2 t, Num t) => Target r s0 t -> Target r s1 t -> Target r s2 t
  unsafeConvolution image (Target [] kernel) = unitary image (\dims image' ->
    Target dims $
      let inputShape   = shapeVal (Proxy :: Proxy s0)
          resultShape  = shapeVal (Proxy :: Proxy s2)
          batchSize    = head inputShape * product dims
          convInShape  = batchSize:tail inputShape
          convOutShape = batchSize:tail resultShape
      in  forceShape convInShape (unsafeReshape image') (\image'' -> 
          forceShape convOutShape (unsafeConvolution image'' kernel) (\output -> 
          forceShape (dims ++ resultShape) (unsafeReshape output) coerce)))
  unsafeConvolution input kernel = binary input kernel (\dims input' kernel' -> Target dims $
    let extraBatchDim  = fromIntegral $ length dims
        longBatchSize  = product dims
        shortBatchSize = head $ shapeVal s0
        totalBatchSize = longBatchSize * shortBatchSize
        kernelShape = dims ++ shapeVal s1
        kernelPerm = [i + extraBatchDim | i <- [0..shapeRank s1 - 1]] ++ [0..extraBatchDim - 1]
        middle = init . tail
        windowSize = middle $ shapeVal s0
        kernelSize = middle $ shapeVal s1
        outFeat    = last   $ shapeVal s1
        outputSize = [w - k + 1 | (w, k) <- zip windowSize kernelSize]
        reshapedOutputShape = longBatchSize:shortBatchSize:outputSize ++ [outFeat, longBatchSize]
        diagedOutputShape   = longBatchSize:shortBatchSize:outputSize ++ [outFeat]
        outputRemoveDim     = fromIntegral $ length diagedOutputShape
        outputFinalShape    = dims ++ shortBatchSize:outputSize ++ [outFeat]
    in  forceShape 
          (totalBatchSize : tail (shapeVal s0)) (unsafeReshape input') 
            (\reshapedInput ->
        forceShape 
          [kernelShape !! fromInteger i | i <- kernelPerm] (unsafeTranspose kernel' kernelPerm) 
            (\transposedKernel ->
        forceShape 
          (init (shapeVal s1) ++ [outFeat * longBatchSize]) (unsafeReshape transposedKernel)   
            (\reshapedKernel ->
        forceShape 
          (totalBatchSize:outputSize ++ [outFeat * longBatchSize]) (unsafeConvolution reshapedInput reshapedKernel) 
            (\output ->
        forceShape reshapedOutputShape 
          (unsafeReshape output) 
            (\reshapedOutput ->
        forceShape diagedOutputShape 
          (unsafeDiagonal 0 outputRemoveDim reshapedOutput) 
            (\diagedOutput ->
        forceShape outputFinalShape (unsafeReshape diagedOutput) coerce)))))))
    where s0 = Proxy :: Proxy s0
          s1 = Proxy :: Proxy s1

  unsafeIota          = Target [] . unsafeIota 
  unsafeLinspace axis = Target [] . unsafeLinspace axis

  unsafePairwiseAdd lhs rhs = binary lhs rhs (\dims lhs' -> Target dims . coerce . unsafePairwiseAdd lhs' . assumeEqShape)
  unsafePairwiseSub lhs rhs = binary lhs rhs (\dims lhs' -> Target dims . coerce . unsafePairwiseSub lhs' . assumeEqShape)
  unsafePairwiseMul lhs rhs = binary lhs rhs (\dims lhs' -> Target dims . coerce . unsafePairwiseMul lhs' . assumeEqShape)
  unsafePairwiseDiv lhs rhs = binary lhs rhs (\dims lhs' -> Target dims . coerce . unsafePairwiseDiv lhs' . assumeEqShape)

  unsafePairwiseAbs operand = unitary operand (\dims -> Target dims . coerce . unsafePairwiseAbs)
  unsafePairwiseExp operand = unitary operand (\dims -> Target dims . coerce . unsafePairwiseExp)
  unsafePairwiseLog operand = unitary operand (\dims -> Target dims . coerce . unsafePairwiseLog)
  unsafePairwiseSin operand = unitary operand (\dims -> Target dims . coerce . unsafePairwiseSin)
  unsafePairwiseCos operand = unitary operand (\dims -> Target dims . coerce . unsafePairwiseCos)

  unsafePairwiseTanh operand = unitary operand (\dims -> Target dims . coerce . unsafePairwiseTanh)

  unsafePairwiseNegate operand = unitary operand (\dims -> Target dims . coerce . unsafePairwiseNegate)
  unsafePairwiseSignum operand = unitary operand (\dims -> Target dims . coerce . unsafePairwiseSignum)

  select false true cond = tertiary false true cond (\dims false' true' -> Target dims . coerce . select false' (assumeEqShape true') . assumeEqShape)

  branch false true (Target [] cond) = binary false true (\dims false' true' -> Target dims . coerce . branch false' (assumeEqShape true') $ cond)
  branch false true cond = branch false true (unsafeBroadcast cond [])

  isEQ lhs rhs = binary lhs rhs (\dims lhs' -> Target dims . coerce . isEQ lhs' . assumeEqShape)
  isNE lhs rhs = binary lhs rhs (\dims lhs' -> Target dims . coerce . isNE lhs' . assumeEqShape)
  isGT lhs rhs = binary lhs rhs (\dims lhs' -> Target dims . coerce . isGT lhs' . assumeEqShape)
  isGE lhs rhs = binary lhs rhs (\dims lhs' -> Target dims . coerce . isGE lhs' . assumeEqShape)
  isLT lhs rhs = binary lhs rhs (\dims lhs' -> Target dims . coerce . isLT lhs' . assumeEqShape)
  isLE lhs rhs = binary lhs rhs (\dims lhs' -> Target dims . coerce . isLE lhs' . assumeEqShape)

  unsafeArgmax :: forall s s' t. (Ord t, T s t, T s' t) => Int -> Target r s t -> Target r s' Int64
  unsafeArgmax axis operand = unitary operand (\dims operand' -> Target dims $
    let axis' = axis + length dims 
    in  forceShape (dims ++ shapeVal s') (unsafeArgmax axis' operand') coerce)
    where s' = Proxy :: Proxy s'

-- -- VMap transform
class Vectorizable f where
  type Vectorized (i :: Nat) f = r | r -> i f
  vmap' :: KnownNat i => [Integer] -> ([Integer] -> f) -> Vectorized i f

instance (T s t, TensorOp r, Transformable r) => Vectorizable (Target r s t) where
  type Vectorized i (Target r s t) = Target r (i ': s) t
  vmap' :: forall i. KnownNat i => [Integer] -> ([Integer] -> Target r s t) -> Vectorized i (Target r s t)
  vmap' dm f = Target dm $ capture (f dm) (dm ++ [i]) coerce
    where i = natVal (Proxy :: Proxy i)

instance (T s t, TensorOp r, Transformable r, Vectorizable f) => Vectorizable (Target r s t -> f) where
  type Vectorized i (Target r s t -> f) = Target r (i ': s) t -> Vectorized i f
  vmap' :: forall i. KnownNat i => [Integer] -> ([Integer] -> Target r s t -> f) -> Vectorized i (Target r s t -> f)
  vmap' dm f arg@(Target ds _) = vmap' dm' f'
    where i    = natVal (Proxy :: Proxy i)
          dm'  = if length dm < length ds then ds else dm
          f' d = capture arg d (f d . Target (d ++ [i]) . coerce)
vmap :: (KnownNat i, Vectorizable (a -> b)) => (a -> b) -> Vectorized i (a -> b) 
vmap f = vmap' [] (const f)

instance JNT r => JNT (Target r) where
  fromTracer = Target [] . fromTracer
  toTracer (Target _ i) = toTracer i

instance (Transformable r, GNT r) => GNT (Target r) where
  type Ins (Target r) = Ins r
  fromReverse = Target [] . fromReverse
  toReverse (Target _ r) = toReverse r
