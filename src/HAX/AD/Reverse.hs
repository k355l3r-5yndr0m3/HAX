{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE UndecidableInstances #-}
module HAX.AD.Reverse where
import Prelude hiding (reverse)
import HAX.Tensor.Tensorial
import HAX.Tensor.Tensor

import HAX.AD.Gradient
import HAX.Utils

import Control.Exception

import Data.Proxy
import Data.List hiding (reverse)

import Foreign.C

import Data.Int (Int64)
import GHC.IsList
import Data.Word (Word8)
import Data.Data (Typeable)
import HAX.Tensor.Tracer (Tracer)

-- TODO: Consider using coerse instead of Dynamic for gradient
--       Slightly more safe and more performance
-- TODO: Use pattern syn 
-- TODO: Remove overlapping instances
newtype Reverse r s t = Reverse (r s t, G r s t)
primal :: Reverse r s t -> r s t 
primal (Reverse t) = fst t

cotangent :: Reverse r s t -> G r s t
cotangent (Reverse t) = snd t

pattern R :: r s t -> G r s t -> Reverse r s t
pattern R p c = Reverse (p, c)
{-# COMPLETE R #-}

instance IsList (r s t) => IsList (Reverse r s t) where
  type Item (Reverse r s t) = Item (r s t)
  fromList = Reverse . (, nograd) . fromList


-- ConvertOp 
instance ConvertOp r => ConvertOp (Reverse r) where
  convert (Reverse (f, f')) = Reverse (convert f, f' . convert)

-- Warning: if in unsafeGather clipping occure, the gradient will not be correct
-- TODO: Fix by adding manual clipping

-- TODO: Maybe add a type class (or whatever it is called) instead of this constraint, a litle safer
-- NOTE: This is strange, this constraint require undiciable instance if Reverse is constrained to Reverse (r :: Shape -> Type -> Type) (s :: Shape) (t :: Type)
--        Why? (More reason to do the above)
instance TensorOp r => TensorOp (Reverse r) where
  unsafeBroadcast (R f f') dims = R (unsafeBroadcast f dims) (unsafeBroadcastGrad f' dims)
  unsafeTranspose (R f f') perm = R (unsafeTranspose f perm) (f' . (`unsafeTranspose` perm'))
    where perm' = map snd $ sortOn fst $ zip perm [0..] 
  unsafeReshape   (R f f')      = R (unsafeReshape   f)      (f' . unsafeReshape)
  unsafeSlice     (R f f') slic = R (unsafeSlice     f slic) (unsafeSliceGrad     f' slic)
  unsafePad padv  (R f f') padd = R (unsafePad padv  f padd) (unsafePadGrad       f' padd)
    where unsafePadGrad :: forall s0 s1 t. (TensorOp r, T s0 t, T s1 t) => G r s0 t -> [(Integer, Integer, Integer)] -> r s1 t -> Gradient
          unsafePadGrad g' padding i = g' $ unsafeSlice i slicing
            where (low, high, internal) = unzip3 padding
                  slicing = zipWith4 (\l h a j -> (l, a - h, j + 1)) low high s internal
                  s = shapeVal (Proxy :: Proxy s1)
  unsafeReverse   (R f f') dims = R (unsafeReverse   f dims) (\i -> f' $ unsafeReverse i dims)
  unsafeScatter   (R f f') (R g _) (R h h') uwd iwd sdtod ivd =
    R (unsafeScatter f g h uwd iwd sdtod ivd) (unsafeScatterGrad f' g h' uwd iwd sdtod ivd)
  unsafeGather    (R f f') (R g _) offsetAxes collapsedAxes startAxisMap idxVectorAxis sliceSizes =
    R (unsafeGather f g offsetAxes collapsedAxes startAxisMap idxVectorAxis sliceSizes) 
      (unsafeGatherGrad f' g offsetAxes collapsedAxes startAxisMap idxVectorAxis)
  unsafeConcat axis (R f f') (R g g') = 
    R (unsafeConcat axis f g) (unsafeConcatGrad axis f' g')
    where unsafeConcatGrad :: forall s0 s1 s2 t. (TensorOp r, T s0 t, T s1 t, T s2 t) => Integer -> G r s0 t -> G r s1 t -> r s2 t -> Gradient
          unsafeConcatGrad dims _f' _g' i =
            _f' (unsafeSlice i lhsSlicing) <+> _g' (unsafeSlice i rhsSlicing)
            where lhsSlicing = (0, , 1) <$> shapeVal (Proxy :: Proxy s0)
                  offs = shapeVal (Proxy :: Proxy s0) !! fromInteger dims
                  limt = shapeVal (Proxy :: Proxy s2) !! fromInteger dims
                  rhsSlicing = [if d == dims then (offs, limt, 1) else (0, s, 1) | (d, s) <- zip [0..] $ shapeVal (Proxy :: Proxy s1)]


  splat t = R (splat t) nograd

  unsafeLinspace a r = R (unsafeLinspace a r) nograd
  unsafeIota i = R (unsafeIota i) nograd

  unsafeDotGeneral (R f f') (R g g') attr = R (unsafeDotGeneral f g attr) (unsafeDotGeneralGrad f f' g g' attr)

  unsafeReduceAdd (R f f') dims = R (unsafeReduceAdd f dims) (unsafeReduceAddGrad f' dims)
  unsafeReduceMul (R f f') dims = R (unsafeReduceMul f dims) (unsafeReduceMulGrad f f' dims)

  unsafePairwiseAdd (R f f') (R g g') = R (unsafePairwiseAdd f g) (\i -> f' i <+> g' i)
  unsafePairwiseSub (R f f') (R g g') = R (unsafePairwiseSub f g) (\i -> f' i <+> g' (unsafePairwiseNegate i))
  unsafePairwiseMul (R f f') (R g g') = R (unsafePairwiseMul f g) (\i -> f' (unsafePairwiseMul i g) <+> g' (unsafePairwiseMul f i))
  unsafePairwiseDiv (R f f') (R g g') = R (unsafePairwiseDiv f g) (\i -> f' (i `unsafePairwiseDiv` g) <+> g' (unsafePairwiseNegate (i `unsafePairwiseMul` (f `unsafePairwiseDiv` (g `unsafePairwiseMul` g)))))
  
  unsafePairwiseNegate (R f f') = R (unsafePairwiseNegate f) (f' . unsafePairwiseNegate)
  unsafePairwiseAbs    (R f f') = R (unsafePairwiseAbs f)    (\i -> f' $ unsafePairwiseSignum f `unsafePairwiseMul` i)
  unsafePairwiseSignum (R f _)  = R (unsafePairwiseSignum f) nograd

  unsafePairwiseSin    (R f f') = R (unsafePairwiseSin f) (\i -> f' (i `unsafePairwiseMul` unsafePairwiseCos f))
  unsafePairwiseCos    (R f f') = R (unsafePairwiseCos f) (\i -> f' (unsafePairwiseNegate i `unsafePairwiseMul` unsafePairwiseSin f))
  unsafePairwiseTanh   (R f f') = R y (\i -> f' (i `unsafePairwiseSub` (i `unsafePairwiseMul` (y `unsafePairwiseMul` y))))
    where y = unsafePairwiseTanh f
  unsafePairwiseExp (R f f') = 
    Reverse (y, \i -> f' (i `unsafePairwiseMul` y))
    where y = unsafePairwiseExp f
  unsafePairwiseLog (R f f') = 
    R (unsafePairwiseLog f) (\i -> f' (i `unsafePairwiseDiv` f))


  unsafeConvolution (R f f') (R g g') = R (unsafeConvolution f g) (unsafeConvolutionGrad f f' g g')

  branch (R f f') (R t t') (R c _) = R (branch f t c) (branchGrad f' t' c)
  select (R f f') (R t t') (R c _) = R (select f t c) (selectGrad f' t' c)

  isEQ (Reverse (lhs, _)) (Reverse (rhs, _)) = Reverse (isEQ lhs rhs, const zero)
  isNE (Reverse (lhs, _)) (Reverse (rhs, _)) = Reverse (isNE lhs rhs, const zero)

  isGT (Reverse (lhs, _)) (Reverse (rhs, _)) = Reverse (isGT lhs rhs, const zero)
  isGE (Reverse (lhs, _)) (Reverse (rhs, _)) = Reverse (isGE lhs rhs, const zero)
  isLT (Reverse (lhs, _)) (Reverse (rhs, _)) = Reverse (isLT lhs rhs, const zero)
  isLE (Reverse (lhs, _)) (Reverse (rhs, _)) = Reverse (isLE lhs rhs, const zero)




instance JitIn (r s t) => JitIn (Reverse r s t) where 
  type JitI (Reverse r s t) = JitI (r s t) -- Should just evaluate to Tensor s t 
  jitIn i t = (i', R t' undefined, bs)
    where (i', t', bs) = jitIn i t

instance JitOut (r s t) => JitOut (Reverse r s t) where
  type JitO (Reverse r s t) = JitO (r s t)
  jitOut = jitOut . primal

-- Reversable
-- class Reversable r where
--   type ReversedType r
--   constructReverse :: CIntPtr -> ReversedType r -> (CIntPtr, r)
--   gradReify :: Proxy r -> CIntPtr -> Gradient -> (CIntPtr, ReversedType r, Gradient)
-- 
-- instance (TensorOp r, KnownShape s) => Reversable (Reverse r s Bool) where
--   type ReversedType (Reverse r s Bool) = r s Bool
--   constructReverse i t = (i + 1, Reverse (t, nograd))
--   gradReify _ i (Gradient gs) = (i + 1, splat False, g')
--     where (_, Gradient -> g') = partition ((i ==) . fst) gs
-- 
-- instance (TensorOp r, KnownShape s) => Reversable (Reverse r s Int64) where
--   type ReversedType (Reverse r s Int64) = r s Int64
--   constructReverse i t = (i + 1, Reverse (t, nograd))
--   gradReify _ i (Gradient gs) = (i + 1, splat 0, g')
--     where (_, Gradient -> g') = partition ((i ==) . fst) gs
-- 
-- instance (TensorOp r, KnownShape s) => Reversable (Reverse r s Word8) where
--   type ReversedType (Reverse r s Word8) = r s Word8
--   constructReverse i t = (i + 1, Reverse (t, nograd))
--   gradReify _ i (Gradient gs) = (i + 1, splat 0, g')
--     where (_, Gradient -> g') = partition ((i ==) . fst) gs
-- 
-- instance Cotangent (r s Float) => Reversable (Reverse r s Float) where
--   type ReversedType (Reverse r s Float) = r s Float
--   constructReverse i t = (i + 1, Reverse (t, independent i))
--   gradReify _ i (Gradient gs) = (i + 1, sum (fromDyn' <$> g), g')
--     where (fmap snd -> g, Gradient -> g') = partition ((i ==) . fst) gs
-- 
-- instance (Reversable a, Reversable b) => Reversable (a <&> b) where
--   type ReversedType (a <&> b) = ReversedType a <&> ReversedType b
--   constructReverse i0 (a :&: b) = (i2, a' :&: b')
--     where (i1, a') = constructReverse i0 a
--           (i2, b') = constructReverse i1 b
--   gradReify _ i0 g0 = (i2, a' :&: b', g2)
--     where (i1, a', g1) = gradReify (Proxy :: Proxy a) i0 g0
--           (i2, b', g2) = gradReify (Proxy :: Proxy b) i1 g1

-- TODO: Implement General
-- class ReverseMode f where
--   type Rev g f
--   type GradResult f
--   rgrad' :: (Gradient -> g, CIntPtr) -> f -> Rev g f
--   rgradReify :: Annotated CIntPtr f -> Gradient -> GradResult f
-- 
-- instance (Reversable j, Num (r s t)) => ReverseMode (j -> Reverse r s t) where
--   type Rev g (j -> Reverse r s t)      = ReversedType j -> g
--   type GradResult (j -> Reverse r s t) = ReversedType j
--   rgrad' (reifier, i) f t = reifier $ cotangent (f $ snd $ constructReverse i t) 1
--   rgradReify (Annotated i) (gradReify (Proxy :: Proxy j) i  -> (_, g, Gradient g')) = assert (null g') g
-- 
-- instance (Reversable j, ReverseMode (a -> b)) => ReverseMode (j -> (a -> b)) where
--   type Rev g (j -> (a -> b))      = ReversedType j -> Rev g (a -> b)
--   type GradResult (j -> (a -> b)) = ReversedType j <&> GradResult (a -> b)
--   rgrad' (reifier, i) f t = rgrad' (reifier, i') (f r)
--     where (i', r) = constructReverse i t
--   rgradReify (Annotated i) (gradReify (Proxy :: Proxy j) i -> (i', g, g')) = g :&: rgradReify (Annotated i' :: Annotated CIntPtr (a -> b)) g'
-- 
-- type RGrad f = Rev (GradResult f) f

-- rgrad :: forall f. ReverseMode f => f -> RGrad f
rgrad :: Grad f => f -> GradF f
rgrad = grad 

instance (Num t, T s t, TensorOp r) => Num (Reverse r s t) where
  (+) = unsafePairwiseAdd
  (-) = unsafePairwiseSub
  (*) = unsafePairwiseMul

  abs = unsafePairwiseAbs
  negate = unsafePairwiseNegate
  signum = unsafePairwiseSignum

  fromInteger = splat . fromInteger

instance (Fractional t, T s t, TensorOp r) => Fractional (Reverse r s t) where
  (/) = unsafePairwiseDiv
  fromRational = splat . fromRational

instance (Floating t, T s t, TensorOp r) => Floating (Reverse r s t) where
  pi = splat pi
  exp = unsafePairwiseExp
  log = unsafePairwiseLog
  sin = unsafePairwiseSin
  cos = unsafePairwiseCos


class Grad f where  
  type GradF' f g
  type GradF  f
  grad' :: CIntPtr -> (Gradient -> (g, Gradient)) -> f -> GradF' f g
  grad  :: f -> GradF f

instance (GradIn a, TensorOp r, Tensorial t, Fractional t) => Grad (a -> Reverse r '[] t) where
  type GradF' (a -> Reverse r '[] t) g = GradI a -> g <&> GradI a
  type GradF  (a -> Reverse r '[] t)   = GradI a -> GradI a
  grad' i recover f t = fst $ recover' (n $ splat 1)
    where recover' g = 
            let (a,  g' ) = recover g
                (as, g'') = rec g'
            in  (a :&: as, g'')
          (t', _, rec) = gradIn i t
          R _ n = f t'
  grad f t = fst $ recover $ g $ splat 0
    where (t', _, recover) = gradIn 0 t
          R _ g            = f t'

instance (GradIn t, Grad (a -> b)) => Grad (t -> a -> b) where
  type GradF' (t -> a -> b) g = GradI t -> GradF' (a -> b) (g <&> GradI t)
  type GradF  (t -> a -> b)   = GradI t -> GradF' (a -> b) (GradI t)
  grad' i recover f t = grad' i' recover' (f t')
    where recover' g = 
            let (a,  g' ) = recover g
                (as, g'') = rec g'
            in  (a :&: as, g'')
          (t', i', rec) = gradIn i t
  grad f t = grad' i' recover (f t')
    where (t', i', recover) = gradIn 0 t



class GradIn t where
  type GradI t
  gradIn :: CIntPtr -> GradI t -> (t, CIntPtr, Gradient -> (GradI t, Gradient))

instance (T s t, TensorOp r) => GradIn (Reverse r s t) where
  type GradI (Reverse r s t) = r s t
  gradIn i r = (R r (independent i), i + 1, \(Gradient gs) ->
    let (fmap snd -> g, gs') = partition ((== i) . fst) gs
    in  (gradientSum g, Gradient gs'))

instance GradIn a => GradIn [a] where
  type GradI [a] = [GradI a]
  gradIn i []     = ([], i, ([], ))
  gradIn i (a:as) = (a':as', i'', \gr -> 
    let (r , gr' ) = g' gr
        (r', gr'') = g'' gr'
    in  (r:r', gr''))
    where (a' , i' , g' ) = gradIn i  a
          (as', i'', g'') = gradIn i' as 

instance (GradIn a, GradIn b) => GradIn (a, b) where
  type GradI (a, b) = (GradI a, GradI b)
  gradIn i (a, b) = ((a', b'), i'', \gr -> 
    let (_a, g' ) = ag gr
        (_b, g'') = bg g'
    in  ((_a, _b), g''))
    where (a', i' , ag) = gradIn i  a
          (b', i'', bg) = gradIn i' b

instance (GradIn a, GradIn b) => GradIn (a <&> b) where
  type GradI (a <&> b) = GradI a <&> GradI b
  gradIn i (a :&: b) = (a' :&: b', i'', \gr -> 
    let (_a, g' ) = ag gr
        (_b, g'') = bg g'
    in  (_a :&: _b, g''))
    where (a', i' , ag) = gradIn i  a
          (b', i'', bg) = gradIn i' b




