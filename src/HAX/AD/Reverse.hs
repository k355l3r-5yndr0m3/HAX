{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE DefaultSignatures #-}
module HAX.AD.Reverse where
import Prelude hiding (reverse)
import HAX.Tensor

import HAX.AD.Gradient
import HAX.Utils

import Data.Proxy
import Data.List hiding (reverse)
import Data.Kind

import Foreign.C

import GHC.IsList
import GHC.TypeError
import GHC.Generics
import Data.Bifunctor (Bifunctor(first))
import Control.Exception (assert)
import Data.Coerce (coerce)

-- TODO: Consider using coerse instead of Dynamic for gradient
--       Slightly more safe and more performance
-- TODO: Use pattern syn 
-- TODO: Remove overlapping instances
newtype Reverse  r s t = Reverse (r s t, G r s t) deriving Generic
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
  assumeEqShape :: forall s s' t. (KnownShape s, KnownShape s') => Reverse r s t -> Reverse r s' t
  assumeEqShape = assert (shapeVal (Proxy :: Proxy s) == shapeVal (Proxy :: Proxy s')) coerce

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
  unsafePairwiseDiv (R f f') (R g g') = R (unsafePairwiseDiv f g) $ \i -> f' (i &/ g) <+> g' (unsafePairwiseNegate ((i &* f) &/ (g &* g)))
    where (&*) = unsafePairwiseMul
          (&/) = unsafePairwiseDiv
  
  unsafePairwiseNegate (R f f') = R (unsafePairwiseNegate f) (f' . unsafePairwiseNegate)
  unsafePairwiseAbs    (R f f') = R (unsafePairwiseAbs f)    (\i -> f' $ unsafePairwiseSignum f `unsafePairwiseMul` i)
  unsafePairwiseSignum (R f _)  = R (unsafePairwiseSignum f) nograd

  unsafePairwiseSin    (R f f') = R (unsafePairwiseSin f) (\i -> f' (i `unsafePairwiseMul` unsafePairwiseCos f))
  unsafePairwiseCos    (R f f') = R (unsafePairwiseCos f) (\i -> f' (unsafePairwiseNegate i `unsafePairwiseMul` unsafePairwiseSin f))
  unsafePairwiseTanh   (R f f') = R y (\i -> f' (i `unsafePairwiseSub` (i `unsafePairwiseMul` (y `unsafePairwiseMul` y))))
    where y = unsafePairwiseTanh f
  unsafePairwiseExp (R f f') = 
    Reverse (y, f' . (`unsafePairwiseMul` y))
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

  unsafeArgmax axis (R f _) = R (unsafeArgmax axis f) nograd

class Grad f where  
  type GradF'  f g
  type FGradF' f g
  type GradF   f
  type FGradF  f
  grad' :: CIntPtr -> (Gradient -> (g, Gradient)) -> f -> GradF' f g
  grad  :: f -> GradF f

  fgrad' :: CIntPtr -> (Gradient -> (g, Gradient)) -> f -> FGradF' f g
  fgrad  :: f -> FGradF f

instance (GNT r, T s t, Fractional t) => Grad (r s t) where
  type GradF'  (r s t) g = g
  type FGradF' (r s t) g = (Ins r s t, g)

  type GradF  _          = TypeError (Text "rgrad must be applied to a function")
  type FGradF _          = TypeError (Text "rgrad must be applied to a function")
  grad' _ recover (toReverse -> R _ g) = fst . recover . g . splat $ 1
  grad = undefined

  fgrad' _ recover (toReverse -> R f g) = (f, fst . recover . g . splat $ 1)
  fgrad = undefined

instance (GradIn t, Grad a) => Grad (t -> a) where
  type GradF'  (t -> a) g = GradI t -> GradF'  a (g <&> GradI t)
  type FGradF' (t -> a) g = GradI t -> FGradF' a (g <&> GradI t)

  type GradF  (t -> a)    = GradI t -> GradF'  a (GradI t)
  type FGradF (t -> a)    = GradI t -> FGradF' a (GradI t) 
  grad' i recover f t = grad' i' recover' (f t')
    where recover' g = 
            let (a,  g' ) = recover g
                (as, g'') = rec g'
            in  (a :&: as, g'')
          (i', t', rec) = gradIn i t
  grad f t = grad' i' recover (f t')
    where (i', t', recover) = gradIn 0 t

  fgrad' i recover f t = fgrad' i' recover' (f t')
    where recover' g = 
            let (a,  g' ) = recover g
                (as, g'') = rec g'
            in  (a :&: as, g'')
          (i', t', rec) = gradIn i t
  fgrad f t = fgrad' i' recover (f t')
    where (i', t', recover) = gradIn 0 t

class GGradIn t where
  type GGradI t :: k -> Type
  gGradIn :: CIntPtr -> GGradI t x -> (CIntPtr, t x, Gradient -> (GGradI t x, Gradient))
instance GGradIn V1 where
  type GGradI V1 = V1
  gGradIn i v = (i, v, (v, ))
instance GGradIn U1 where
  type GGradI U1 = U1
  gGradIn i v = (i, v, (v, ))
instance (GGradIn a, GGradIn b) => GGradIn (a :+: b) where
  type GGradI (a :+: b) = GGradI a :+: GGradI b
  gGradIn i (L1 a) = (i', L1 a', first L1 . g)
    where (i', a', g) = gGradIn i a
  gGradIn i (R1 b) =(i', R1 b', first R1 . g)
    where (i', b', g) = gGradIn i b
instance (GGradIn a, GGradIn b) => GGradIn (a :*: b) where
  type GGradI (a :*: b) = GGradI a :*: GGradI b
  gGradIn i (a :*: b) = (i'', a' :*: b', \gr -> 
    let (_a, g' ) = ag gr
        (_b, g'') = bg g'
    in  (_a :*: _b, g''))
    where (i' , a', ag) = gGradIn i  a
          (i'', b', bg) = gGradIn i' b
instance GradIn c => GGradIn (K1 i c) where
  type GGradI (K1 i c) = K1 i (GradI c)
  gGradIn i (K1 c) = (i', K1 c', first K1 . g)
    where (i', c', g) = gradIn i c
instance GGradIn f => GGradIn (M1 i t f) where
  type GGradI (M1 i t f) = M1 i t (GGradI f)
  gGradIn i (M1 f) = (i', M1 f', first M1 . g)
    where (i', f', g) = gGradIn i f

class GradIn t where
  type GradI t
  gradIn :: CIntPtr -> GradI t -> (CIntPtr, t, Gradient -> (GradI t, Gradient))
  default gradIn :: (Generic (GradI t), Generic t, GGradIn (Rep t), Rep (GradI t) ~ GGradI (Rep t)) => CIntPtr -> GradI t -> (CIntPtr, t, Gradient -> (GradI t, Gradient))
  gradIn i (from -> t) = (i', to t', first to . g)
    where (i', t', g) = gGradIn i t

class TensorOp (Ins r) => GNT (r :: Z) where
  type Ins r :: Z
  fromReverse :: Reverse (Ins r) s t -> r s t
  toReverse   :: r s t -> Reverse (Ins r) s t

instance TypeError (Text "Reverse wrapper is required") => GNT Tensor where
  type Ins Tensor = Tensor
  fromReverse = undefined
  toReverse   = undefined

instance TypeError (Text "Reverse wrapper is required") => GNT Tracer where
  type Ins Tracer = Tracer
  fromReverse = undefined
  toReverse   = undefined

instance TensorOp r => GNT (Reverse r) where
  type Ins (Reverse r) = r
  fromReverse = id
  toReverse   = id

instance (T s t, GNT r) => GradIn (r s t) where
  type GradI (r s t) = (Ins r) s t
  gradIn i r = (i + 1, fromReverse $ R r (independent i), \(Gradient gs) ->
    let (fmap snd -> g, gs') = partition ((== i) . fst) gs
    in  (gradientSum g, Gradient gs'))
instance GradIn Rational where
  type GradI Rational = Rational
  gradIn = (, , (0,))

instance GradIn a => GradIn [a] where
  type GradI [a] = [GradI a]

instance (GradIn a, GradIn b) => GradIn (a, b) where
  type GradI (a, b) = (GradI a, GradI b)

instance (GradIn a, GradIn b) => GradIn (a <&> b) where
  type GradI (a <&> b) = GradI a <&> GradI b

instance JNT r => JNT (Reverse r) where
  fromTracer = Reverse . (, undefined) . fromTracer
  toTracer = toTracer . primal 
