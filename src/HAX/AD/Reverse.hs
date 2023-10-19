{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE QuantifiedConstraints #-}
{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE OverloadedRecordDot #-}
module HAX.AD.Reverse where
import HAX.Tensor.Tensorial

import HAX.AD.Gradient
import HAX.Utils

import Control.Exception

import Data.Proxy
import Data.List

import Foreign.C

import Stablehlo.Dialect.Stablehlo.Attributes

-- TODO: Consider using coerse instead of Dynamic for gradient
--       Slightly more safe and more performance
-- TODO: Use pattern syn 
newtype Reverse r s t = Reverse (r s t, r s t -> Gradient)
primal :: Reverse r s t -> r s t 
primal (Reverse t) = fst t

cotangent :: Reverse r s t -> r s t -> Gradient 
cotangent (Reverse t) = snd t

-- TODO: Implement none differentiable types
instance Num (r s t) => Num (Reverse r s t) where
  (Reverse (f, f')) + (Reverse (g, g')) = 
    Reverse (f + g, \i -> f' i <+> g' i)

  (Reverse (f, f')) - (Reverse (g, g')) = 
    Reverse (f - g, \i -> f' i <+> g' (negate i))

  (Reverse (f, f')) * (Reverse (g, g')) = 
    Reverse (f * g, \i -> f' (i * g) <+> g' (i * f))

  negate (Reverse (f, f')) = 
    Reverse (negate f, f' . negate)

  abs    (Reverse (f, f')) = 
    Reverse (abs f, \i -> f' (i * signum f))

  signum (Reverse (f, f')) = 
    Reverse (signum f, \_ -> f' 0 )
  
  fromInteger a = 
    Reverse (fromInteger a, const zero)

-- TODO: Implement none differentiable types
instance Fractional (r s t) => Fractional (Reverse r s t) where
  recip (Reverse (f, f')) = 
    Reverse (r, \i -> f' (negate i * (r * r)))
    where r = recip f
  (Reverse (f, f')) / (Reverse (g, g')) = 
    Reverse (f / g, \i -> f' (i / g) <+> g' (negate (i * f / (g * g))))
  fromRational r = 
    Reverse (fromRational r, const zero)

-- TODO: Implement none differentiable types
instance Floating (r s t) => Floating (Reverse r s t) where
  pi = Reverse (pi, const zero)
  sin (Reverse (f, f')) = 
    Reverse (sin f, \i -> f' (i * cos f))
  cos (Reverse (f, f')) = 
    Reverse (cos f, \i -> f' (negate i * sin f))
  tanh (Reverse (f, f')) = 
    Reverse (y, \i -> f' (i - i * y * y))
    where y = tanh f
  exp (Reverse (f, f')) = 
    Reverse (y, \i -> f' (i * y))
    where y = exp f
  log (Reverse (f, f')) = 
    Reverse (log f, \i -> f' (i / f))

-- ShapeOp
differentiableUnsafeBroadcast :: forall r t s0 s1. (ShapeOp r t, MathOp r t, Num t, KnownShape s0, KnownShape s1) => Reverse r s0 t -> [Integer] -> Reverse r s1 t
differentiableUnsafeBroadcast (Reverse (f, f')) _map = 
  Reverse (unsafeBroadcast f _map, \ i -> 
    let reduceDims = 
          let generator :: (Integer, [Integer], Integer) -> [Integer]
              generator (lower, []  , upper) = [lower..upper]
              generator (lower, a:as, upper) = [lower..a - 1] ++ generator (a + 1, as, upper)
          in  generator (0, sort _map, shapeRank (Proxy :: Proxy s1) - 1)
        (secondBroadcast, reductionResult) = 
          let indexedIndices = zip (zip [0..] (shapeVal (Proxy :: Proxy s0))) _map 
          in  unzip (fst <$> sortOn snd indexedIndices)
        derivative :: forall c. KnownShape c => Proxy c -> r s0 t
        derivative _ = 
          let t :: r c t = unsafeReduceAdd i reduceDims
          in  unsafeBroadcast t secondBroadcast
    in  f' $ reifyShape reductionResult derivative)

differentiableUnsafeTranspose :: forall r t s0 s1. (ShapeOp r t, KnownShape s0, KnownShape s1) => Reverse r s0 t -> [Integer] -> Reverse r s1 t
differentiableUnsafeTranspose (Reverse (f, f')) perm = 
  Reverse (unsafeTranspose f perm, \ i -> 
    let perm' = map snd $ sortOn fst $ zip perm [0..] 
    in  f' (unsafeTranspose i perm'))

differentiableUnsafeReshape :: forall r t s0 s1. (ShapeOp r t, KnownShape s0, KnownShape s1) => Reverse r s0 t -> Reverse r s1 t
differentiableUnsafeReshape (Reverse (f, f')) = Reverse (unsafeReshape f, f' . unsafeReshape)

instance (ShapeOp r Float, MathOp r Float) => ShapeOp (Reverse r) Float where
  splat i = Reverse (splat i, const zero)
  unsafeBroadcast = differentiableUnsafeBroadcast
  unsafeTranspose = differentiableUnsafeTranspose
  unsafeReshape   = differentiableUnsafeReshape

indifferentiable :: Reverse r s t -> (r s t -> r s' t) -> Reverse r s' t
indifferentiable (Reverse (f, _)) y = Reverse (y f, const zero)
instance {-# OVERLAPPABLE #-} ShapeOp r t => ShapeOp (Reverse r) t where
  splat i = Reverse (splat i, const zero)
  unsafeBroadcast x _map = indifferentiable x (`unsafeBroadcast` _map)
  unsafeTranspose x perm = indifferentiable x (`unsafeTranspose` perm)
  unsafeReshape   x      = indifferentiable x unsafeReshape

-- MathOp
differentiableUnsafeReduceMul :: forall r t s0 s1. (ShapeOp r t, MathOp r t, Fractional (r s0 t), Num (r s1 t), T s0 t, T s1 t, Num t) => Reverse r s0 t -> [Integer] -> Reverse r s1 t
differentiableUnsafeReduceMul (Reverse (f, f')) dims = 
  Reverse (g, \ i -> 
    let _map = 
          let generator :: (Integer, [Integer], Integer) -> [Integer]
              generator (lower, []  , upper) = [lower..upper]
              generator (lower, a:as, upper) = [lower..a - 1] ++ generator (a + 1, as, upper)
          in  generator (0, dims, shapeRank (Proxy :: Proxy s0) - 1)
    in  f' (unsafeBroadcast (i * g) _map / f))
  where g = unsafeReduceMul f dims

differentiableUnsafeReduceAdd :: forall r t s0 s1. (MathOp r t, Num t, T s0 t, T s1 t) => Reverse r s0 t -> [Integer] -> Reverse r s1 t
differentiableUnsafeReduceAdd (Reverse (f, f')) dims = 
  Reverse (unsafeReduceAdd f dims, \ i -> 
    let _map = 
          let generator :: (Integer, [Integer], Integer) -> [Integer]
              generator (lower, []  , upper) = [lower..upper]
              generator (lower, a:as, upper) = [lower..a - 1] ++ generator (a + 1, as, upper)
          in  generator (0, dims, shapeRank (Proxy :: Proxy s0) - 1)
    in  f' $ unsafeBroadcast i _map)

differentiableUnsafeDotGeneral :: forall r t s0 s1 s2. (MathOp r t, T s0 t, T s1 t, T s2 t) => Reverse r s0 t -> Reverse r s1 t -> DotDimensionNumbersAttr -> Reverse r s2 t
differentiableUnsafeDotGeneral (Reverse (f, f')) (Reverse (g, g')) attr = 
  Reverse (unsafeDotGeneral f g attr, \ i -> 
    let lhsShape     = fromInteger <$> shapeVal p0 :: Num i => [i] 
        rhsShape     = fromInteger <$> shapeVal p1 :: Num i => [i]
        batching     = getBatchingDims attr
        contracting  = getContractingDims attr
        batchShape   = map ((lhsShape !!) . fromIntegral . fst) batching -- NOTE: This does not check for dimensional consistency, TODO: add assertion later
        -- the *OtherDims are indices that are neither the batching dimensions nor the contracted dimensions
        lhsOtherDims = gel (0, map fst (batching ++ contracting), fromInteger $ shapeRank p0 - 1)
        rhsOtherDims = gel (0, map snd (batching ++ contracting), fromInteger $ shapeRank p1 - 1)
        -- the *OtherShape is the shape
        lhsOtherShape = map ((lhsShape !!) . fromIntegral) lhsOtherDims
        rhsOtherShape = map ((rhsShape !!) . fromIntegral) rhsOtherDims
        -- `unsafeDotGeneral f g attr` is expected to have the shape batchShape ++ lhsOtherShape ++ rhsOtherShape (see stablehlo specs)
        -- constractShape is like batchShape but for constracting dims. TODO: Add assertion
        contractShape = map ((lhsShape !!) . fromIntegral . fst) contracting
        df :: r s0 t = 
          let -- intermediateShape is the shape of the output from the general dot produce between i and g
              intermediateShape = batchShape ++ lhsOtherShape ++ contractShape
              df' :: forall si. KnownShape si => Proxy si -> r s0 t
              df' _ =  
                let attr' = DotDimensionNumbersAttr {
                      getBatchingDims    = zip [0..] (map snd batching),
                      getContractingDims = zip [fromIntegral $ length batching + length lhsOtherDims..] rhsOtherDims
                    }
                    d :: r si t = unsafeDotGeneral i g attr'
                    transposition = map fst batching ++ lhsOtherDims ++ map fst contracting
                    perm          = map snd $ sortOn fst $ zip transposition [0..]
                in  unsafeTranspose d perm -- unsafeBroadcast d transposition
          in  reifyShape intermediateShape df'
        dg :: r s1 t = 
          let intermediateShape = batchShape ++ contractShape ++ rhsOtherShape
              dg' :: forall si. KnownShape si => Proxy si -> r s1 t 
              dg' _ = 
                let attr' = DotDimensionNumbersAttr {
                      getBatchingDims    = zip (map fst batching) [0..],
                      getContractingDims = zip lhsOtherDims [fromIntegral $ length batching..]
                    }
                    d :: r si t = unsafeDotGeneral f i attr'
                    transposition = map snd batching ++ map snd contracting ++ rhsOtherDims
                    perm          = map snd $ sortOn fst $ zip transposition [0..]
                in  unsafeTranspose d perm -- unsafeBroadcast d transposition
          in  reifyShape intermediateShape dg'
    in  f' df <+> g' dg)
  where gel :: (Num i, Ord i, Enum i) => (i, [i], i) -> [i] -- Generate exclusive list
        gel (start, exclude, end) = 
          let gel' :: (Num i, Ord i, Enum i) => (i, [i], i) -> [i]
              gel' (s, []  , e) = [s..e]
              gel' (s, a:as, e) = [s..a-1] ++ gel' (a+1,as,e)
          in  gel' (start, sort exclude, end)
        p0 :: Proxy s0 = Proxy
        p1 :: Proxy s1 = Proxy

differentiableUnsafeConvolution :: forall s0 s1 s2 r t. (KnownShape s0, KnownShape s1, KnownShape s2, MathOp r t, forall s. KnownShape s => Fractional (r s Float)) => Reverse r s0 t -> Reverse r s1 t -> ConvBatchingDimInfo -> [ConvSpatialDimInfo] -> ConvFeaturesDimInfo -> Reverse r s2 t
differentiableUnsafeConvolution (Reverse (f, f')) (Reverse (g, g')) batching spatial features = Reverse (unsafeConvolution f g batching spatial features, \i -> kernelGradient i <+> inputGradient i)
  where kernelGradient i = 
          let 
          in  undefined
        inputGradient  i =
          let paddedOutputShape = undefined
          in  undefined
        kernelShape = shapeVal (Proxy :: Proxy s1)
        inputShape  = shapeVal (Proxy :: Proxy s0)
        outputShape = shapeVal (Proxy :: Proxy s2)
        rhsDilation = [ (d.kernelDim, d.rhsDilation) | d <- spatial ]
        lhsDilation = [ (d.inputDim , d.lhsDilation) | d <- spatial ]

instance (MathOp r Float, forall s. KnownShape s => Fractional (r s Float)) => MathOp (Reverse r) Float where
  linspace r = Reverse (linspace r, const zero)
  unsafeReduceMul = differentiableUnsafeReduceMul
  unsafeReduceAdd = differentiableUnsafeReduceAdd
  unsafeDotGeneral = differentiableUnsafeDotGeneral
  unsafeConvolution = differentiableUnsafeConvolution

instance {-# OVERLAPPABLE #-} MathOp r t => MathOp (Reverse r) t where
  linspace r = Reverse (linspace r, const zero)
  unsafeReduceMul (Reverse (f, _)) dims = Reverse (unsafeReduceMul f dims, const zero)
  unsafeReduceAdd (Reverse (f, _)) dims = Reverse (unsafeReduceAdd f dims, const zero)
  unsafeDotGeneral (Reverse (f, _)) (Reverse (g, _)) attr = Reverse (unsafeDotGeneral f g attr, const zero)
  unsafeConvolution (Reverse (lhs, _)) (Reverse (rhs, _)) batching spatial features = Reverse (unsafeConvolution lhs rhs batching spatial features, const zero)

-- SelectOp
differentiableBranch :: forall r t s. (SelectOp r t, T s t, Num (r s t)) => Reverse r s t -> Reverse r s t -> Reverse r '[] Pred -> Reverse r s t
differentiableBranch (Reverse (f, f')) (Reverse (t, t')) (Reverse (cond, _)) = 
    Reverse (branch f t cond, \i -> f' (branch i 0 cond) <+> t' (branch 0 i cond))

differentiableSelect :: forall r t s. (SelectOp r t, T s t, Num (r s t)) => Reverse r s t -> Reverse r s t -> Reverse r s Pred -> Reverse r s t
differentiableSelect (Reverse (f, f')) (Reverse (t, t')) (Reverse (cond, _)) = 
    Reverse (select f t cond, \i -> f' (select i 0 cond) <+> t' (select 0 i cond))

instance (SelectOp r Float, forall s. KnownShape s => Num (r s Float)) => SelectOp (Reverse r) Float where
  branch = differentiableBranch
  select = differentiableSelect

instance {-# OVERLAPPABLE #-} SelectOp r t => SelectOp (Reverse r) t where
  branch (Reverse (f, _)) (Reverse (t, _)) (Reverse (cond, _)) =
    Reverse (branch f t cond, const zero)
  select (Reverse (f, _)) (Reverse (t, _)) (Reverse (cond, _)) =
    Reverse (select f t cond, const zero)
-- Comparison
instance EqualOp r t => EqualOp (Reverse r) t where
  isEQ (Reverse (lhs, _)) (Reverse (rhs, _)) = Reverse (isEQ lhs rhs, const zero)
  isNE (Reverse (lhs, _)) (Reverse (rhs, _)) = Reverse (isNE lhs rhs, const zero)
instance OrderOp r t => OrderOp (Reverse r) t where
  isGT (Reverse (lhs, _)) (Reverse (rhs, _)) = Reverse (isGT lhs rhs, const zero)
  isGE (Reverse (lhs, _)) (Reverse (rhs, _)) = Reverse (isGE lhs rhs, const zero)
  isLT (Reverse (lhs, _)) (Reverse (rhs, _)) = Reverse (isLT lhs rhs, const zero)
  isLE (Reverse (lhs, _)) (Reverse (rhs, _)) = Reverse (isLE lhs rhs, const zero)

-- For trace to work
instance (T s t, TraceableElement (r s t)) => TraceableElement (Reverse r s t) where
  constructTracer i = (i', Reverse (t, undefined), tt)
    where (i', t, tt) = constructTracer i
  deconstructTracer = deconstructTracer . primal

-- Reversable
class Cotangent (ReversedType r) => Reversable r where
  type ReversedType r
  constructReverse :: CIntPtr -> ReversedType r -> (CIntPtr, r)
  gradReify :: Proxy r -> CIntPtr -> Gradient -> (CIntPtr, ReversedType r, Gradient)

instance Cotangent (r s t) => Reversable (Reverse r s t) where
  type ReversedType (Reverse r s t) = r s t
  constructReverse i t = (i + 1, Reverse (t, independent i))
  gradReify _ i (Gradient gs) = (i + 1, sum (fromDyn' <$> g), g')
    where (fmap snd -> g, Gradient -> g') = partition ((i ==) . fst) gs

instance (Reversable a, Reversable b) => Reversable (a <&> b) where
  type ReversedType (a <&> b) = ReversedType a <&> ReversedType b
  constructReverse i0 (a :&: b) = (i2, a' :&: b')
    where (i1, a') = constructReverse i0 a
          (i2, b') = constructReverse i1 b
  gradReify _ i0 g0 = (i2, a' :&: b', g2)
    where (i1, a', g1) = gradReify (Proxy :: Proxy a) i0 g0
          (i2, b', g2) = gradReify (Proxy :: Proxy b) i1 g1

-- TODO: Implement General
class ReverseMode f where
  type Rev g f
  type GradResult f
  rgrad' :: (Gradient -> g, CIntPtr) -> f -> Rev g f
  rgradReify :: Annotated CIntPtr f -> Gradient -> GradResult f

instance (Reversable j, Num (r s t)) => ReverseMode (j -> Reverse r s t) where
  type Rev g (j -> Reverse r s t)      = ReversedType j -> g
  type GradResult (j -> Reverse r s t) = ReversedType j
  rgrad' (reifier, i) f t = reifier $ cotangent (f $ snd $ constructReverse i t) 1
  rgradReify (Annotated i) (gradReify (Proxy :: Proxy j) i  -> (_, g, Gradient g')) = assert (null g') g

instance (Reversable j, ReverseMode (a -> b)) => ReverseMode (j -> (a -> b)) where
  type Rev g (j -> (a -> b))      = ReversedType j -> Rev g (a -> b)
  type GradResult (j -> (a -> b)) = ReversedType j <&> GradResult (a -> b)
  rgrad' (reifier, i) f t = rgrad' (reifier, i') (f r)
    where (i', r) = constructReverse i t
  rgradReify (Annotated i) (gradReify (Proxy :: Proxy j) i -> (i', g, g')) = g :&: rgradReify (Annotated i' :: Annotated CIntPtr (a -> b)) g'

type RGrad f = Rev (GradResult f) f
rgrad :: forall f. ReverseMode f => f -> RGrad f
rgrad = rgrad' (rgradReify (Annotated 0 :: Annotated CIntPtr f), 0)
-- TODO: Implement General
