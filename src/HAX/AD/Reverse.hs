{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE QuantifiedConstraints #-}
{-# LANGUAGE ViewPatterns #-}
module HAX.AD.Reverse where
import HAX.Tensor.Tensorial

import HAX.AD.Gradient
import HAX.Utils

import Control.Exception

import Data.Proxy
import Data.List
import Data.Word

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

instance Fractional (r s t) => Fractional (Reverse r s t) where
  recip (Reverse (f, f')) = 
    Reverse (r, \i -> f' (negate i * (r * r)))
    where r = recip f
  (Reverse (f, f')) / (Reverse (g, g')) = 
    Reverse (f / g, \i -> f' (i / g) <+> g' (negate (i * f / (g * g))))

  fromRational r = 
    Reverse (fromRational r, const zero)

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
  

-- TODO: Tries to implement in differentiable types such as int
instance (ShapeOp r t, MathOp r t, Num t) => ShapeOp (Reverse r) t where
  unsafeBroadcast :: forall s0 s1. (KnownShape s0, KnownShape s1) => Reverse r s0 t -> [Integer] -> Reverse r s1 t
  unsafeBroadcast (Reverse (f, f')) _map = 
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

  splat i = Reverse (splat i, const zero)

  unsafeTranspose :: forall s0 s1. (KnownShape s0, KnownShape s1) => Reverse r s0 t -> [Integer] -> Reverse r s1 t
  unsafeTranspose (Reverse (f, f')) perm = 
    Reverse (unsafeTranspose f perm, \ i -> 
      let perm' = map snd $ sortOn fst $ zip perm [0..] 
      in  f' (unsafeTranspose i perm'))

instance (MathOp r t, forall s. KnownShape s => Fractional (r s t), Num t) => MathOp (Reverse r) t where
  unsafeReduceMul :: forall s0 s1. (T s0 t, T s1 t) => Reverse r s0 t -> [Integer] -> Reverse r s1 t
  unsafeReduceMul (Reverse (f, f')) dims = 
    Reverse (g, \ i -> 
      let _map = 
            let generator :: (Integer, [Integer], Integer) -> [Integer]
                generator (lower, []  , upper) = [lower..upper]
                generator (lower, a:as, upper) = [lower..a - 1] ++ generator (a + 1, as, upper)
            in  generator (0, dims, shapeRank (Proxy :: Proxy s0) - 1)
      in  f' (unsafeBroadcast (i * g) _map / f))
    where g = unsafeReduceMul f dims

  unsafeReduceAdd :: forall s0 s1. (T s0 t, T s1 t) => Reverse r s0 t -> [Integer] -> Reverse r s1 t
  unsafeReduceAdd (Reverse (f, f')) dims = 
    Reverse (unsafeReduceAdd f dims, \ i -> 
      let _map = 
            let generator :: (Integer, [Integer], Integer) -> [Integer]
                generator (lower, []  , upper) = [lower..upper]
                generator (lower, a:as, upper) = [lower..a - 1] ++ generator (a + 1, as, upper)
            in  generator (0, dims, shapeRank (Proxy :: Proxy s0) - 1)
      in  f' $ unsafeBroadcast i _map)

  unsafeDotGeneral :: forall s0 s1 s2. (KnownShape s0, KnownShape s1, KnownShape s2) => Reverse r s0 t -> Reverse r s1 t -> DotDimensionNumbersAttr -> Reverse r s2 t
  unsafeDotGeneral (Reverse (f, f')) (Reverse (g, g')) attr = 
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

  linspace r = Reverse (linspace r, const zero)

instance (SelectOp r t, forall s. KnownShape s => Num (r s t)) => SelectOp (Reverse r) t where
  branch :: forall s. KnownShape s => Reverse r s t -> Reverse r s t -> Reverse r '[] Word8 -> Reverse r s t
  branch (Reverse (f, f')) (Reverse (t, t')) (Reverse (cond, _)) = 
    Reverse (branch f t cond, \i -> f' (branch i 0 cond) <+> t' (branch 0 i cond))

  select :: forall s. KnownShape s => Reverse r s t -> Reverse r s t -> Reverse r s Word8 -> Reverse r s t
  select (Reverse (f, f')) (Reverse (t, t')) (Reverse (cond, _)) = 
    Reverse (select f t cond, \i -> f' (select i 0 cond) <+> t' (select 0 i cond))

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
