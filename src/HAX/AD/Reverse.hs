{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE QuantifiedConstraints #-}
{-# LANGUAGE ViewPatterns #-}
module HAX.AD.Reverse where
import HAX.Tensor.Tracer
import HAX.Tensor.Tensorial

import HAX.AD.Gradient

import Data.Proxy
import Data.List
import Data.Bifunctor

import Stablehlo.Dialect.Stablehlo.Attributes
import MLIR

newtype Reverse r s t = Reverse (r s t, r s t -> Gradient)
primal :: Reverse r s t -> r s t 
primal (Reverse t) = fst t

cotangent :: Reverse r s t -> r s t -> Gradient 
cotangent (Reverse t) = snd t

-- TODO: Restrict this to only continuous types (Float, Double, etc)
--       Discrete types don't have derivatives
instance Num (r s t) => Num (Reverse r s t) where
  (Reverse (f, f')) + (Reverse (g, g')) = 
    Reverse (f + g, \ i -> f' i <+> g' i)

  (Reverse (f, f')) - (Reverse (g, g')) = 
    Reverse (f - g, \ i -> f' i <+> g' (negate i))

  (Reverse (f, f')) * (Reverse (g, g')) = 
    Reverse (f * g, \ i -> f' (i * g) <+> g' (i * f))

  negate (Reverse (f, f')) = 
    Reverse (negate f, f' . negate)

  abs    (Reverse (f, f')) = 
    Reverse (abs f, \ i -> f' (i * signum f))

  signum (Reverse (f, f')) = 
    Reverse (signum f, \ _ -> f' 0 )
  
  fromInteger a = 
    Reverse (fromInteger a, const zero)

instance Fractional (r s t) => Fractional (Reverse r s t) where
  recip (Reverse (f, f')) = 
    Reverse (recip f, \ i -> f' (negate i / (f * f)))

  (Reverse (f, f')) / (Reverse (g, g')) = 
    Reverse (f / g, \ i -> f' (i / g) <+> g' (negate (f / (g * g))))

  fromRational r = 
    Reverse (fromRational r, const zero)


instance (TensorOp r t, Fractional t, forall s. KnownShape s => Fractional (r s t)) => TensorOp (Reverse r) t where
  unsafeReduce = error "unsafeReduce is not directly differentiable, use reduceAdd or reduceSum instead"  

  -- TODO: Find a more elegant way
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


  unsafeReduceAdd :: forall s0 s1. (T s0 t, T s1 t) => Reverse r s0 t -> [Integer] -> Reverse r s1 t
  unsafeReduceAdd (Reverse (f, f')) dims = 
    Reverse (unsafeReduceAdd f dims, \ i -> 
      let _map = 
            let generator :: (Integer, [Integer], Integer) -> [Integer]
                generator (lower, []  , upper) = [lower..upper]
                generator (lower, a:as, upper) = [lower..a - 1] ++ generator (a + 1, as, upper)
            in  generator (0, dims, shapeRank (Proxy :: Proxy s0) - 1)
      in  f' $ unsafeBroadcast i _map)

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

-- instance T s t => Traceable (Reverse Tracer s t) where
--   trace' _ (primal -> u) = (fmap (fmap singleton) . sharing' u, ([], [_type]))
--     where _type = tensorType' (Proxy :: Proxy (Tracer s t))
-- 
-- instance (T s t, Traceable f) => Traceable (Reverse Tracer s t -> f) where 
--   trace' i f = first (_type :) <$> trace' (i + 1) (f argn)
--     where argn = Reverse (Tracer (\ a -> (a, ) <$> blockArg i), undefined)
--           _type = tensorType' (Proxy :: Proxy (Tracer s t))
-- 
