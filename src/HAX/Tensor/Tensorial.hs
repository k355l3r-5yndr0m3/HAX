{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE QuantifiedConstraints #-}
{-# LANGUAGE GeneralisedNewtypeDeriving #-}
{-# LANGUAGE NoStarIsType #-}
{-# LANGUAGE TypeFamilyDependencies #-}
{-# LANGUAGE DefaultSignatures #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
module HAX.Tensor.Tensorial where
import Prelude hiding (pred)
import HAX.PjRt.BufferType

import HAX.AD.Gradient

import Control.Monad
import Control.Exception (assert)

import Data.Kind 
import Data.Dynamic
import Data.Proxy
import Data.Reflection hiding (Z)
import Data.Primitive
import Data.Int
import Data.List hiding (transpose)
import Data.Bifunctor
import Data.Maybe (fromJust)
import Data.Coerce (Coercible, coerce)

import Foreign hiding (sizeOf, rotate)
import Foreign.C (CIntPtr)

import GHC.TypeLits
import GHC.Real (infinity)

import MLIR
import qualified Stablehlo.Dialect.Stablehlo as SHLO

import Stablehlo.Dialect.Stablehlo.Attributes
import System.Random.Stateful

-- Shape
type N n = KnownNat n
type Shape = [Nat]
class Typeable s => KnownShape (s :: Shape) where
  shapeVal :: Proxy s -> [Integer]
  shapeRank :: Proxy s -> Integer
  shapeValHead :: Proxy s -> Integer
  
instance KnownShape '[] where
  shapeVal _ = []
  shapeRank _ = 0
  shapeValHead _ = undefined

instance (KnownNat a, KnownShape as) => KnownShape (a ': as) where
  shapeVal _ = natVal (Proxy :: Proxy a) : shapeVal (Proxy :: Proxy as)
  shapeRank _ = 1 + shapeRank (Proxy :: Proxy as)
  shapeValHead _ = natVal (Proxy :: Proxy a)

shapeVal' :: forall s. KnownShape s => [Integer]
shapeVal' = shapeVal (Proxy :: Proxy s)

shapeRank' :: forall s. KnownShape s => Integer
shapeRank' = shapeRank (Proxy :: Proxy s)
reifyShape :: forall r. [Integer] -> (forall (s :: Shape). KnownShape s => Proxy s -> r) -> r
reifyShape []     f = f (Proxy :: Proxy '[])
reifyShape (a:as) f = reifyNat a (\ p -> reifyShape as (f . k p))
  where k :: Proxy p -> Proxy ps -> Proxy (p ': ps)
        k _ _ = Proxy

type family ReverseListImpl (retro :: [a]) (pro :: [a]) :: [a] where
  ReverseListImpl r '[] = r
  ReverseListImpl r (a ': as) = ReverseListImpl (a ': r) as
type ReverseList l = ReverseListImpl '[] l

type family (x :: [a]) ++ (y :: [a]) :: [a] where
  '[] ++ b = b
  a ++ '[] = a
  (a ': as) ++ b = a ': (as ++ b)

type family IsPrefixOf (prefix :: [a]) (string :: [a]) (f :: Constraint) :: Constraint where
  IsPrefixOf '[] _ _ = ()
  IsPrefixOf (a ': as) (a ': bs) f = IsPrefixOf as bs f
  IsPrefixOf _ _ f = f
type IsSuffixOf suffix string f = IsPrefixOf (ReverseList suffix) (ReverseList string) f

type family ShapeNatAt (s :: [a]) (i :: Nat) :: a where
  ShapeNatAt (a ': as) 0 = a
  ShapeNatAt (a ': as) i = ShapeNatAt as (i - 1)
  ShapeNatAt '[]       _ = TypeError (Text "Indexing out of bound" :$$: 
                                      Text "Tries smaller number")
type family UniqueImpl1 (a :: a') (as :: [a']) (f :: Constraint) :: Constraint where
  UniqueImpl1 a '[] f = ()
  UniqueImpl1 a (a ': as) f = f
  UniqueImpl1 a (b ': as) f = UniqueImpl1 a as f
type family UniqueImpl0 (a :: [a']) (f :: Constraint) :: Constraint where
  UniqueImpl0 '[] f = ()
  UniqueImpl0 (a ': as) f = (UniqueImpl1 a as f, UniqueImpl0 as f)
type Unique a = UniqueImpl0 a (TypeError (Text "Elements of " :<>: ShowType a :<>: Text " are not unique"))

type family BroadcastConsistentContraint (org :: Shape) (map :: Shape) (targ :: Shape) :: Constraint where
  BroadcastConsistentContraint '[] '[] _ = ()
  BroadcastConsistentContraint (o ': os) (m ': ms) targ = (o ~ ShapeNatAt targ m, BroadcastConsistentContraint os ms targ)
  BroadcastConsistentContraint _ _ _ = TypeError (Text "Given map not of the correct size")

type Broadcast org map targ = (BroadcastConsistentContraint org map targ, Unique map, KnownShape map, KnownShape org, KnownShape targ)
type Broadcast' org targ = (KnownShape org, KnownShape targ, IsSuffixOf org targ (TypeError (ShowType org :<>: Text " is not a suffix of " :<>: ShowType targ :$$: Text "Maybe use broadcast")))

type family TensorProduct (lhs :: Shape) (rhs :: Shape) :: Shape where
  TensorProduct '[] rhs = rhs
  TensorProduct (a ': as) rhs = a ': TensorProduct as rhs
type TensorProductConstraint l r p = (KnownShape l, KnownShape r, p ~ TensorProduct l r, KnownShape p)

type family ReplaceAtIdx (l :: [a]) (i :: Nat) (r :: a) :: [a] where
  ReplaceAtIdx '[]       _ _ = TypeError (Text "Index out of bound")
  ReplaceAtIdx (a ': as) 0 r = r ': as
  ReplaceAtIdx (a ': as) i r = a ': ReplaceAtIdx as (i - 1) r

type family ToMaybeList (l :: [a]) :: [Maybe a] where
  ToMaybeList '[]       = '[]
  ToMaybeList (a ': as) = 'Just a ': ToMaybeList as
type family FromMaybeList (l :: [Maybe a]) :: [a] where
  FromMaybeList '[]              = '[]
  FromMaybeList ('Just a ': as)  = a ': FromMaybeList as
  FromMaybeList ('Nothing ': as) = FromMaybeList as
type family ReduceImpl (l :: [Maybe a]) (r :: Shape) :: [Maybe a] where
  ReduceImpl l '[]       = l
  ReduceImpl l (a ': as) = ReduceImpl (ReplaceAtIdx l a 'Nothing) as
type Reduce l r = FromMaybeList (ReduceImpl (ToMaybeList l) r)

type family SameLength (lhs :: [a]) (rhs :: [a]) (e :: Constraint) :: Constraint where
  SameLength '[]       '[]       _ = ()
  SameLength (l ': ls) (r ': rs) e = SameLength ls rs e
  SameLength _         _         e = e

type family TransposeImpl (operand :: Shape) (perm :: Shape) :: Shape where
  TransposeImpl operand '[]       = '[]
  TransposeImpl operand (a ': as) = ShapeNatAt operand a ': TransposeImpl operand as

type Transpose operand perm result = (SameLength operand perm (TypeError (ShowType operand :<>: Text " must be the same length as " :<>: ShowType perm)), 
                                      result ~ TransposeImpl operand perm, KnownShape perm, KnownShape operand, KnownShape result)
type family Product (a :: Shape) :: Natural where
  Product '[]       = 1
  Product (a ': as) = a * Product as
type Reshapable operand result = (Product operand ~ Product result)

type family Head (list :: [a]) :: a where
  Head (a ': _) = a
  Head _        = TypeError (Text "Given list is not long enough")

type family Tail (list :: [a]) :: [a] where
  Tail '[]      = TypeError (Text "Given list is not long enough")
  Tail (_ ': a) = a

type family Init (list :: [a]) :: [a] where
  Init '[]       = TypeError (Text "Given list is not long enough")
  Init '[a]      = '[]
  Init (a ': as) = a ': Init as

type family Last (list :: [a]) :: a where
  Last '[]       = TypeError (Text "Given list is not long enough")
  Last '[a]      = a
  Last (a ': as) = Last as

type Middle l = Init (Tail l)

type family Iota (dim :: Nat) (shape :: Shape) :: Constraint where 
  Iota _ '[] = TypeError (Text "Tensor rank is less then required by iota dim.")
  Iota 0 _   = ()
  Iota n (_ ': as) = Iota (n - 1) as

type ConcatConstraint (a :: Constraint) (b :: Constraint) = (a, b) :: Constraint
type family ConvolutionShapeConstraint (input :: Shape) (kernel :: Shape) (output :: Shape) :: Constraint where
  ConvolutionShapeConstraint '[] '[] '[] = ()
  ConvolutionShapeConstraint (i ': is) (k ': ks) (o ': os) = (i + 1 - k ~ o, o + k - 1 ~ i, i + 1 - o ~ k, ConvolutionShapeConstraint is ks os)
  ConvolutionShapeConstraint _ _ _ = TypeError (Text "Spatial rank in convolution is not the same")

type Convolution (input :: Shape) (kernel :: Shape) (output :: Shape) = (KnownShape input, KnownShape kernel, KnownShape output, Head input ~ Head output, Last input ~ Head kernel, Last kernel ~ Last output, ConvolutionShapeConstraint (Middle input) (Middle kernel) (Middle output))
type Convolution' (input :: Shape) (kernel :: Shape) (output :: Shape) = (KnownShape input, KnownShape kernel, KnownShape output, Last input ~ Head kernel, Last kernel ~ Last output, ConvolutionShapeConstraint (Init input) (Middle kernel) (Init output))

shapeOf :: forall r s t. KnownShape s => r s t -> [Integer]
shapeOf _ = shapeVal (Proxy :: Proxy s)

data Biproxy a b = Biproxy

-- Tensorial
type Z = Shape -> Type -> Type
tensorTypeOf :: forall r s t. T s t => r s t -> AnyType
tensorTypeOf _ = tensorTypeOf' (fromIntegral <$> shapeVal (Proxy :: Proxy s)) (Proxy :: Proxy t)

tentype  :: forall s t. T s t => AnyType
tentype = tensorTypeOf' (fromIntegral <$> shapeVal' @s) (Proxy :: Proxy t)

tentype' :: forall t. Tensorial t => [Int64] -> AnyType
tentype' = (`tensorTypeOf'` (Proxy :: Proxy t))

scaltype :: forall t. Tensorial t => AnyType 
scaltype = tensorTypeOf' [] (Proxy :: Proxy t)

class (Prim (StorageType t), Storable (StorageType t), Typeable t) => Tensorial t where
  type StorageType t = r | r -> t

  pjrtBufferType :: Proxy t -> BufferType
  tensorTypeOf' :: [Int64] -> Proxy t -> AnyType

  staticSizeOf   :: Proxy t -> Int

  splatConstant :: [Int64] ->  t  -> BlockM Value
  elemsConstant :: [Int64] -> [t] -> BlockM Value

  fromHaskell :: t -> StorageType t
  toHaskell   :: StorageType t -> t
  literalPad  :: t
  comparisonType :: Proxy t -> ComparisonTypeAttr
  
  independent :: (TensorOp r, T s t) => CIntPtr -> r s t -> Gradient
  default independent :: (KnownShape s, Typeable r, Fractional t) => CIntPtr -> r s t -> Gradient
  independent idx val = Gradient [(idx, toDyn val)]

  gradientSum :: (TensorOp r, T s t) => [Dynamic] -> r s t
  default gradientSum :: (TensorOp r, T s t, Fractional t) => [Dynamic] -> r s t
  gradientSum [] = splat 0 
  gradientSum gs = foldl1 unsafePairwiseAdd [fromDyn i $ error "Gradient ID scrambled!" | i <- gs]

  showTensorial :: StorageType t -> String
  default showTensorial :: Show (StorageType t) => StorageType t -> String
  showTensorial = show

  -- The lowest possible value (if this can be sorted)
  -- max maxIdent a = a and max a maxIdent = a
  maxIdent :: Ord t => t
  default maxIdent :: (Ord t, Bounded t) => t 
  maxIdent = minBound

  updateParameter :: (TensorOp r, KnownShape s) => Double -> r s t -> r s t -> r s t
  default updateParameter :: (TensorOp r, KnownShape s, Fractional t) => Double -> r s t -> r s t -> r s t
  updateParameter (splat . realToFrac -> stepsize) initial gradient = initial `unsafePairwiseSub` delta
    where delta = gradient `unsafePairwiseMul` stepsize

  tensorialUniformRM :: StatefulGen g m => Int -> (t, t) -> g -> m (IO (Ptr (StorageType t))) 
  default tensorialUniformRM :: (StatefulGen g m, UniformRange (StorageType t)) => Int -> (t, t) -> g -> m (IO (Ptr (StorageType t))) 
  tensorialUniformRM n (bimap fromHaskell fromHaskell -> range) key = do 
    entropy <- replicateM n $ uniformRM range key
    return $ do 
      buffer <- mallocArray n
      pokeArray buffer entropy 
      return buffer
  tensorialUniformM :: StatefulGen g m => Int -> g -> m (IO (Ptr (StorageType t)))
  default tensorialUniformM :: (StatefulGen g m, UniformRange (StorageType t), Num t) => Int -> g -> m (IO (Ptr (StorageType t))) 
  tensorialUniformM = (`tensorialUniformRM` (-1, 1))

  -- For gradient
  -- TODO: Remove unneeded gradient function (ie those that does not needed additional constraint)
  unsafeBroadcastGrad :: (TensorOp r, T s0 t, T s1 t) => G r s0 t -> [Integer] -> r s1 t -> Gradient
  default unsafeBroadcastGrad :: forall r s0 s1. (TensorOp r, T s0 t, T s1 t, Fractional t) => G r s0 t -> [Integer] -> r s1 t -> Gradient
  --unsafeBroadcastGrad _ _ = nograd
  unsafeBroadcastGrad f' dims i =
    reifyShape reduceResult $ \(same (unsafeReduceAdd i reduceDims) -> reduced) -> 
      f' $ unsafeTranspose reduced perm
    where same :: KnownShape s => r s t -> Proxy s -> r s t
          same = const 
          reduceDims   = [0..r1 - 1] \\ dims
          reduceResult = (s1 !!) . fromInteger <$> dims 
          perm = 
            let sorted = sort dims
                assocs = zip sorted [0..]
                permed = fmap (`lookup` assocs) dims
            in  fromJust <$> permed
          r1 = shapeRank (Proxy :: Proxy s1)
          s1 = shapeVal  (Proxy :: Proxy s1)

  unsafeSliceGrad      :: (TensorOp r, T s0 t, T s1 t) => G r s0 t -> [(Integer, Integer, Integer)] -> r s1 t -> Gradient
  default unsafeSliceGrad :: forall s0 s1 r. (TensorOp r, T s0 t, T s1 t, Fractional t) => G r s0 t -> [(Integer, Integer, Integer)] -> r s1 t -> Gradient
  unsafeSliceGrad f' slicing i = f' $ unsafePad 0 i (zip3 starts higher interior)
    where (starts, _, strides) = unzip3 slicing
          interior = fmap (+(-1)) strides
          higher   = zipWith4 (\low axis st tot -> tot - low - (axis - 1) * st - 1) starts (shapeVal (Proxy :: Proxy s1)) strides (shapeVal (Proxy :: Proxy s0))


  unsafeScatterGrad :: (TensorOp r, T s0 t, T s1 t, T s2 t) => G r s0 t -> r s1 Int64 -> G r s2 t -> [Integer] -> [Integer] -> [Integer] -> Integer -> r s0 t -> Gradient
  default unsafeScatterGrad :: forall r s0 s1 s2. (TensorOp r, T s0 t, T s1 t, T s2 t, Fractional t) => G r s0 t -> r s1 Int64 -> G r s2 t -> [Integer] -> [Integer] -> [Integer] -> Integer -> r s0 t -> Gradient
  unsafeScatterGrad f' g h' updateWindowAxes insertWindowAxes sdtod indexVecAxis i = lhs <+> rhs
    where lhs = f' $ unsafeScatter i g (splat 0 :: r s2 t) updateWindowAxes insertWindowAxes sdtod indexVecAxis 
          rhs = h' $ unsafeGather  i g updateWindowAxes insertWindowAxes sdtod indexVecAxis sliceSizes
          sliceSizes  = 
            let generator :: [Integer] -> [Integer] -> [Integer]
                generator a []     = a
                generator a (j:js) = generator (let k = fromInteger j in take k a ++ 1 : drop k a) js
            in  generator bounds $ sort insertWindowAxes
          resultShape = shapeVal (Proxy :: Proxy s2)
          bounds      = (resultShape !!) . fromInteger <$> updateWindowAxes

  unsafeGatherGrad     :: (TensorOp r, T s0 t, T s1 t, T s2 t) => G r s0 t -> r s1 Int64 -> [Integer] -> [Integer] -> [Integer] -> Integer -> r s2 t -> Gradient
  default unsafeGatherGrad :: (TensorOp r, T s0 t, T s1 t, T s2 t, Fractional t) => G r s0 t -> r s1 Int64 -> [Integer] -> [Integer] -> [Integer] -> Integer -> r s2 t -> Gradient
  unsafeGatherGrad f' g offsetAxes collapsedAxes startAxisMap idxVectorAxis i = 
    f' $ unsafeScatter (splat 0) g i offsetAxes collapsedAxes startAxisMap idxVectorAxis

  unsafeDotGeneralGrad :: (TensorOp r, KnownShape s0, KnownShape s1, KnownShape s2) => r s0 t -> G r s0 t -> r s1 t -> G r s1 t -> DotDimensionNumbersAttr -> G r s2 t
  default unsafeDotGeneralGrad :: forall s0 s1 s2 r. (TensorOp r, KnownShape s0, KnownShape s1, KnownShape s2, Fractional t) => r s0 t -> G r s0 t -> r s1 t -> G r s1 t -> DotDimensionNumbersAttr -> G r s2 t
  unsafeDotGeneralGrad f f' g g' attr i =
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
      in  f' df <+> g' dg
    where gel :: (Num i, Ord i, Enum i) => (i, [i], i) -> [i] -- Generate exclusive list
          gel (start, exclude, end) = 
            let gel' :: (Num i, Ord i, Enum i) => (i, [i], i) -> [i]
                gel' (s, []  , e) = [s..e]
                gel' (s, a:as, e) = [s..a-1] ++ gel' (a+1,as,e)
            in  gel' (start, sort exclude, end)
          p0 :: Proxy s0 = Proxy
          p1 :: Proxy s1 = Proxy

  -- This is work regardless of t, maybe remove this 
  unsafeReduceAddGrad  :: (TensorOp r, KnownShape s0, KnownShape s1) => G r s0 t -> [Integer] -> G r s1 t
  default unsafeReduceAddGrad :: forall s0 s1 r. (TensorOp r, KnownShape s0, KnownShape s1, Fractional t) => G r s0 t -> [Integer] -> G r s1 t
  unsafeReduceAddGrad f' dims i = f' $ unsafeBroadcast i _map
    where _map = 
            let generator :: (Integer, [Integer], Integer) -> [Integer]
                generator (lower, []  , upper) = [lower..upper]
                generator (lower, a:as, upper) = [lower..a - 1] ++ generator (a + 1, as, upper)
            in  generator (0, dims, shapeRank (Proxy :: Proxy s0) - 1)

  unsafeReduceMulGrad  :: (TensorOp r, KnownShape s0, KnownShape s1) => r s0 t -> G r s0 t -> [Integer] -> G r s1 t
  default unsafeReduceMulGrad :: forall s0 s1 r. (TensorOp r, KnownShape s0, KnownShape s1, Fractional t) => r s0 t -> G r s0 t -> [Integer] -> G r s1 t
  unsafeReduceMulGrad f f' dims i =
      let _map = let generator :: (Integer, [Integer], Integer) -> [Integer]
                     generator (lower, []  , upper) = [lower..upper]
                     generator (lower, a:as, upper) = [lower..a - 1] ++ generator (a + 1, as, upper)
                 in  generator (0, dims, shapeRank (Proxy :: Proxy s0) - 1)
      in  f' (unsafeBroadcast (i `unsafePairwiseMul` g) _map `unsafePairwiseDiv` f)
    where g = unsafeReduceMul f dims

  unsafeConvolutionGrad :: (TensorOp r, KnownShape s0, KnownShape s1, KnownShape s2) => r s0 t -> G r s0 t -> r s1 t -> G r s1 t -> G r s2 t
  default unsafeConvolutionGrad :: forall s0 s1 s2 r. (TensorOp r, KnownShape s0, KnownShape s1, KnownShape s2, Fractional t) => r s0 t -> G r s0 t -> r s1 t -> G r s1 t -> G r s2 t
  unsafeConvolutionGrad f f' g g' i = f' inputGradient <+> g' kernelGradient
    where inputGradient :: r s0 t
          inputGradient = 
            let result :: forall rotkern padshape. (KnownShape rotkern, KnownShape padshape) => Proxy rotkern -> Proxy padshape -> r s0 t
                result _ _ = 
                  let rkernel :: r rotkern t = unsafeTranspose (unsafeReverse g spatials) (rotate dims)
                      expad = fmap (+(-1)) kerShape
                      -- inpad = fmap (+(-1)) (middle outShape) -- the inpad should be zero, why did I do this?
                      padder = (0, 0, 0):fmap (\a -> (a, a, 0)) expad ++ [(0, 0, 0)]
                      padded :: r padshape t = unsafePad 0 i padder
                  in  unsafeConvolution padded rkernel
                padShape = batchSize:zipWith (\a b -> (a - 1) * 2 + b) kerShape (middle outShape) ++ [outChann]
            in  reifyShape padShape $ reifyShape (rotate rhsShape) result
          kernelGradient :: r s1 t
          kernelGradient =
            let result :: forall rotinput. (KnownShape rotinput) => Proxy rotinput -> r s1 t
                result _ =
                  let rotinput :: r rotinput t = unsafeTranspose f (rotate dims)
                  in  unsafeConvolution rotinput i
            in  reifyShape (rotate lhsShape) result
          batchSize = assert (head lhsShape == head outShape) $ head lhsShape
          outChann  = assert (last rhsShape == last outShape) $ last outShape
          lhsShape = shapeVal (Proxy :: Proxy s0)
          rhsShape = shapeVal (Proxy :: Proxy s1)
          outShape = shapeVal (Proxy :: Proxy s2)
          kerShape = middle rhsShape
          rotate s = last s:middle s ++ [head s]
          middle s = init $ tail s
          spatials = [1..fromIntegral $ length rhsShape - 2]
          dims     = [0..fromIntegral $ length rhsShape - 1]


  -- TODO: Remove this
  branchGrad :: (TensorOp r, KnownShape s) => G r s t -> G r s t -> r '[] Bool -> G r s t
  default branchGrad :: (TensorOp r, KnownShape s, Fractional t) => G r s t -> G r s t -> r '[] Bool -> G r s t
  branchGrad f' t' c i = f' (branch i (splat 0) c) <+> t' (branch (splat 0) i c)

  selectGrad :: (TensorOp r, KnownShape s) => G r s t -> G r s t -> r s Bool -> G r s t
  default selectGrad :: (TensorOp r, KnownShape s, Fractional t) => G r s t -> G r s t -> r s Bool -> G r s t
  selectGrad f' t' c i = f' (select i (splat 0) c) <+> t' (select (splat 0) i c)

instance Tensorial Float where
  type StorageType Float = Float

  pjrtBufferType _ = f32
  tensorTypeOf' shape _ = toAnyType (RankedTensorType shape F32Type NullAttr)
  staticSizeOf   _ = sizeOf (0 :: Float)
  
  splatConstant shape value = SHLO._ConstantOp attr $ toAnyType _type
    where attr  = DenseIntOrFPElements _type value
          _type = RankedTensorType shape F32Type NullAttr
  elemsConstant shape value = SHLO._ConstantOp attr $ toAnyType _type
    where attr  = DenseIntOrFPElements _type value
          _type = RankedTensorType shape F32Type NullAttr

  fromHaskell = id
  toHaskell   = id
  
  literalPad = 0
  comparisonType _ = ComparisonTypeFloat
  tensorialUniformM = (`tensorialUniformRM` (-0.2, 0.2))

  maxIdent = -fromRational infinity
  
instance Tensorial Word8 where
  type StorageType Word8 = Word8

  pjrtBufferType _ = u8
  tensorTypeOf' shape _ = toAnyType (RankedTensorType shape UI8 NullAttr)
  staticSizeOf   _ = sizeOf (0 :: Word8)

  splatConstant shape value = SHLO._ConstantOp attr $ toAnyType _type
    where attr  = DenseIntOrFPElements _type value
          _type = RankedTensorType shape UI8 NullAttr
  elemsConstant shape value = SHLO._ConstantOp attr $ toAnyType _type
    where attr  = DenseIntOrFPElements _type value
          _type = RankedTensorType shape UI8 NullAttr

  fromHaskell = id
  toHaskell   = id

  literalPad = 0
  comparisonType _ = ComparisonTypeUnsigned
  updateParameter _ i _ = i

  unsafeBroadcastGrad _ _ = nograd
  unsafeSliceGrad _ _ = nograd
  unsafeGatherGrad _ _ _ _ _ _ = nograd
  unsafeScatterGrad _ _ _ _ _ _ _ = nograd
  unsafeDotGeneralGrad _ _ _ _ _ = nograd
  unsafeConvolutionGrad _ _ _ _ = nograd
  
  unsafeReduceAddGrad _ _ = nograd
  unsafeReduceMulGrad _ _ _ = nograd

  branchGrad _ _ _ = nograd
  selectGrad _ _ _ = nograd

  independent _ = nograd
  gradientSum _ = splat 0

instance Tensorial Int64 where
  type StorageType Int64 = Int64

  pjrtBufferType _ = s64
  tensorTypeOf' shape _ = toAnyType (RankedTensorType shape I64 NullAttr)
  staticSizeOf   _ = sizeOf (0 :: Int64)

  splatConstant shape value = SHLO._ConstantOp attr $ toAnyType _type
    where attr  = DenseIntOrFPElements _type value
          _type = RankedTensorType shape I64 NullAttr
  elemsConstant shape value = SHLO._ConstantOp attr $ toAnyType _type
    where attr  = DenseIntOrFPElements _type value
          _type = RankedTensorType shape I64 NullAttr

  fromHaskell = id
  toHaskell   = id

  literalPad = 0
  comparisonType _ = ComparisonTypeSigned
  updateParameter _ i _ = i

  unsafeBroadcastGrad _ _ = nograd
  unsafeSliceGrad _ _ = nograd
  unsafeGatherGrad _ _ _ _ _ _ = nograd
  unsafeScatterGrad _ _ _ _ _ _ _ = nograd
  unsafeDotGeneralGrad _ _ _ _ _ = nograd
  unsafeConvolutionGrad _ _ _ _ = nograd

  unsafeReduceAddGrad _ _ = nograd
  unsafeReduceMulGrad _ _ _ = nograd

  branchGrad _ _ _ = nograd
  selectGrad _ _ _ = nograd

  independent _ = nograd
  gradientSum _ = splat 0


newtype Pred = Pred { unPred :: Word8 } deriving (Num, Eq, Prim, Storable)
instance Show Pred where 
  show (Pred 0) = "False"
  show _        = "True "

instance Tensorial Bool where
  type StorageType Bool = Pred

  pjrtBufferType _ = pred
  tensorTypeOf' shape _ = toAnyType (RankedTensorType shape (I 1) NullAttr)
  staticSizeOf   _ = sizeOf (0 :: Word8)

  splatConstant shape value = SHLO._ConstantOp attr $ toAnyType _type
    where attr  = DenseIntOrFPElements _type value
          _type = RankedTensorType shape (I 1) NullAttr
  elemsConstant shape value = SHLO._ConstantOp attr $ toAnyType _type
    where attr  = DenseIntOrFPElements _type value
          _type = RankedTensorType shape (I 1) NullAttr

  fromHaskell i = if i then 1 else 0
  toHaskell (Pred i) = i > 0

  literalPad = False
  comparisonType _ = ComparisonTypeUnsigned
  updateParameter _ i _ = i
  tensorialUniformRM n (bimap (unPred . fromHaskell) (unPred . fromHaskell) -> range) key = do 
    (fmap Pred -> entropy) <- replicateM n $ uniformRM range key
    return $ do 
      buffer <- mallocArray n
      pokeArray buffer entropy 
      return buffer
  tensorialUniformM = (`tensorialUniformRM` (False, True))

  unsafeBroadcastGrad _ _ = nograd
  unsafeSliceGrad _ _ = nograd
  unsafeGatherGrad _ _ _ _ _ _ = nograd
  unsafeScatterGrad _ _ _ _ _ _ _ = nograd
  unsafeDotGeneralGrad _ _ _ _ _ = nograd
  unsafeConvolutionGrad _ _ _ _ = nograd

  unsafeReduceAddGrad _ _ = nograd
  unsafeReduceMulGrad _ _ _ = nograd

  branchGrad _ _ _ = nograd
  selectGrad _ _ _ = nograd

  independent _ = nograd
  gradientSum _ = splat False

type family ListItem a where
  ListItem [a] = a
  ListItem _   = TypeError (Text "Type is not a list")

class KnownShape s => TensorLiteral (s :: Shape) where
  type Literal s t
  fromTensorLiteral :: Proxy s -> q -> (t -> q) -> Literal s t -> [q]

instance TensorLiteral '[] where
  type Literal '[] t = t
  fromTensorLiteral _ _ c a = [c a]

instance (KnownNat i, TensorLiteral is) => TensorLiteral (i ': is) where
  type Literal (i ': is) t = [Literal is t]
  fromTensorLiteral (fromInteger . product . shapeVal -> nelem) p c l = take nelem (concatMap f l ++ repeat p)
    where f = fromTensorLiteral (Proxy :: Proxy is) p c

class ConvertOp (r :: Shape -> Type -> Type) where
  convert :: (T s f, T s g) => r s f -> r s g

type Transformable r = forall s s' t. Coercible (r s t) (r s' t)
type T s t = (KnownShape s, Tensorial t)
forceShape :: [Integer] -> (forall s. KnownShape s => r s t) -> (forall s. KnownShape s => r s t -> a) -> a
forceShape shape value func = reifyShape shape $ \(same value -> value') -> func value'
  where same :: KnownShape s => r s t -> Proxy s -> r s t
        same = const

showShape :: forall r s t. KnownShape s => r s t -> String
showShape _ = show (shapeVal (Proxy :: Proxy s))

-- The lhs (image, or volume, or whatever) is [BATCHING DIM, ...(SPATIAL DIM)..., FEATURE DIM]
--     rhs (kernel)                        is [IN FEAT DIM,  ...(SPATIAL DIM)..., OUT FEAT DIM]
--     output                              is [BATCHING DIM, ...(SPATIAL DIM)..., FEATURE DIM]
-- This is to simplify implementation
class (Transformable r, Typeable r) => TensorOp (r :: Shape -> Type -> Type) where
  assumeEqShape :: forall s s' t. (KnownShape s, KnownShape s') => r s t -> r s' t -- bypass the problem where s should be s', but compiler cannot prove it
  assumeEqShape = assert (shapeVal (Proxy :: Proxy s) == shapeVal (Proxy :: Proxy s')) coerce

  assumeShape   :: r s t -> Proxy s' -> r s' t
  assumeShape = const . coerce

  coerceShape   :: r s t -> r s' t
  coerceShape !a = coerce a

  -- without type checking, internal use only
  -- TODO: Add simple constraints to the dtype, since that does not make other things more difficult
  -- TODO: Add Shapeless Rs
  unsafeBroadcast      :: (T s0 t, T s1 t) => r s0 t -> [Integer] -> r s1 t
  unsafeTranspose      :: (T s0 t, T s1 t) => r s0 t -> [Integer] -> r s1 t
  unsafeReshape        :: (T s0 t, T s1 t) => r s0 t -> r s1 t 
  unsafeSlice          :: (T s0 t, T s1 t) => r s0 t -> [(Integer, Integer, Integer)] -> r s1 t
  unsafePad            :: (T s0 t, T s1 t) => t -> r s0 t -> [(Integer, Integer, Integer)] -> r s1 t
  unsafeReverse        :: (T s0 t) => r s0 t -> [Integer] -> r s0 t
  unsafeScatter        :: (T s0 t, T s1 t, T s2 t) => r s0 t -> r s1 Int64 -> r s2 t -> [Integer] -> [Integer] -> [Integer] -> Integer -> r s0 t
  unsafeGather         :: (T s0 t, T s1 t, T s2 t) => r s0 t -> r s1 Int64 -> [Integer] -> [Integer] -> [Integer] -> Integer -> [Integer] -> r s2 t
  unsafeConcat         :: (T s0 t, T s1 t, T s2 t) => Integer -> r s0 t -> r s1 t -> r s2 t
  unsafeDotGeneral     :: (T s0 t, T s1 t, T s2 t, Num t) => r s0 t -> r s1 t -> DotDimensionNumbersAttr -> r s2 t
  unsafeReduceAdd      :: (T s0 t, T s1 t, Num t) => r s0 t -> [Integer] -> r s1 t
  unsafeReduceMul      :: (T s0 t, T s1 t, Num t) => r s0 t -> [Integer] -> r s1 t
  unsafeConvolution    :: (T s0 t, T s1 t, T s2 t, Num t) => r s0 t -> r s1 t -> r s2 t
  unsafeIota           :: (T s t, Enum t) => Int -> r s t
                       
  unsafePairwiseAdd    :: (T s t, Num t) => r s t -> r s t -> r s t
  unsafePairwiseSub    :: (T s t, Num t) => r s t -> r s t -> r s t
  unsafePairwiseMul    :: (T s t, Num t) => r s t -> r s t -> r s t
  unsafePairwiseDiv    :: (T s t, Fractional t) => r s t -> r s t -> r s t
  
  unsafePairwiseNegate :: (T s t, Num t) => r s t -> r s t
  unsafePairwiseAbs    :: (T s t, Num t) => r s t -> r s t
  unsafePairwiseSignum :: (T s t, Num t) => r s t -> r s t

  unsafePairwiseSin    :: (T s t, Floating t) => r s t -> r s t
  unsafePairwiseCos    :: (T s t, Floating t) => r s t -> r s t
  unsafePairwiseTanh   :: (T s t, Floating t) => r s t -> r s t

  unsafePairwiseExp    :: (T s t, Floating t) => r s t -> r s t
  unsafePairwiseLog    :: (T s t, Floating t) => r s t -> r s t

  isEQ                 :: (T s t, Eq t) => r s t -> r s t -> r s Bool
  isNE                 :: (T s t, Eq t) => r s t -> r s t -> r s Bool
                       
  isGT                 :: (T s t, Ord t) => r s t -> r s t -> r s Bool
  isGE                 :: (T s t, Ord t) => r s t -> r s t -> r s Bool
  isLT                 :: (T s t, Ord t) => r s t -> r s t -> r s Bool
  isLE                 :: (T s t, Ord t) => r s t -> r s t -> r s Bool
                       
  branch               :: (T s t) => r s t -> r s t -> r '[] Bool -> r s t
  select               :: (T s t) => r s t -> r s t -> r s   Bool -> r s t
                       
  splat                :: (T s t) => t -> r s t

  -- TODO: Return both the argmax and max values
  --       Implement argmin
  unsafeArgmax         :: (Ord t, T s t, T s' t) => Int -> r s t -> r s' Int64
  unsafeMultiDimArgmax :: (Ord t, T s t, T s' t) => [Int] -> r s t -> r s' Int64

  -- TODO: Move these to a different class
  unsafeLinspace       :: forall s t. (T s t, Fractional t, Enum t) => Integer -> (t, t) -> r s t
  unsafeLinspace (fromInteger -> axis) (low, high) = unsafeIota axis `unsafePairwiseMul` splat delta
    where delta = (high - low) / (nstep - 1)
          nstep = fromInteger $ shapeVal (Proxy :: Proxy s) !! axis

  unsafeMultiIota      :: forall s t. (T s t, Enum t) => [Integer] -> Integer -> r s t
  unsafeMultiIota []     _ = error "idim needs to be given"
  unsafeMultiIota [fromInteger -> a]    d = assert (shapeVal (Proxy :: Proxy s) !! fromInteger d == 1) unsafeIota a
  unsafeMultiIota ((fromInteger -> a):as) d = forceShape highShape (unsafeMultiIota as d) (forceShape lowShape (unsafeIota a) (unsafeConcat d))
    where changeAt i f ls = let (begin, end) = splitAt i ls in  begin ++ f (head end) : tail end
          shape = shapeVal (Proxy :: Proxy s)
          d' = fromInteger d
          lowShape  = changeAt d' (const 1) shape
          highShape = changeAt d' (+(-1)) shape

  unsafeSplit :: forall s s' t. (T s t, T s' t) => r s t -> [r s' t]
  unsafeSplit operand = 
    if sliceShape /= operandShape then 
      assert (differByOne operandShape sliceShape) [unsafeSlice operand $ slicing i | i <- [0..nstep - 1]]
    else 
      [unsafeReshape operand]
    where operandShape = shapeOf operand
          sliceShape   = shapeVal (Proxy :: Proxy s')
          differByOne (a:as) (b:bs) 
            | a == b    = differByOne as bs 
            | otherwise = as == bs
          differByOne _ _ = False
          differAxis lhs rhs = 
            let differAxis' i (a:as) (b:bs)
                  | a == b    = differAxis' (i + 1) as bs 
                  | otherwise = i
                differAxis' _ _ _ = undefined
            in  differAxis' 0 lhs rhs
          splitAxis = differAxis operandShape sliceShape
          operandDim = operandShape !! splitAxis
          sliceDim   = sliceShape   !! splitAxis
          nstep      = operandDim `div` sliceDim
          rank       = length operandShape
          slicing i  = zipWith (\a o -> if a == splitAxis then (i * sliceDim, (i + 1) * sliceDim, 1) else (0, o, 1)) [0..rank - 1] operandShape

class Struct t r ~ t => Structure t (r :: Z) | t -> r where
  type Struct t (r' :: Z) = k | k -> r'
  change :: t -> (forall s j. T s j => r s j -> r' s j) -> Struct t r'
instance (TensorOp r, T s t) => Structure (r s t) r where
  type Struct (r s t) r' = r' s t
  change t f = f t
instance (Structure a r, Structure b r) => Structure (a, b) r where
  type Struct (a, b) r' = (Struct a r', Struct b r')
  change (a, b) f = (change a f, change b f)

-- For functions derivable from TensorOp
--    the function should require more than one function from TensorOp
class TensorOp r => FusedOp r where
  countTrue :: KnownShape s => r s Bool -> r '[] Int64
  countTrue = reduceAdd' . select 0 1
  

  mean'  :: forall s t. (T s t, Fractional t) => r s t -> r '[] t
  mean' = (/n) . reduceAdd' 
    where n = fromIntegral $ product (shapeVal (Proxy :: Proxy s))

  -- Loss func
  crossEntropy :: forall s t. (TensorOp r, T s t, Floating t) => r s t -> r s t -> r '[] t
  crossEntropy p q = negate . mean' $ (p * log q)

  mse :: (T s t, Fractional t) => r s t -> r s t -> r '[] t
  mse x y = mean' $ d * d
    where d = x - y

  -- activation 
  relu :: (T s t, TensorOp r, Num t, Ord t) => r s t -> r s t
  relu x = select 0 x (x `isGT` 0)
  
  leakyrelu :: (T s t, TensorOp r, Num t, Ord t) => r '[] t -> r s t -> r s t
  leakyrelu alpha = leakyrelu' $ broadcast alpha (Proxy :: Proxy '[])
  
  leakyrelu' :: (T s t, TensorOp r, Num t, Ord t) => r s t -> r s t -> r s t
  leakyrelu' alpha x = x * select alpha 1 (x `isGT` 0)
  
  sigmoid :: (T s t, Floating t) => r s t -> r s t
  sigmoid x = recip (1 + exp (negate x))

  softmax :: forall s t. (T s t, Floating t) => [Int] -> r s t -> r s t 
  softmax axis (unsafePairwiseExp -> operand) = 
    reifyShape (rais axis' shape) $ \(same (unsafeReduceAdd operand $ fromIntegral <$> axis') -> k) -> 
      operand `unsafePairwiseDiv` (unsafeBroadcast k (rais axis' (zipWith const [0..] shape)) + 1e-3)
    where rai i l = take i l ++ drop (i + 1) l
          rais (sort -> y) l = 
            let rais' []     k = k
                rais' (i:is) k = 
                  rais' ((\v -> v - 1) <$> is) (rai i k)
            in  rais' y l
          same :: r s' t -> Proxy s' -> r s' t
          same = const 
          shape = shapeVal (Proxy :: Proxy s)
          rank = length (shapeVal (Proxy :: Proxy s))
          axis' = [i | i <- axis, 0 <= i && i < rank]

  softmax' :: forall s t. (T s t, Floating t) => r s t -> r s t
  softmax' = softmax (take rank [0..])
    where rank = fromIntegral $ shapeRank (Proxy :: Proxy s)

  mha :: forall query head key key' val val' queries keys out t. (Tensorial t, Floating t, N queries, N head, N key', N val', N val, N query, N key, N keys, N out) => r [query, head, key'] t -> r [key, head, key'] t -> r [val, head, val'] t -> r [head, val', out] t -> r [queries, query] t -> r [keys, key] t -> r [keys, val] t -> r [queries, out] t
  mha wq wk wv wo queries keys values = result
    where keys'      :: r [keys, head, key'] t    = unsafeDotGeneral keys      wk      (DotDimensionNumbersAttr { getBatchingDims = [], getContractingDims = [(1, 0)] })
          values'    :: r [keys, head, val'] t    = unsafeDotGeneral values    wv      (DotDimensionNumbersAttr { getBatchingDims = [], getContractingDims = [(1, 0)] })
          queries'   :: r [queries, head, key'] t = unsafeDotGeneral queries   wq      (DotDimensionNumbersAttr { getBatchingDims = [], getContractingDims = [(1, 0)] })
          weights_   :: r [head, queries, keys] t = unsafeDotGeneral queries'  keys'   (DotDimensionNumbersAttr { getBatchingDims = [(1, 1)], getContractingDims = [(2, 2)] })
          weights    :: r [queries, head, keys] t = transpose weights_ (Proxy :: Proxy [1, 0, 2])
          weights'   :: r [queries, head, keys] t = softmax [2] weights / realToFrac (recip $ sqrt $ fromInteger $ natVal (Proxy :: Proxy key') :: Float)
          attention_ :: r [head, queries, val'] t = unsafeDotGeneral weights'  values' (DotDimensionNumbersAttr { getBatchingDims = [(1, 1)], getContractingDims = [(2, 0)] })
          attention  :: r [queries, head, val'] t = transpose attention_ (Proxy :: Proxy [1, 0, 2])
          result     :: r [queries, out] t        = unsafeDotGeneral attention wo      (DotDimensionNumbersAttr { getBatchingDims = [], getContractingDims = [(1, 0), (2, 1)] })

instance {-# OVERLAPPABLE #-} TensorOp r => FusedOp r

-- Reduction
reduceAdd :: (TensorOp r, T s t, T s' t, KnownShape (Reduce s s'), Num t) => r s t -> Proxy s' -> r (Reduce s s') t
reduceAdd operand = unsafeReduceAdd operand . shapeVal

reduceAdd' :: forall r s t. (TensorOp r, T s t, Num t) => r s t -> r '[] t
reduceAdd' = (`unsafeReduceAdd` [0..shapeRank (Proxy :: Proxy s) - 1])

reduceMul :: (TensorOp r, T s t, T s' t, KnownShape (Reduce s s'), Num t) => r s t -> Proxy s' -> r (Reduce s s') t
reduceMul operand = unsafeReduceMul operand . shapeVal

reduceMul' :: forall r s t. (TensorOp r, T s t, Num t) => r s t -> r '[] t
reduceMul' = (`unsafeReduceMul` [0..shapeRank (Proxy :: Proxy s) - 1])

type family Split (lhs :: [a]) (rhs :: [a]) :: Constraint where
  Split (a ': ls) (a ': rs) = Split ls rs
  Split (a ': ls) (b ': rs) = (ls ~ rs)
  Split '[] '[] = ()
  Split _ _ = TypeError (Text "lhs and rhs can only differ by at most one elem")

split :: (TensorOp r, T s t, T s' t, Split s s') => r s t -> [r s' t]
split = unsafeSplit

type family ValidIdx (x :: [a]) (y :: Nat) :: Constraint where
  ValidIdx _   0 = ()
  ValidIdx '[]     _ = (TypeError (Text "Not a valid index"))
  ValidIdx (a':as) i = ValidIdx as (i - 1) 
linspace :: forall axis r s t. (TensorOp r, T s t, Fractional t, Enum t, KnownNat axis, ValidIdx s axis) => Proxy axis -> (t, t) -> r s t
linspace = unsafeLinspace . natVal

onehot :: forall r s c k. (TensorOp r, KnownShape s, KnownNat c, k ~ (s ++ '[c]), c ~ Last k, KnownShape k, s ~ Init k) => r s Int64 -> r k Bool
onehot indices = isEQ c i
  where c :: r k Int64 = unsafeBroadcast indices [0..r - 1] 
        i :: r k Int64 = unsafeIota (fromInteger r)
        r = shapeRank (Proxy :: Proxy s)

(@%) :: forall r t n ns. (KnownNat n, KnownShape ns, Tensorial t, TensorOp r) => r (n ': ns) t -> Integer -> r ns t
operand @% (fromInteger -> index :: r '[] Int64) = unsafeGather operand index [0..resultRank - 1] [0] [0] 0 (1:resultShape)
  where resultRank  = shapeRank (Proxy :: Proxy ns)
        resultShape = shapeVal  (Proxy :: Proxy ns)
infixl 5 @%

unsafeDiagonal :: forall s0 s1 r t. (T s0 t, T s1 t, TensorOp r) => Integer -> Integer -> r s0 t -> r s1 t
unsafeDiagonal keepAxes removeAxes input = assert (inputShape !! fromInteger keepAxes == inputShape !! fromInteger removeAxes) $ 
  reifyShape startShape result
  where inputShape = shapeVal (Proxy :: Proxy s0)
        diagLength = inputShape !! fromInteger keepAxes
        startShape = [diagLength, 2]
        offsetAxes = take (length inputShape - 2) $ [0..keepAxes - 1] ++ [keepAxes + 1..]
        collapsedAxes = [keepAxes, removeAxes]
        startIdxMap   = [keepAxes, removeAxes]
        sliceSize     = 
          let replaceAtWithOne i a = take i a ++ 0: drop (i + 1) a 
          in  replaceAtWithOne (fromInteger keepAxes) $ replaceAtWithOne (fromInteger removeAxes) inputShape
        result :: forall startShape. KnownShape startShape => Proxy startShape -> r s1 t
        result _ = 
          let start :: r startShape Int64 = unsafeIota 0
          in  unsafeGather input start offsetAxes collapsedAxes startIdxMap 1 sliceSize

-- With type checking
broadcast :: (Broadcast org map targ, Tensorial t, TensorOp r) => r org t -> Proxy map -> r targ t
broadcast operand (shapeVal -> _map) = unsafeBroadcast operand _map

broadcast' :: forall r t org targ. (Tensorial t, Broadcast' org targ, TensorOp r) => r org t -> r targ t
broadcast' operand = unsafeBroadcast operand _map
  where targ = shapeVal (Proxy :: Proxy targ)
        org  = shapeVal (Proxy :: Proxy org)
        _map = take (length org) [fromIntegral (length targ - length org)..] 

transpose :: (Tensorial t, Transpose operand perm result, TensorOp r) => r operand t -> Proxy perm -> r result t
transpose operand = unsafeTranspose operand . shapeVal

reshape :: (Tensorial t, KnownShape s0, KnownShape s1, Reshapable s0 s1, TensorOp r) => r s0 t -> r s1 t
reshape = unsafeReshape

iota :: (Iota d s, KnownShape s, KnownNat d, Tensorial t, Enum t, TensorOp r) => Proxy d -> r s t
iota = unsafeIota . fromInteger . natVal

linearMap :: (KnownNat i, KnownNat o, Tensorial t, Num t, TensorOp r) => r '[i, o] t -> r '[i] t -> r '[o] t 
linearMap mat vec = unsafeDotGeneral mat vec dotAttr 
  where dotAttr = DotDimensionNumbersAttr { getBatchingDims = [], getContractingDims = [(0, 0)] }

-- TODO: Implement + - * / etc with automatic broadcasting
prod :: (TensorProductConstraint lhs rhs p, Tensorial t, Num t, TensorOp r) => r lhs t -> r rhs t -> r p t
prod x y = unsafeDotGeneral x y attr
  where attr = DotDimensionNumbersAttr { getContractingDims = [], getBatchingDims = [] }

matmul :: (T '[m, n] t, T '[n, p] t, T '[m, p] t, Num t, TensorOp r) => r '[m, n] t -> r '[n, p] t -> r '[m, p] t
matmul lhs rhs = unsafeDotGeneral lhs rhs attr
  where attr = DotDimensionNumbersAttr { getContractingDims = [(1, 0)], getBatchingDims = [] }

convolution :: (Convolution input kernel output, Num t, Tensorial t, TensorOp r) => r input t -> r kernel t -> r output t 
convolution = unsafeConvolution

convolution' :: forall input kernel output r t. (Tensorial t, Num t, TensorOp r, Convolution' input kernel output) => r input t -> r kernel t -> r output t
convolution' input kernel = unsafeReshape (unsafeConvolution input' kernel :: r (1 ': output) t)
  where input' = unsafeReshape input :: r (1 ': input) t

argmax :: forall axis r s t s'. (TensorOp r, T s t, Ord t, KnownNat axis, s' ~ Argmax s axis, T s' Int64) => r s t -> r s' Int64
argmax = unsafeArgmax $ fromIntegral $ natVal (Proxy :: Proxy axis)

mdargmax :: forall axes r s s' t. (TensorOp r, KnownShape axes, T s t, T s' Int64, Ord t, s' ~ MultiDimArgmax s axes) => r s t -> r s' Int64
mdargmax = unsafeMultiDimArgmax (fromInteger <$> shapeVal' @axes)

(|#|) :: (Tensorial t, TensorProductConstraint l r p, Num t, TensorOp a) => a l t -> a r t -> a p t
(|#|) = prod
infixl 9 |#|

(|@|) :: (Tensorial t, KnownNat n, KnownNat m, KnownNat q, Num t, TensorOp r) => r '[n, m] t -> r '[m, q] t -> r '[n, q] t
(|@|) = matmul
infixl 8 |@|

type family Append (x :: [a]) (y :: a) :: [a] where 
  Append '[]       y = '[y]
  Append (a ': as) y = a ': Append as y

type family Argmax (a :: Shape) (d :: Nat) :: Shape where
  Argmax (a ': as) 0 = as
  Argmax (a ': as) i = a ': Argmax as (i - 1)
  Argmax '[]       _ = TypeError (Text "Reduction axis invalid")

type family Length (l :: [a]) :: Nat where
  Length '[] = 0
  Length (_ ': l) = 1 + Length l

type MultiDimArgmax s r = Append (Reduce s r) (Length r) 

-- Foldind over tensors 
-- Use these functions if memory is limited 
determindSpliting :: [Integer] -> [Integer] -> [[(Integer, Integer, Integer)]]
determindSpliting op sl = 
  let axis = differAxis op sl
      odim = op !! axis
      sdim = sl !! axis
      step = odim `div` sdim
      rank = length op
      slicing i = zipWith (\a o -> if a == axis then (i * sdim, (i + 1) * sdim, 1) else (0, o, 1)) [0..rank - 1] op
  in  map slicing [0..step - 1]
  where differAxis lhs rhs = 
          let differAxis' i (a:as) (b:bs)
                | a == b    = differAxis' (i + 1) as bs 
                | otherwise = if as == bs then i else error "Invalid shapes for spliting"
              differAxis' _ _ _ = error "Invalid shapes for spliting"
          in  differAxis' 0 lhs rhs

foldlSplit :: forall r s s' t a. (TensorOp r, T s t, T s' t, Split s s') => (a -> r s' t -> a) -> a -> r s t -> a
foldlSplit func inital operand = 
  if operandShape == sliceShape then 
    func inital (assumeEqShape operand) 
  else 
    assert (differByOne operandShape sliceShape) (foldl (\accum slice -> func accum (unsafeSlice operand slice)) inital [slicing i | i <- [0..nstep - 1]])
  where operandShape = shapeVal (Proxy :: Proxy s )
        sliceShape   = shapeVal (Proxy :: Proxy s')
        differByOne (a:as) (b:bs) 
          | a == b    = differByOne as bs 
          | otherwise = as == bs
        differByOne _ _ = False
        differAxis lhs rhs = 
          let differAxis' i (a:as) (b:bs)
                | a == b    = differAxis' (i + 1) as bs 
                | otherwise = i
              differAxis' _ _ _ = undefined
          in  differAxis' 0 lhs rhs
        splitAxis = differAxis operandShape sliceShape
        operandDim = operandShape !! splitAxis
        sliceDim   = sliceShape   !! splitAxis
        nstep      = operandDim `div` sliceDim
        rank       = length operandShape
        slicing i  = zipWith (\a o -> if a == splitAxis then (i * sliceDim, (i + 1) * sliceDim, 1) else (0, o, 1)) [0..rank - 1] operandShape

foldlSplit2 :: forall r s0 s1 s0' s1' t0 t1 a. (TensorOp r, T s0 t0, T s0' t0, T s1 t1, T s1' t1, Split s0 s0', Split s1 s1') => (a -> r s0' t0 -> r s1' t1 -> a) -> a -> r s0 t0 -> r s1 t1 -> a
foldlSplit2 func inital lhs rhs = 
  if lhsShape == lhsShape' && rhsShape == rhsShape' then
    func inital (assumeEqShape lhs) (assumeEqShape rhs)
  else
    foldl (\accum (ls, rs) -> 
      let l = unsafeSlice lhs ls
          r = unsafeSlice rhs rs
      in  func accum l r) inital (zip lhs' rhs')
  where lhsShape  = shapeVal (Proxy :: Proxy s0)
        rhsShape  = shapeVal (Proxy :: Proxy s1)
        lhsShape' = shapeVal (Proxy :: Proxy s0') 
        rhsShape' = shapeVal (Proxy :: Proxy s1') 
        lhs' = determindSpliting lhsShape lhsShape'
        rhs' = determindSpliting rhsShape rhsShape'

foldlSplit3 :: forall r s0 s1 s0' s1' s2 s2' t0 t1 t2 a. (TensorOp r, T s0 t0, T s0' t0, T s1 t1, T s1' t1, T s2 t2, T s2' t2, Split s0 s0', Split s1 s1', Split s2 s2') => (a -> r s0' t0 -> r s1' t1 -> r s2' t2 -> a) -> a -> r s0 t0 -> r s1 t1 -> r s2 t2 -> a
foldlSplit3 func inital lhs rhs thd = 
  if lhsShape == lhsShape' && rhsShape == rhsShape' && thdShape == thdShape' then
    func inital (assumeEqShape lhs) (assumeEqShape rhs) (assumeEqShape thd)
  else
    foldl (\accum (ls, rs, ts) -> 
      let l = unsafeSlice lhs ls
          r = unsafeSlice rhs rs
          t = unsafeSlice thd ts
      in  func accum l r t) inital (zip3 lhs' rhs' thd')
  where lhsShape  = shapeVal (Proxy :: Proxy s0)
        rhsShape  = shapeVal (Proxy :: Proxy s1)
        thdShape  = shapeVal (Proxy :: Proxy s2)
        lhsShape' = shapeVal (Proxy :: Proxy s0') 
        rhsShape' = shapeVal (Proxy :: Proxy s1') 
        thdShape' = shapeVal (Proxy :: Proxy s2')
        lhs' = determindSpliting lhsShape lhsShape'
        rhs' = determindSpliting rhsShape rhsShape'
        thd' = determindSpliting thdShape thdShape'

-- Orphans
instance (TensorOp r, T s t, Num t) => Num (r s t) where
  (+) = unsafePairwiseAdd
  (-) = unsafePairwiseSub
  (*) = unsafePairwiseMul

  abs = unsafePairwiseAbs
  negate = unsafePairwiseNegate
  signum = unsafePairwiseSignum

  fromInteger = splat . fromInteger

instance (TensorOp r, T s t, Fractional t) => Fractional (r s t) where
  (/) = unsafePairwiseDiv
  fromRational = splat . fromRational

instance (Floating t, T s t, TensorOp r) => Floating (r s t) where
  pi = splat pi
  exp = unsafePairwiseExp
  log = unsafePairwiseLog
  sin = unsafePairwiseSin
  cos = unsafePairwiseCos
