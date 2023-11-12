{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE QuantifiedConstraints #-}
{-# LANGUAGE UndecidableSuperClasses #-}
{-# LANGUAGE GeneralisedNewtypeDeriving #-}
{-# LANGUAGE NoStarIsType #-}
{-# LANGUAGE TypeFamilyDependencies #-}
module HAX.Tensor.Tensorial where
import Prelude hiding (pred)
import HAX.PjRt.BufferType

import HAX.Utils
import HAX.AD.Gradient

import Data.IntMap.Strict (IntMap, empty)
import Data.Kind 
import Data.Proxy
import Data.Reflection
import Data.Primitive
import Data.Int
import Data.Bifunctor
import Data.List 

import Foreign hiding (sizeOf)
import Foreign.C (CIntPtr)

import GHC.TypeLits

import MLIR
import qualified MLIR.Dialect.Func           as Func
import qualified Stablehlo.Dialect.Stablehlo as SHLO

import Stablehlo.Dialect.Stablehlo.Attributes
import Control.Exception (assert)
import Data.Maybe (fromJust)

-- Shape
type Shape = [Nat]
class KnownShape (s :: Shape) where
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

type family ZipWith (x :: [a]) (y :: [b]) (f :: a -> b -> c) :: [c] where
  ZipWith '[] _ _ = '[]
  ZipWith _ '[] _ = '[]
  ZipWith (a ': as) (b ': bs) f = f a b ': ZipWith as bs f

type family Foldl (x :: a) (f :: a -> b -> a) (y :: [b]) :: a where
  Foldl a _ '[] = a
  Foldl a f (b ': bs) = Foldl (f a b) f bs 

type ConcatConstraint (a :: Constraint) (b :: Constraint) = (a, b) :: Constraint

type family ConvolutionShapeConstraint (input :: Shape) (kernel :: Shape) (output :: Shape) :: Constraint where
  ConvolutionShapeConstraint '[] '[] '[] = ()
  ConvolutionShapeConstraint (i ': is) (k ': ks) (o ': os) = (i + 1 - k ~ o, o + k - 1 ~ i, i + 1 - o ~ k, ConvolutionShapeConstraint is ks os)
  ConvolutionShapeConstraint _ _ _ = TypeError (Text "Spatial rank in convolution is not the same")

type Convolution (input :: Shape) (kernel :: Shape) (output :: Shape) = (KnownShape input, KnownShape kernel, KnownShape output, Head input ~ Head output, Last input ~ Head kernel, Last kernel ~ Last output, ConvolutionShapeConstraint (Middle input) (Middle kernel) (Middle output))
type Convolution' (input :: Shape) (kernel :: Shape) (output :: Shape) = (KnownShape input, KnownShape kernel, KnownShape output, Last input ~ Head kernel, Last kernel ~ Last output, ConvolutionShapeConstraint (Init input) (Middle kernel) (Init output))

-- Tensorial
class (Prim (StorageType a), Storable (StorageType a), TypeGet (SHLOType a)) => Tensorial a where
  type SHLOType a
  type StorageType a = r | r -> a

  pjrtBufferType  :: Proxy a -> BufferType
  shloTensorType  :: Proxy a -> SHLOType a
  
  shloTensorType' :: Proxy a -> AnyType
  shloTensorType' = toAnyType . shloTensorType

  staticSizeOf   :: Proxy a -> Int

  splatConstant :: [Int64] -> a   -> BlockM Value
  elemsConstant :: [Int64] -> [a] -> BlockM Value

  fromHaskell :: a -> StorageType a
  toHaskell   :: StorageType a -> a

  literalPad  :: a

  comparisonType :: Proxy a -> ComparisonTypeAttr

instance Tensorial Float where
  type SHLOType Float = F32Type
  type StorageType Float = Float

  pjrtBufferType _ = f32
  shloTensorType _ = F32Type
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
  
instance Tensorial Word8 where
  type SHLOType Word8 = IntegerType
  type StorageType Word8 = Word8

  pjrtBufferType _ = u8
  shloTensorType _ = UI8
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

instance Tensorial Int64 where
  type SHLOType Int64 = IntegerType
  type StorageType Int64 = Int64

  pjrtBufferType _ = s64
  shloTensorType _ = I64
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

newtype Pred = Pred Word8 deriving (Num, Eq, Prim, Storable)
instance Show Pred where 
  show (Pred 0) = "False"
  show _        = "True "

instance Tensorial Bool where
  type SHLOType Bool = IntegerType
  type StorageType Bool = Pred

  pjrtBufferType _ = pred
  shloTensorType _ = I 1
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


-- Traceable
-- NOTE: Consider separating the arguments of a function and its outputs
type family NotFunction f :: Constraint where 
  NotFunction (a -> b) = TypeError (ShowType (a -> b) :<>: Text " is a function!")
  NotFunction _        = ()

-- TODO: Implement dynamic length structures like list
class NotFunction t => TraceableElement t where -- Undecidable super class extension because of this
  constructTracer   :: NotFunction t => CIntPtr -> (CIntPtr, t, [AnyType])
  deconstructTracer :: NotFunction t => t -> (IntMap Value -> BlockM (IntMap Value, [Value]), ([AnyType], [AnyType]))

instance (TraceableElement a, TraceableElement b) => TraceableElement (a <&> b) where
  constructTracer i0 = (i2, a :&: b, at ++ bt)
    where (i1, a, at) = constructTracer i0
          (i2, b, bt) = constructTracer i1
  deconstructTracer (a :&: b) = (\ t0 -> do 
    (t1, _a) <- a' t0
    (t2, _b) <- b' t1
    return (t2, _a ++ _b), join aSig bSig)
    where (a', aSig) = deconstructTracer a 
          (b', bSig) = deconstructTracer b
          join :: ([AnyType], [AnyType]) -> ([AnyType], [AnyType]) -> ([AnyType], [AnyType])
          join (_a, _b) (_c, _d) = (_a ++ _c, _b ++ _d)

instance TraceableElement (Proxy a) where 
  constructTracer     = (, Proxy, [])
  deconstructTracer _ = (\ t -> return (t, []), ([], []))

instance (TraceableElement a0, TraceableElement a1) => 
  TraceableElement (a0, a1) where
  constructTracer i0 = (i2, (t0, t1), k0 ++ k1)
    where (i1, t0, k0) = constructTracer i0
          (i2, t1, k1) = constructTracer i1
  deconstructTracer (a0, a1) = (\ t0 -> do 
    (t1, _a0) <- a0' t0
    (t2, _a1) <- a1' t1
    return (t2, _a0 ++ _a1), join a0Sig a1Sig)
    where join :: ([AnyType], [AnyType]) -> ([AnyType], [AnyType]) -> ([AnyType], [AnyType])
          join (_a, _b) (_c, _d) = (_a ++ _c, _b ++ _d)
          (a0', a0Sig) = deconstructTracer a0
          (a1', a1Sig) = deconstructTracer a1


-- NOTE: What the performance difference between IntMap Value being outside/inside tuple
--       The Traceable class should not be instanced directly, instead use TraceableElement 
--       for types that can be inputs or outputs
class Traceable f where
  trace' :: CIntPtr -> f -> (IntMap Value -> BlockM (IntMap Value, [Value]), ([AnyType], [AnyType]))

-- Note since a <+> is a tree, care must be apply when traverse it so flatteninng and inflatting can be consistent
instance {-# OVERLAPPABLE #-} TraceableElement t => Traceable t where
  trace' _ = deconstructTracer

instance {-# OVERLAPPING #-} (TraceableElement t, Traceable f) => Traceable (t -> f) where
  trace' i f = second (first (tt++)) $ trace' i' (f t)
    where (i', t, tt) = constructTracer i

trace :: Traceable f => f -> (BlockM [Value], ([AnyType], [AnyType]))
trace f = (fmap snd (_fst empty), _snd)
  where (_fst, _snd) = trace' 0 f

traceDebug :: Traceable (a -> b) => (a -> b) -> IO ()
traceDebug (trace -> (value, (ins, outs))) = 
  runContextM $ do 
    loadDialect_ Func.dialect
    loadDialect_ SHLO.dialect
    m <- moduleOp $ do 
      Func._FuncOp "main"
                   (TypeAttr $ FunctionType ins outs)
                   Nothing Nothing Nothing $ do 
        bb0 <- blockGet ins
        blockDef bb0 $ do 
          _out <- value 
          Func._ReturnOp _out
    moduleDump m
    moduleDestroy m

type T s t = (KnownShape s, Tensorial t)
tensorType :: forall a s t. T s t => Proxy (a s t) -> RankedTensorType NullAttr (SHLOType t)
tensorType _ = RankedTensorType shape _type NullAttr
  where shape = fromInteger <$> shapeVal (Proxy :: Proxy s)
        _type = shloTensorType (Proxy :: Proxy t)

tensorType' :: T s t => Proxy (a s t) -> AnyType 
tensorType' = toAnyType . tensorType

class PairwiseOp (r :: Shape -> Type -> Type) where

class ConvertOp (r :: Shape -> Type -> Type) where
  convert :: (T s f, T s g) => r s f -> r s g

class Tensorial t => DifferentiableShapeOp t where
  unsafeBroadcastGrad :: (MathOp r, T s0 t, T s1 t) => G r s0 t -> [Integer] -> r s1 t -> Gradient
  unsafeBroadcastGrad _ _ = nograd
  unsafeTransposeGrad :: (ShapeOp r, T s0 t, T s1 t) => G r s0 t -> [Integer] -> r s1 t -> Gradient
  unsafeTransposeGrad _ _ = nograd

  unsafeReshapeGrad   :: (ShapeOp r, T s0 t, T s1 t) => G r s0 t -> r s1 t -> Gradient
  unsafeReshapeGrad _ = nograd

  unsafeSliceGrad     :: (ShapeOp r, T s0 t, T s1 t) => G r s0 t -> [(Integer, Integer, Integer)] -> r s1 t -> Gradient
  unsafeSliceGrad _ _ = nograd
  unsafePadGrad       :: (ShapeOp r, T s0 t, T s1 t) => G r s0 t -> [(Integer, Integer, Integer)] -> r s1 t -> Gradient
  unsafePadGrad _ _ = nograd

  unsafeReverseGrad   :: (T s0 t, ShapeOp r) => G r s0 t -> [Integer] -> r s0 t -> Gradient
  unsafeReverseGrad _ _ = nograd

  unsafeScatterGrad   :: (ShapeOp r, T s0 t, T s1 t, T s2 t) => G r s0 t -> r s1 Int64 -> G r s2 t -> [Integer] -> [Integer] -> [Integer] -> Integer -> r s0 t -> Gradient
  unsafeScatterGrad _ _ _ _ _ _ _ = nograd
  unsafeGatherGrad    :: (ShapeOp r, T s0 t, T s1 t, T s2 t) => G r s0 t -> r s1 Int64 -> [Integer] -> [Integer] -> [Integer] -> Integer -> r s2 t -> Gradient
  unsafeGatherGrad _ _ _ _ _ _ = nograd

  unsafeConcatGrad    :: (ShapeOp r, T s0 t, T s1 t, T s2 t) => Integer -> G r s0 t -> G r s1 t -> r s2 t -> Gradient
  unsafeConcatGrad _ _ _ = nograd
instance DifferentiableShapeOp Float where
  unsafeBroadcastGrad :: forall s0 s1 r. (MathOp r, KnownShape s0, KnownShape s1) => G r s0 Float -> [Integer] -> r s1 Float -> Gradient
  unsafeBroadcastGrad f' dims i = 
    reifyShape reduceResult $ \(same (unsafeReduceAdd i reduceDims) -> reduced) -> 
      f' $ unsafeTranspose reduced perm
    where same :: KnownShape s => r s t -> Proxy s -> r s t
          same = const 
          r1 = shapeRank (Proxy :: Proxy s1)
          s1 = shapeVal  (Proxy :: Proxy s1)
          reduceDims   = [0..r1 - 1] \\ dims
          reduceResult = (s1 !!) . fromInteger <$> dims 
          perm = 
            let sorted = sort dims
                assocs = zip sorted [0..]
                permed = fmap (`lookup` assocs) dims
            in  fromJust <$> permed

  unsafeTransposeGrad f' perm i = f' (unsafeTranspose i perm')
    where perm' = map snd $ sortOn fst $ zip perm [0..] 

  unsafeReshapeGrad f' i = f' $ unsafeReshape i

  unsafeSliceGrad :: forall s0 s1 r. (ShapeOp r, KnownShape s0, KnownShape s1) => G r s0 Float -> [(Integer, Integer, Integer)] -> r s1 Float -> Gradient
  unsafeSliceGrad f' slicing i = f' $ unsafePad 0 i (zip3 starts higher interior)
    where (starts, _, strides) = unzip3 slicing
          interior = fmap (+(-1)) strides
          higher   = zipWith4 (\low axis st tot -> tot - low - (axis - 1) * st - 1) starts (shapeVal (Proxy :: Proxy s1)) strides (shapeVal (Proxy :: Proxy s0))

  unsafePadGrad :: forall s0 s1 r. (ShapeOp r, T s0 Float, T s1 Float) => G r s0 Float -> [(Integer, Integer, Integer)] -> r s1 Float -> Gradient
  unsafePadGrad f' padding i = f' $ unsafeSlice i slicing
    where (low, high, internal) = unzip3 padding
          slicing = zipWith4 (\l h a j -> (l, a - h, j + 1)) low high s internal
          s = shapeVal (Proxy :: Proxy s1)

  unsafeReverseGrad :: forall s0 r. (KnownShape s0, ShapeOp r) => G r s0 Float -> [Integer] -> r s0 Float -> Gradient
  unsafeReverseGrad f' dims i = f' $ unsafeReverse i dims

  unsafeGatherGrad f' g offsetAxes collapsedAxes startAxisMap idxVectorAxis i = 
    f' $ unsafeScatter (splat 0) g i offsetAxes collapsedAxes startAxisMap idxVectorAxis

  unsafeScatterGrad :: forall s0 s1 s2 r. (ShapeOp r, T s0 Float, T s1 Float, T s2 Float) => G r s0 Float -> r s1 Int64 -> G r s2 Float -> [Integer] -> [Integer] -> [Integer] -> Integer -> r s0 Float -> Gradient
  unsafeScatterGrad f' g h' updateWindowAxes insertWindowAxes sdtod indexVecAxis i = lhs <+> rhs
    where lhs = f' $ unsafeScatter i g (splat 0 :: r s2 Float) updateWindowAxes insertWindowAxes sdtod indexVecAxis 
          rhs = h' $ unsafeGather  i g updateWindowAxes insertWindowAxes sdtod indexVecAxis sliceSizes
          sliceSizes  = 
            let generator :: [Integer] -> [Integer] -> [Integer]
                generator a []     = a
                generator a (j:js) = generator (let k = fromInteger j in take k a ++ 1 : drop k a) js
            in  generator bounds $ sort insertWindowAxes
          resultShape = shapeVal (Proxy :: Proxy s2)
          bounds      = (resultShape !!) . fromInteger <$> updateWindowAxes

  unsafeConcatGrad :: forall s0 s1 s2 r. (ShapeOp r, KnownShape s0, KnownShape s1, KnownShape s2) => Integer -> G r s0 Float -> G r s1 Float -> r s2 Float -> Gradient
  unsafeConcatGrad dims f' g' i =
    f' (unsafeSlice i lhsSlicing) <+> g' (unsafeSlice i rhsSlicing)
    where lhsSlicing = (0, , 1) <$> shapeVal (Proxy :: Proxy s0)
          offs = shapeVal (Proxy :: Proxy s0) !! fromInteger dims
          limt = shapeVal (Proxy :: Proxy s2) !! fromInteger dims
          rhsSlicing = [if d == dims then (offs, limt, 1) else (0, s, 1) | (d, s) <- zip [0..] $ shapeVal (Proxy :: Proxy s1)]

instance DifferentiableShapeOp Int64
instance DifferentiableShapeOp Word8
instance DifferentiableShapeOp Bool

class ShapeOp (r :: Shape -> Type -> Type) where
  -- without type checking, internal use only
  -- it is easy to detach type from reality
  unsafeBroadcast    :: (T s0 t, T s1 t, DifferentiableShapeOp t, MathOp r) => r s0 t -> [Integer] -> r s1 t
  unsafeTranspose    :: (T s0 t, T s1 t, DifferentiableShapeOp t) => r s0 t -> [Integer] -> r s1 t
  unsafeReshape      :: (T s0 t, T s1 t, DifferentiableShapeOp t) => r s0 t -> r s1 t 
  unsafeSlice        :: (T s0 t, T s1 t, DifferentiableShapeOp t) => r s0 t -> [(Integer, Integer, Integer)] -> r s1 t
  
  -- TODO: Add user defined padding_value
  unsafePad          :: (T s0 t, T s1 t, DifferentiableShapeOp t) => t -> r s0 t -> [(Integer, Integer, Integer)] -> r s1 t
  unsafeReverse      :: (T s0 t, DifferentiableShapeOp t) => r s0 t -> [Integer] -> r s0 t

  -- TODO: Add more test for these
  unsafeScatter      :: (T s0 t, T s1 t, T s2 t, DifferentiableShapeOp t) => r s0 t -> r s1 Int64 -> r s2 t -> [Integer] -> [Integer] -> [Integer] -> Integer -> r s0 t
  unsafeGather       :: (T s0 t, T s1 t, T s2 t, DifferentiableShapeOp t) => r s0 t -> r s1 Int64 -> [Integer] -> [Integer] -> [Integer] -> Integer -> [Integer] -> r s2 t

  -- Concatination
  unsafeConcat :: (T s0 t, T s1 t, T s2 t, DifferentiableShapeOp t) => Integer -> r s0 t -> r s1 t -> r s2 t

  -- TODO: Move this to a different class
  splat :: T s t => t -> r s t

onehot :: forall r s c k. (MathOp r, EqualOp r, KnownShape s, KnownNat c, k ~ (s ++ '[c]), c ~ Last k, KnownShape k, s ~ Init k, DifferentiableShapeOp Int64) => r s Int64 -> r k Bool
onehot indices = isEQ c i
  where c :: r k Int64 = unsafeBroadcast indices [0..r - 1] 
        i :: r k Int64 = unsafeIota r
        r = shapeRank (Proxy :: Proxy s)


(@%) :: forall r t n ns. (ShapeOp r, KnownNat n, KnownShape ns, Num (r '[] Int64), Tensorial t, DifferentiableShapeOp t) => r (n ': ns) t -> Integer -> r ns t
operand @% (fromInteger -> index :: r '[] Int64) = unsafeGather operand index [0..resultRank - 1] [0] [0] 0 (1:resultShape)
  where resultRank  = shapeRank (Proxy :: Proxy ns)
        resultShape = shapeVal  (Proxy :: Proxy ns)

unsafeDiagonal :: forall s0 s1 r t. (T s0 t, T s1 t, MathOp r, DifferentiableShapeOp t) => Integer -> Integer -> r s0 t -> r s1 t
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
broadcast :: (Broadcast org map targ, ShapeOp r, Tensorial t, DifferentiableShapeOp t) => r org t -> Proxy map -> r targ t
broadcast operand (shapeVal -> _map) = unsafeBroadcast operand _map

broadcast' :: forall r t org targ. (ShapeOp r, Tensorial t, Broadcast' org targ, DifferentiableShapeOp t) => r org t -> r targ t
broadcast' operand = unsafeBroadcast operand _map
  where targ = shapeVal (Proxy :: Proxy targ)
        org  = shapeVal (Proxy :: Proxy org)
        _map = take (length org) [fromIntegral (length targ - length org)..] 

transpose :: (ShapeOp r, Tensorial t, Transpose operand perm result, DifferentiableShapeOp t) => r operand t -> Proxy perm -> r result t
transpose operand = unsafeTranspose operand . shapeVal

reshape :: (ShapeOp r, Tensorial t, KnownShape s0, KnownShape s1, Reshapable s0 s1, DifferentiableShapeOp t) => r s0 t -> r s1 t
reshape = unsafeReshape





-- Consider making Num a superclass
--    since all of these function do either addition or multiplication in complex way

-- The lhs (image, or volume, or whatever) is [BATCHING DIM, ...(SPATIAL DIM)..., FEATURE DIM]
--     rhs (kernel)                        is [IN FEAT DIM,  ...(SPATIAL DIM)..., OUT FEAT DIM]
--     output                              is [BATCHING DIM, ...(SPATIAL DIM)..., FEATURE DIM]
-- This is to simplify implementation


-- TODO: Change integer to int (or int64)
class ShapeOp r => MathOp r where
  unsafeDotGeneral  :: (T s0 t, T s1 t, T s2 t, Num t) => r s0 t -> r s1 t -> DotDimensionNumbersAttr -> r s2 t
  unsafeReduceAdd   :: (T s0 t, T s1 t, Num t) => r s0 t -> [Integer] -> r s1 t
  unsafeReduceMul   :: (T s0 t, T s1 t, Num t) => r s0 t -> [Integer] -> r s1 t

  -- For padding, use explicitly pad the input, this simplify gradient calculation, similarly for dialation
  unsafeConvolution :: (T s0 t, T s1 t, T s2 t) => r s0 t -> r s1 t -> r s2 t
  unsafeIota        :: (T s t) => Integer -> r s t

  -- TODO: Move this to a different class
  linspace :: (KnownNat n, Fractional t, Enum t, Tensorial t) => (t, t) -> r '[n] t

  unsafeMultiIota :: forall s t. (T s t) => [Integer] -> Integer -> r s t
  unsafeMultiIota []     _ = error "idim needs to be given"
  unsafeMultiIota [a]    d = assert (shapeVal (Proxy :: Proxy s) !! fromInteger d == 1) unsafeIota a
  unsafeMultiIota (a:as) d = 
    reifyShape (changeAt d' (const 1) shape) $ \(same (unsafeIota a) -> a') ->
      reifyShape (changeAt d' (+(-1)) shape) $ \(same (unsafeMultiIota as d) -> as') ->
        unsafeConcat d a' as'
    where changeAt :: Int -> (a -> a) -> [a] -> [a]
          changeAt i f n
            | i >= 0    = 
              let changeAt' _ []     = []
                  changeAt' j (b:bs) = if j == 0 then f b:bs else b:changeAt' (j - 1) bs
              in  changeAt' i n
            | otherwise = error "Negative index"
          shape = shapeVal (Proxy :: Proxy s)
          same :: KnownShape p => r p t -> Proxy p -> r p t
          same = const
          d' = fromInteger d
-- TODO: Implement safe version of multiiota





iota :: (Iota d s, MathOp r, KnownShape s, KnownNat d, Tensorial t) => Proxy d -> r s t
iota = unsafeIota . natVal

linearMap :: (MathOp r, KnownNat i, KnownNat o, Tensorial t, Num t) => r '[i, o] t -> r '[i] t -> r '[o] t 
linearMap mat vec = unsafeDotGeneral mat vec dotAttr 
  where dotAttr = DotDimensionNumbersAttr { getBatchingDims = [], getContractingDims = [(0, 0)] }

-- Reduction
reduceAdd :: (MathOp r, T s t, T s' t, KnownShape (Reduce s s'), Num t) => r s t -> Proxy s' -> r (Reduce s s') t
reduceAdd operand = unsafeReduceAdd operand . shapeVal

reduceAdd' :: forall r s t. (MathOp r, T s t, Num t) => r s t -> r '[] t
reduceAdd' = (`unsafeReduceAdd` [0..shapeRank (Proxy :: Proxy s) - 1])

reduceMul :: (MathOp r, T s t, T s' t, KnownShape (Reduce s s'), Num t) => r s t -> Proxy s' -> r (Reduce s s') t
reduceMul operand = unsafeReduceMul operand . shapeVal

reduceMul' :: forall r s t. (MathOp r, T s t, Num t) => r s t -> r '[] t
reduceMul' = (`unsafeReduceMul` [0..shapeRank (Proxy :: Proxy s) - 1])

-- TODO: Implement + - * / etc with automatic broadcasting
prod :: (MathOp r, TensorProductConstraint lhs rhs p, Tensorial t, Num t) => r lhs t -> r rhs t -> r p t
prod x y = unsafeDotGeneral x y attr
  where attr = DotDimensionNumbersAttr { getContractingDims = [], getBatchingDims = [] }

matmul :: (MathOp r, T '[m, n] t, T '[n, p] t, T '[m, p] t, Num t) => r '[m, n] t -> r '[n, p] t -> r '[m, p] t
matmul lhs rhs = unsafeDotGeneral lhs rhs attr
  where attr = DotDimensionNumbersAttr { getContractingDims = [(1, 0)], getBatchingDims = [] }

convolution :: (Convolution input kernel output, MathOp r, Num t, Tensorial t) => r input t -> r kernel t -> r output t 
convolution = unsafeConvolution

convolution' :: forall input kernel output r t. (MathOp r, Tensorial t, Convolution' input kernel output, DifferentiableShapeOp t) => r input t -> r kernel t -> r output t
convolution' input kernel = unsafeReshape (unsafeConvolution input' kernel :: r (1 ': output) t)
  where input' = unsafeReshape input :: r (1 ': input) t

class Tensorial t => SelectOp r t where
  branch :: KnownShape s => r s t -> r s t -> r '[] Bool -> r s t
  select :: KnownShape s => r s t -> r s t -> r s   Bool -> r s t

class EqualOp r where
  isEQ :: T s t => r s t -> r s t -> r s Bool
  isNE :: T s t => r s t -> r s t -> r s Bool

class EqualOp r => OrderOp r where
  isGT :: T s t => r s t -> r s t -> r s Bool
  isGE :: T s t => r s t -> r s t -> r s Bool
  isLT :: T s t => r s t -> r s t -> r s Bool
  isLE :: T s t => r s t -> r s t -> r s Bool

(|#|) :: (Tensorial t, MathOp a, TensorProductConstraint l r p, Num t) => a l t -> a r t -> a p t
(|#|) = prod
infixl 9 |#|

(|@|) :: (Tensorial t, MathOp r, KnownNat n, KnownNat m, KnownNat q, Num t) => r '[n, m] t -> r '[m, q] t -> r '[n, q] t
(|@|) = matmul
infixl 8 |@|















-- TODO: Add conditional
relu :: (SelectOp r t, KnownShape s, Num (r s t), OrderOp r) => r s t -> r s t
relu x = select 0 x (x `isGT` 0)

leakyrelu :: (Num (r s t), SelectOp r t, KnownShape s, OrderOp r, ShapeOp r, DifferentiableShapeOp t) => r '[] t -> r s t -> r s t
leakyrelu alpha = leakyrelu' $ broadcast alpha (Proxy :: Proxy '[])

leakyrelu' :: (Num (r s t), SelectOp r t, KnownShape s, OrderOp r) => r s t -> r s t -> r s t
leakyrelu' alpha x = x * select alpha 1 (x `isGT` 0)

l2Loss :: (MathOp a, KnownShape s, Num t, Num (a s t), Tensorial t) => a s t -> a '[] t
l2Loss x = reduceAdd' $ x * x 

sigmoid :: Floating a => a -> a
sigmoid x = recip (1 + exp (negate x))
