{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE QuantifiedConstraints #-}
{-# LANGUAGE UndecidableSuperClasses #-}
{-# LANGUAGE GeneralisedNewtypeDeriving #-}
module HAX.Tensor.Tensorial where
import Prelude hiding (pred)
import HAX.PjRt.BufferType

import HAX.Utils

import Data.IntMap.Strict (IntMap, empty)
import Data.Primitive.ByteArray
import Data.Kind 
import Data.Proxy
import Data.Reflection
import Data.Primitive
import Data.Int
import Data.Bifunctor

import Foreign hiding (sizeOf)
import Foreign.C (CIntPtr)

import GHC.TypeLits

import MLIR
import qualified MLIR.Dialect.Func           as Func
import qualified Stablehlo.Dialect.Stablehlo as SHLO

import Stablehlo.Dialect.Stablehlo.Attributes

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
type family InitEq (lhs :: [a]) (rhs :: [a]) (f :: Constraint) :: Constraint where
  InitEq '[] _ _ = ()
  InitEq _ '[] _ = ()
  InitEq (a ': lhs) (a ': rhs) f = InitEq lhs rhs f
  InitEq _ _ f   = f
type TailEq lhs rhs f = InitEq (ReverseList lhs) (ReverseList rhs) f

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

-- Tensorial
class (Prim a, Storable a, DenseIntOrFPElementsAttr (DenseElemsAttr a), DenseIntOrFPElementsAttr (DenseSplatAttr a), TypeGet (SHLOType a) ) => Tensorial a where
  type SHLOType a
  type DenseSplatAttr a
  type DenseElemsAttr a

  pjrtBufferType  :: Proxy a -> BufferType
  shloTensorType  :: Proxy a -> SHLOType a
  
  shloTensorType' :: Proxy a -> AnyType
  shloTensorType' = toAnyType . shloTensorType

  staticSizeOf   :: Proxy a -> Int

  denseSplatAttr :: [Int64] -> a -> DenseSplatAttr a
  denseElemsAttr :: [Int64] -> PrimArray a -> DenseElemsAttr a

  unitElement :: a  
  nullElement :: a


instance Tensorial Float where
  type SHLOType Float = F32Type
  type DenseSplatAttr Float = DenseIntOrFPElements (RankedTensorType NullAttr F32Type) Float
  type DenseElemsAttr Float = DenseElementsRawBuffer (RankedTensorType NullAttr F32Type) 

  pjrtBufferType _ = f32
  shloTensorType _ = F32Type
  staticSizeOf   _ = sizeOf (0 :: Float)
  
  denseSplatAttr shape = DenseIntOrFPElements (RankedTensorType shape F32Type NullAttr)
  denseElemsAttr shape tensorData = DenseElementsRawBuffer (RankedTensorType shape F32Type NullAttr) (primArrayToByteArray tensorData)
    where primArrayToByteArray :: PrimArray a -> ByteArray 
          primArrayToByteArray (PrimArray a) = ByteArray a

  unitElement = 1
  nullElement = 0

instance Tensorial Word8 where
  type SHLOType Word8 = IntegerType
  type DenseSplatAttr Word8 = DenseIntOrFPElements (RankedTensorType NullAttr IntegerType) Word8
  type DenseElemsAttr Word8 = DenseElementsRawBuffer (RankedTensorType NullAttr IntegerType)

  pjrtBufferType _ = u8
  shloTensorType _ = UI8
  staticSizeOf   _ = sizeOf (0 :: Word8)

  denseSplatAttr shape = DenseIntOrFPElements (RankedTensorType shape UI8 NullAttr)
  denseElemsAttr shape tensorData = DenseElementsRawBuffer (RankedTensorType shape UI8 NullAttr) (primArrayToByteArray tensorData)
    where primArrayToByteArray :: PrimArray a -> ByteArray 
          primArrayToByteArray (PrimArray a) = ByteArray a
  
  unitElement = 1
  nullElement = 1
newtype Pred = Pred Word8 deriving (Num, Eq, Prim, Storable)
instance Show Pred where 
  show (Pred 0) = "False"
  show _        = "True "
instance Tensorial Pred where
  type SHLOType Pred = IntegerType
  type DenseSplatAttr Pred = DenseIntOrFPElements (RankedTensorType NullAttr IntegerType) Bool
  type DenseElemsAttr Pred = DenseIntOrFPElements (RankedTensorType NullAttr IntegerType) [Bool]
  
  pjrtBufferType _ = pred
  shloTensorType _ = I 1
  staticSizeOf   _ = sizeOf (0 :: Word8)

  denseSplatAttr shape (Pred w) = DenseIntOrFPElements (RankedTensorType shape (I 1) NullAttr) (w > 0)
  denseElemsAttr shape tensorData = DenseIntOrFPElements (RankedTensorType shape (I 1) NullAttr) ((Pred 0 /=) <$> primArrayToList tensorData)

  unitElement = Pred 1
  nullElement = Pred 0

-- Traceable
-- NOTE: Consider separating the arguments of a function and its outputs
type family NotFunction f :: Constraint where 
  NotFunction (a -> b) = TypeError (ShowType (a -> b) :<>: Text " is a function!")
  NotFunction _        = ()

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

trace :: Traceable (a -> b) => (a -> b) -> (BlockM [Value], ([AnyType], [AnyType]))
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

class Tensorial t => ShapeOp (a :: Shape -> Type -> Type) t where
  -- without type checking, internal use only
  -- it is easy to detach type from reality
  unsafeBroadcast  :: (KnownShape s0, KnownShape s1) => a s0 t -> [Integer] -> a s1 t
  unsafeTranspose  :: (KnownShape s0, KnownShape s1) => a s0 t -> [Integer] -> a s1 t

  broadcast :: Broadcast org map targ => a org t -> Proxy map -> a targ t
  broadcast operand (shapeVal -> _map) = unsafeBroadcast operand _map

  broadcast' :: forall org targ. Broadcast' org targ => a org t -> a targ t
  broadcast' operand = unsafeBroadcast operand _map
    where targ = shapeVal (Proxy :: Proxy targ)
          org  = shapeVal (Proxy :: Proxy org)
          _map = take (length org) [fromIntegral (length targ - length org)..] 

  transpose :: Transpose operand perm result => a operand t -> Proxy perm -> a result t
  transpose operand = unsafeTranspose operand . shapeVal

  splat :: (KnownShape s) => t -> a s t

class ShapeOp r t => MathOp r t where
  unsafeDotGeneral :: (KnownShape s0, KnownShape s1, KnownShape s2) => r s0 t -> r s1 t -> DotDimensionNumbersAttr -> r s2 t
  unsafeReduceAdd  :: (KnownShape s0, KnownShape s1, Num t) => r s0 t -> [Integer] -> r s1 t
  unsafeReduceMul  :: (KnownShape s0, KnownShape s1, Num t) => r s0 t -> [Integer] -> r s1 t

  linearMap :: (KnownNat i, KnownNat o) => r '[i, o] t -> r '[i] t -> r '[o] t 
  linearMap mat vec = unsafeDotGeneral mat vec dotAttr 
    where dotAttr = DotDimensionNumbersAttr { getBatchingDims = [], getContractingDims = [(0, 0)] }

  sigma :: (KnownShape s, KnownShape s', KnownShape (Reduce s s'), Num t) => r s t -> Proxy s' -> r (Reduce s s') t
  sigma operand = unsafeReduceAdd operand . shapeVal

  sigma' :: forall s. (T s t, Num t) => r s t -> r '[] t
  sigma' = (`unsafeReduceAdd` [0..shapeRank (Proxy :: Proxy s) - 1])

  -- Name collision, yes I hate it
  _pi :: (KnownShape s, KnownShape s', KnownShape (Reduce s s'), Num t) => r s t -> Proxy s' -> r (Reduce s s') t
  _pi operand = unsafeReduceMul operand . shapeVal

  _pi' :: forall s. (T s t, Num t) => r s t -> r '[] t
  _pi' = (`unsafeReduceMul` [0..shapeRank (Proxy :: Proxy s) - 1])

  -- TODO: Implement + - * / etc with automatic broadcasting
  prod :: (TensorProductConstraint lhs rhs p, Tensorial t) => r lhs t -> r rhs t -> r p t
  prod x y = unsafeDotGeneral x y attr
    where attr = DotDimensionNumbersAttr { getContractingDims = [], getBatchingDims = [] }
  
  matmul :: (T '[m, n] t, T '[n, p] t, T '[m, p] t) => r '[m, n] t -> r '[n, p] t -> r '[m, p] t
  matmul lhs rhs = unsafeDotGeneral lhs rhs attr
    where attr = DotDimensionNumbersAttr { getContractingDims = [(1, 0)], getBatchingDims = [] }

  linspace :: (KnownNat n, Fractional t, Enum t) => (t, t) -> r '[n] t

class Tensorial t => SelectOp r t where
  branch :: KnownShape s => r s t -> r s t -> r '[] Pred -> r s t
  select :: KnownShape s => r s t -> r s t -> r s   Pred -> r s t

class Tensorial t => EqualOp r t where
  isEQ :: KnownShape s => r s t -> r s t -> r s Pred
  isNE :: KnownShape s => r s t -> r s t -> r s Pred

class EqualOp r t => OrderOp r t where
  isGT :: KnownShape s => r s t -> r s t -> r s Pred
  isGE :: KnownShape s => r s t -> r s t -> r s Pred
  isLT :: KnownShape s => r s t -> r s t -> r s Pred
  isLE :: KnownShape s => r s t -> r s t -> r s Pred

(|#|) :: (MathOp a t, TensorProductConstraint l r p) => a l t -> a r t -> a p t
(|#|) = prod
infixl 9 |#|

(|@|) :: (MathOp r t, KnownNat n, KnownNat m, KnownNat q) => r '[n, m] t -> r '[m, q] t -> r '[n, q] t
(|@|) = matmul
infixl 8 |@|

-- TODO: Add conditional
relu :: Fractional a => a -> a
relu x = x * ((signum x + 1) / 2)

l2Loss :: (MathOp a t, KnownShape s, Num t, Num (a s t)) => a s t -> a '[] t
l2Loss x = sigma' $ x * x 
