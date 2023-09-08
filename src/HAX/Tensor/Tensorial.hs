{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE QuantifiedConstraints #-}
module HAX.Tensor.Tensorial where
import HAX.PjRt.BufferType

import HAX.Utils

import Data.IntMap.Strict (IntMap, empty)
import Data.Primitive.ByteArray
import Data.Kind 
import Data.Proxy
import Data.Reflection
import Data.Primitive
import Data.Int

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
class (Prim a, DenseIntOrFPElementsAttr (DenseElemsAttr a), DenseIntOrFPElementsAttr (DenseSplatAttr a), TypeGet (SHLOType a) ) => Tensorial a where
  type SHLOType a
  type DenseSplatAttr a
  type DenseElemsAttr a

  pjrtBufferType  :: Proxy a -> BufferType
  shloTensorType  :: Proxy a -> SHLOType a
  
  shloTensorType' :: Proxy a -> AnyType
  shloTensorType' = toAnyType . shloTensorType

  staticSizeOf   :: Proxy a -> Int

  denseSplatAttr :: [Int64] -> a -> DenseSplatAttr a
  -- TODO: Change ByteArray to something else that is parameterized by a 
  denseElemsAttr :: [Int64] -> ByteArray -> Proxy a -> DenseElemsAttr a

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
  denseElemsAttr shape tensorData _ = DenseElementsRawBuffer (RankedTensorType shape F32Type NullAttr) tensorData

  unitElement = 1
  nullElement = 0

-- Traceable
-- NOTE: What the performance difference between IntMap Value being outside/inside tuple
class Traceable f where
  trace' :: CIntPtr -> f -> (IntMap Value -> BlockM (IntMap Value, [Value]), ([AnyType], [AnyType]))
-- Note since a <+> is a tree, care must be apply when traverse it so flatteninng and inflatting can be consistent
instance (Traceable a, Traceable b) => Traceable (a <+> b) where
  trace' _ (a :+: b) = (\ t0 -> do 
    (t1, _lhs) <- fst lhs t0 
    (t2, _rhs) <- fst rhs t1 
    return (t2, _lhs ++ _rhs), join (snd lhs) (snd rhs))
    where lhs = trace' errmsg a
          rhs = trace' errmsg b
          join :: ([AnyType], [AnyType]) -> ([AnyType], [AnyType]) -> ([AnyType], [AnyType])
          join (_a, _b) (_c, _d) = (_a ++ _c, _b ++ _d)
          errmsg = error "the traced function is not regular"
instance Traceable (Proxy a) where
  trace' _ _ = (\ tbl -> return (tbl, []), ([], []))


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

class Tensorial t => TensorOp (a :: Shape -> Type -> Type) t where
  -- without type checking, internal use only
  -- it is easy to detach type from reality
  unsafeBroadcast  :: (KnownShape s0, KnownShape s1) => a s0 t -> [Integer] -> a s1 t
  -- NOTE unsafeReduce by itself cannot be differentiated, use the other version instead
  unsafeReduce     :: (KnownShape s0, KnownShape s1) => a s0 t -> (Value -> Value -> AnyType -> BlockM Value) -> t -> [Integer] -> a s1 t
  unsafeDotGeneral :: (KnownShape s0, KnownShape s1, KnownShape s2) => a s0 t -> a s1 t -> DotDimensionNumbersAttr -> a s2 t
  unsafeTranspose  :: (KnownShape s0, KnownShape s1) => a s0 t -> [Integer] -> a s1 t


  unsafeReduceAdd :: (KnownShape s0, KnownShape s1, Num t) => a s0 t -> [Integer] -> a s1 t
  unsafeReduceAdd operand = unsafeReduce operand SHLO._AddOp 0

  unsafeReduceMul :: (KnownShape s0, KnownShape s1, Num t) => a s0 t -> [Integer] -> a s1 t
  unsafeReduceMul operand = unsafeReduce operand SHLO._MulOp 1


  -- Type checked
  broadcast :: Broadcast org map targ => a org t -> Proxy map -> a targ t
  broadcast operand (shapeVal -> _map) = unsafeBroadcast operand _map

  broadcast' :: forall org targ. Broadcast' org targ => a org t -> a targ t
  broadcast' operand = unsafeBroadcast operand _map
    where targ = shapeVal (Proxy :: Proxy targ)
          org  = shapeVal (Proxy :: Proxy org)
          _map = take (length org) [fromIntegral (length targ - length org)..] 

  transpose :: Transpose operand perm result => a operand t -> Proxy perm -> a result t
  transpose operand = unsafeTranspose operand . shapeVal

  reduceAdd :: (KnownShape s, KnownShape r, KnownShape (Reduce s r), Num t) => a s t -> Proxy r -> a (Reduce s r) t
  reduceAdd operand = unsafeReduceAdd operand . shapeVal

  reduceMul :: (KnownShape s, KnownShape r, KnownShape (Reduce s r), Num t) => a s t -> Proxy r -> a (Reduce s r) t
  reduceMul operand = unsafeReduceMul operand . shapeVal
  
  -- TODO: Implement + - * / etc with automatic broadcasting
  prod :: (TensorProductConstraint l r p, Tensorial t) => a l t -> a r t -> a p t
  prod x y = unsafeDotGeneral x y attr
    where attr = DotDimensionNumbersAttr { getContractingDims = [], getBatchingDims = [] }
  
  matmul :: (T '[m, n] t, T '[n, p] t, T '[m, p] t) => a '[m, n] t -> a '[n, p] t -> a '[m, p] t
  matmul lhs rhs = unsafeDotGeneral lhs rhs attr
    where attr = DotDimensionNumbersAttr { getContractingDims = [(1, 0)], getBatchingDims = [] }

  splat :: KnownShape s => t -> a s t


(|#|) :: (TensorOp a t, TensorProductConstraint l r p) => a l t -> a r t -> a p t
(|#|) = prod
infixl 8 |#|
