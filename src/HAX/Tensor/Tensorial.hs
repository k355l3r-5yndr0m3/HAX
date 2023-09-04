{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE OverloadedStrings #-}
module HAX.Tensor.Tensorial where
import HAX.PjRt.BufferType

import HAX.Utils

import Data.Proxy
import Data.IntMap.Strict (IntMap, empty)
import Data.Primitive
import Data.Kind 
import Data.Int

import GHC.TypeLits

import Foreign.C

import MLIR 
import qualified MLIR.Dialect.Func           as Func
import qualified Stablehlo.Dialect.Stablehlo as SHLO

-- Shape
type Shape = [Nat]
class KnownShape (s :: Shape) where
  shapeVal :: Proxy s -> [Integer]
  shapeRank :: Proxy s -> Integer
  shapeValHead :: Proxy s -> Integer
  
instance KnownShape '[] where
  shapeVal _ = []
  shapeRank _ = 0
  shapeValHead = error "Shape is of rank 0"

instance (KnownNat a, KnownShape as) => KnownShape (a ': as) where
  shapeVal _ = natVal (Proxy :: Proxy a) : shapeVal (Proxy :: Proxy as)
  shapeRank _ = 1 + shapeRank (Proxy :: Proxy as)
  shapeValHead _ = natVal (Proxy :: Proxy a)

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



-- Tensor operation
class TensorOp (a :: Shape -> Type -> Type) where
  -- Automatic broadcasting
  broadcast  :: (Broadcast org map targ, Tensorial t) => a org t -> Proxy map -> a targ t
  broadcast' :: (Broadcast' org targ, Tensorial t) => a org t -> a targ t  
  
  -- TODO: Implement + - * / etc with automatic broadcasting
  prod :: (TensorProductConstraint l r p, Tensorial t) => a l t -> a r t -> a p t
  dot  :: (T s t, Num t) => a s t -> a s t -> a '[] t


(|#|) :: (TensorOp a, TensorProductConstraint l r p, Tensorial t) => a l t -> a r t -> a p t
(|#|) = prod
infixl 8 |#|


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

type T s t = (KnownShape s, Tensorial t)
tensorType :: forall a s t. T s t => Proxy (a s t) -> RankedTensorType NullAttr (SHLOType t)
tensorType _ = RankedTensorType shape _type NullAttr
  where shape = fromInteger <$> shapeVal (Proxy :: Proxy s)
        _type = shloTensorType (Proxy :: Proxy t)

tensorType' :: T s t => Proxy (a s t) -> AnyType 
tensorType' = toAnyType . tensorType

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

