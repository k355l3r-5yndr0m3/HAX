{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ViewPatterns #-}
module HAX.Tensor.Tracer where
import Prelude hiding (lookup)

import HAX.Tensor.Tensorial

import HAX.HList
import HAX.Jit

import Control.Exception

import Data.IntMap.Strict hiding (singleton, null)
import Data.List (singleton)
import Data.Proxy
import Data.Bifunctor
import Foreign

import GHC.StableName

import MLIR

import qualified Stablehlo.Dialect.Stablehlo as SHLO
import Stablehlo.Dialect.Stablehlo.Attributes

-- The problem of transformation, how can this be accompished
-- firstly, gradient
--   what is the gradient of vmap
--     firstly, vmap does not really exist, it just transform the code
--   or just skipping vmap because it is so difficult to implement
newtype Tracer (s :: Shape) t = Tracer (IntMap Value -> BlockM (IntMap Value, Value))


newtype TracerM a = TracerM (IntMap Value -> BlockM (IntMap Value, a))
instance Functor TracerM where
  fmap f (TracerM a) = TracerM $ \ t0 -> do 
    (t1, a') <- a t0 
    return (t1, f a')
instance Applicative TracerM where
  pure a = TracerM $ \ t0 -> return (t0, a)
  TracerM f <*> TracerM a = TracerM $ \ t0 -> do 
    (t1, f') <- f t0 
    (t2, a') <- a t1 
    return (t2, f' a')
instance Monad TracerM where
  TracerM a >>= f = TracerM $ \ t0 -> do 
    (t1, a') <- a t0 
    let TracerM b = f a'
    b t1 

mkTracer :: TracerM Value -> Tracer s t
mkTracer (TracerM f) = Tracer f

sharing' :: forall s t. Tracer s t -> IntMap Value -> BlockM (IntMap Value, Value)
sharing' tracer table = do 
  -- NOTE: the $! should not be needed because it is a newtype (I guess because it is already strict???)
  --       I don't know how haskell work 
  --       Leave it here anyway
  hash <- blockRunIO (hashStableName <$> (makeStableName $! tracer))
  case lookup hash table of
    Just item -> return (table, item)
    Nothing   -> 
      let Tracer f = tracer 
      in do 
        (table', value) <- f table
        return (insert hash value table', value)
sharing :: forall s t. Tracer s t -> TracerM Value 
sharing tracer = TracerM (sharing' tracer)

retval :: BlockM Value -> TracerM Value
retval v = TracerM $ \ table -> 
  (table, ) <$> v

instance (KnownShape s, Tensorial t, Num t) => Num (Tracer s t) where
  lhs + rhs = mkTracer $ do 
    _lhs <- sharing lhs
    _rhs <- sharing rhs
    retval $ SHLO._AddOp _lhs _rhs _type
    where _type = tensorType' (Proxy :: Proxy (Tracer s t))

  lhs - rhs = mkTracer $ do 
    _lhs <- sharing lhs
    _rhs <- sharing rhs
    retval $ SHLO._SubtractOp _lhs _rhs _type
    where _type = tensorType' (Proxy :: Proxy (Tracer s t))
    
  lhs * rhs = mkTracer $ do 
    _lhs <- sharing lhs
    _rhs <- sharing rhs
    retval $ SHLO._MulOp _lhs _rhs _type
    where _type = tensorType' (Proxy :: Proxy (Tracer s t))
  
  signum operand = mkTracer $ do
    _operand <- sharing operand
    retval $ SHLO._SignOp _operand _type
    where _type = tensorType' (Proxy :: Proxy (Tracer s t))

  negate operand = mkTracer $ do 
    _operand <- sharing operand 
    retval $ SHLO._NegOp _operand _type
    where _type = tensorType' (Proxy :: Proxy (Tracer s t))

  abs    operand = mkTracer $ do 
    _operand <- sharing operand
    retval $ SHLO._AbsOp _operand _type
    where _type = tensorType' (Proxy :: Proxy (Tracer s t))

  fromInteger literal = mkTracer $ do 
    retval $ SHLO._ConstantOp (denseSplatAttr shape a) _type 
    where _type = tensorType' (Proxy :: Proxy (Tracer s t))
          shape = fromInteger <$> shapeVal (Proxy :: Proxy s)
          a :: t = fromInteger literal

instance (T s t, Fractional t) => Fractional (Tracer s t) where
  lhs / rhs = mkTracer $ do 
    _lhs <- sharing lhs
    _rhs <- sharing rhs
    retval $ SHLO._DivOp _lhs _rhs _type
    where _type = tensorType' (Proxy :: Proxy (Tracer s t))

  fromRational literal = mkTracer $ do 
    retval $ SHLO._ConstantOp (denseSplatAttr shape a) _type
    where _type = tensorType' (Proxy :: Proxy (Tracer s t))
          shape      = fromIntegral <$> shapeVal (Proxy :: Proxy s)
          a :: t     = fromRational literal




instance T s t => Traceable (Tracer s t) where
  trace' _ u = (fmap (fmap singleton) . sharing' u, ([], [_type]))
    where _type = tensorType' (Proxy :: Proxy (Tracer s t))


instance (T s t, Traceable f) => Traceable (Tracer s t -> f) where 
  trace' i f = first (_type :) <$> trace' (i + 1) (f argn)
    where argn = Tracer (\ a -> (a, ) <$> blockArg i)
          _type = tensorType' (Proxy :: Proxy (Tracer s t))


type JitTracer f = (Jit Tracer f, f ~ JitCache Tracer f)
instance Jit Tracer (Proxy (Tracer s t)) where
  type JitResult Tracer (Proxy (Tracer s t)) = Proxy (Tracer s t)
  type JitCache  Tracer (Proxy (Tracer s t)) = Proxy (Tracer s t)

  jit' = id
  jitInit = id
  jitReify = undefined

instance T s t => Jit Tracer (Tracer s t) where
  type JitResult Tracer (Tracer s t) = Tracer s t
  type JitCache  Tracer (Tracer s t) = Tracer s t
  
  jit' = id
  jitInit = id
  jitReify = undefined


-- Because <+> can form binary tree, great care is needed to flaten and unflaten it
instance (JitTracer a, JitTracer b) => Jit Tracer (a <+> b) where
  type JitResult Tracer (a <+> b) = JitResult Tracer a <+> JitResult Tracer b
  type JitCache  Tracer (a <+> b) = a <+> b
  
  jit' (a :+: b) = jit' a :+: jit' b
  jitInit = id
  jitReify = undefined

instance (T s t, JitTracer f) => Jit Tracer (Tracer s t -> f) where
  type JitResult Tracer (Tracer s t -> f) = Tracer s t -> f
  type JitCache  Tracer (Tracer s t -> f) = Tracer s t -> f

  jit' = id
  jitInit = id
  jitReify = undefined


newtype BroadcastMap = BroadcastMap [Word64]
getBroadcastMap :: KnownShape s => Proxy s -> BroadcastMap
getBroadcastMap = BroadcastMap . fmap fromInteger . shapeVal
instance AttrGet BroadcastMap where
  attrGet (BroadcastMap mapping) = 
    if null mapping then 
      attrGet $ DenseIntOrFPElements (RankedTensorType [0] I64 NullAttr) mapping
    else 
      attrGet $ DenseIntOrFPElements (VectorType [fromIntegral $ length mapping] I64) mapping
instance DenseIntOrFPElementsAttr BroadcastMap
instance DenseIntElementsAttr BroadcastMap

newtype ReduceDims = ReduceDims [Word64]
getReduceDims :: KnownShape s => Proxy s -> ReduceDims
getReduceDims = ReduceDims . fmap fromInteger . shapeVal
instance AttrGet ReduceDims where
  attrGet (ReduceDims dims) = 
    if null dims then 
      attrGet $ DenseIntOrFPElements (RankedTensorType [0] I64 NullAttr) dims
    else 
      attrGet $ DenseIntOrFPElements (VectorType [fromIntegral $ length dims] I64) dims
instance DenseIntOrFPElementsAttr ReduceDims
instance DenseIntElementsAttr ReduceDims

instance TensorOp Tracer where
  unsafeBroadcast :: forall s s' t. (T s t, T s' t) => Tracer s t -> [Integer] -> Tracer s' t
  unsafeBroadcast operand idxmap@(BroadcastMap . fmap fromInteger -> _map) = 
    assert correctness $ mkTracer $ do 
    _operand <- sharing operand
    retval $ SHLO._BroadcastInDimOp _map _operand _type
    where correctness :: Bool
          correctness = 
            let isUnique :: [Integer] -> Bool
                isUnique []     = True
                isUnique (a:as) = notElem a as && isUnique as
                src = shapeVal (Proxy :: Proxy s)
                dst = shapeVal (Proxy :: Proxy s')
            in  isUnique idxmap && src == fmap (dst !!) (fromInteger <$> idxmap)
          _type = tensorType' (Proxy :: Proxy (Tracer s' t))

  -- TODO: Add runtime checking
  unsafeReduce :: forall s0 s1 t. (T s0 t, T s1 t) => Tracer s0 t -> (Value -> Value -> AnyType -> BlockM Value) -> t -> [Integer] -> Tracer s1 t
  unsafeReduce operand body (splat -> initvalue :: Tracer '[] t) dims = mkTracer $ do 
    _operand   <- sharing operand
    _initvalue <- sharing initvalue
    retval $ head <$> SHLO._ReduceOp _dims [_operand, _initvalue] (do 
      bb0 <- blockGet [scalar, scalar]
      blockDef bb0 $ do 
        _arg0 <- blockArg 0 
        _arg1 <- blockArg 1 
        _out  <- body _arg0 _arg1 scalar 
        SHLO._ReturnOp [_out]) [_type]
    where _type = tensorType' (Proxy :: Proxy (Tracer s1 t))
          _dims = ReduceDims (fromInteger <$> dims)
          scalar = tensorType' (Proxy :: Proxy (Tracer '[] t))
  
  unsafeDotGeneral :: forall s0 s1 s2 t. (T s0 t, T s1 t, T s2 t) => Tracer s0 t -> Tracer s1 t -> DotDimensionNumbersAttr -> Tracer s2 t
  unsafeDotGeneral lhs rhs attr = mkTracer $ do 
    _lhs <- sharing lhs 
    _rhs <- sharing rhs 
    retval $ SHLO._DotGeneralOp attr Nothing _lhs _rhs _type
    where _type = tensorType' (Proxy :: Proxy (Tracer s2 t))

  splat :: forall s t. T s t => t -> Tracer s t
  splat value = mkTracer $ do 
    retval $ SHLO._ConstantOp (denseSplatAttr shape value) _type
    where _type = tensorType' (Proxy :: Proxy (Tracer s t))
          shape = fromInteger <$> shapeVal (Proxy :: Proxy s)
