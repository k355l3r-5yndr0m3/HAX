{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DataKinds #-}
module HAX.Tensor.Tracer where
import Prelude hiding (lookup)

import HAX.Math 
import HAX.Tensor.Tensorial

import HAX.HList
import HAX.Jit

import Data.IntMap.Strict hiding (singleton)
import Data.List (singleton)
import Data.Proxy
import Data.Bifunctor
import Foreign

import GHC.StableName

import MLIR

import qualified MLIR.Dialect.Func as Func

import qualified Stablehlo.Dialect.Stablehlo as SHLO
import Stablehlo.Dialect.Stablehlo.Attributes

newtype Tracer (s :: Shape) t = Tracer (IntMap Value -> BlockM (IntMap Value, Value))


instance (KnownShape s, Tensorial t, Num t) => Num (Tracer s t) where
  lhs + rhs = Tracer $ \ t0 -> do 
    (t1, _lhs) <- valueOf lhs t0
    (t2, _rhs) <- valueOf rhs t1
    (t2, ) <$> SHLO._AddOp _lhs _rhs _type
    where _type = tensorType' (Proxy :: Proxy (Tracer s t))

  lhs - rhs = Tracer $ \ t0 -> do 
    (t1, _lhs) <- valueOf lhs t0
    (t2, _rhs) <- valueOf rhs t1
    (t2, ) <$> SHLO._SubtractOp _lhs _rhs _type
    where _type = tensorType' (Proxy :: Proxy (Tracer s t))
    
  lhs * rhs = Tracer $ \ t0 -> do 
    (t1, _lhs) <- valueOf lhs t0
    (t2, _rhs) <- valueOf rhs t1
    (t2, ) <$> SHLO._MulOp _lhs _rhs _type
    where _type = tensorType' (Proxy :: Proxy (Tracer s t))
  
  signum operand = Tracer $ \ t0 -> do
    (t1, _operand) <- valueOf operand t0 
    (t1, ) <$> SHLO._SignOp _operand _type
    where _type = tensorType' (Proxy :: Proxy (Tracer s t))

  negate operand = Tracer $ \ t0 -> do 
    (t1, _operand) <- valueOf operand t0 
    (t1, ) <$> SHLO._NegOp _operand _type
    where _type = tensorType' (Proxy :: Proxy (Tracer s t))

  abs    operand = Tracer $ \ t0 -> do 
    (t1, _operand) <- valueOf operand t0
    (t1, ) <$> SHLO._AbsOp _operand _type
    where _type = tensorType' (Proxy :: Proxy (Tracer s t))

  fromInteger literal = Tracer $ \ t0 -> do 
    (t0, ) <$> SHLO._ConstantOp (denseSplatAttr shape a) _type 
    where _type = tensorType' (Proxy :: Proxy (Tracer s t))
          shape = fromInteger <$> shapeVal (Proxy :: Proxy s)
          a :: t = fromInteger literal



instance (T s t, Fractional t) => Fractional (Tracer s t) where
  lhs / rhs = Tracer $ \ t0 -> do 
    (t1, _lhs) <- valueOf lhs t0
    (t2, _rhs) <- valueOf rhs t1
    (t2, ) <$> SHLO._DivOp _lhs _rhs _type
    where _type = tensorType' (Proxy :: Proxy (Tracer s t))

  fromRational literal = Tracer $ \ t0 -> do 
    (t0, ) <$> SHLO._ConstantOp (denseSplatAttr shape a) _type
    where _type = tensorType' (Proxy :: Proxy (Tracer s t))
          shape      = fromIntegral <$> shapeVal (Proxy :: Proxy s)
          a :: t     = fromRational literal


instance TensorOp Tracer where
  broadcast :: forall t org map targ. (Broadcast org map targ, Tensorial t) => Tracer org t -> Proxy map -> Tracer targ t
  broadcast org _ = Tracer $ \ t0 -> do 
    (t1, _org) <- valueOf org t0 
    (t1, ) <$> SHLO._BroadcastInDimOp mapping _org _type
      where mapping' :: [Word64] = fromInteger <$> shapeVal (Proxy :: Proxy map)
            mapping              = DenseIntOrFPElements (VectorType [fromIntegral $ length mapping'] I64) mapping'
            _type                = tensorType' (Proxy :: Proxy (Tracer targ t))
  broadcast' :: forall org targ t. (Broadcast' org targ, Tensorial t) => Tracer org t -> Tracer targ t  
  broadcast' org = Tracer $ \ t0 -> do 
    (t1, _org) <- valueOf org t0 
    (t1, ) <$> SHLO._BroadcastInDimOp mapping _org _type 
    where _type                = tensorType' (Proxy :: Proxy (Tracer targ t))
          orgRank              = shapeRank (Proxy :: Proxy org)
          targRank             = shapeRank (Proxy :: Proxy targ)
          mapping' :: [Word64] = fromIntegral <$> take (fromInteger orgRank) [targRank - orgRank..]
          mapping              = DenseIntOrFPElements (VectorType [fromIntegral orgRank] I64) mapping'



  prod :: forall l r p t. (TensorProductConstraint l r p, Tensorial t) => Tracer l t -> Tracer r t -> Tracer p t
  prod lhs rhs = Tracer $ \ t0 -> do 
    (t1, _lhs) <- valueOf lhs t0 
    (t2, _rhs) <- valueOf rhs t1 
    (t2, ) <$> SHLO._DotGeneralOp attr Nothing _lhs _rhs _type
    where _type = tensorType' (Proxy :: Proxy (Tracer p t))
          attr  = DotDimensionNumbersAttr {
            getBatchingDims = [],
            getContractingDims = []
          }



valueOf :: forall s t. Tracer s t -> IntMap Value -> BlockM (IntMap Value, Value)
valueOf tracer table = do 
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


instance T s t => Traceable (Tracer s t) where
  trace' _ u = (fmap (fmap singleton) . valueOf u, ([], [_type]))
    where _type = tensorType' (Proxy :: Proxy (Tracer s t))


instance (T s t, Traceable f) => Traceable (Tracer s t -> f) where 
  trace' i f = first (_type :) <$> trace' (i + 1) (f argn)
    where argn = Tracer (\ a -> (a, ) <$> blockArg i)
          _type = tensorType' (Proxy :: Proxy (Tracer s t))


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

