{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE LiberalTypeSynonyms #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE DataKinds #-}
module HAX.Tensor.Tracer where
import Prelude hiding (lookup)

import HAX.Tensor.Tensorial

import HAX.TList
import HAX.Jit

import Data.IntMap.Strict hiding (singleton)
import Data.List (singleton)
import Data.Proxy
import Data.Bifunctor

import GHC.StableName

import MLIR

import qualified MLIR.Dialect.Func as Func
import qualified Stablehlo.Dialect.Stablehlo as SHLO


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



instance (KnownShape s, Tensorial t, Fractional t) => Fractional (Tracer s t) where
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


instance (T s t) => Traceable (Tracer s t) where
  trace' _ u = (fmap singleton <$> valueOf u empty, ([], [_type]))
    where _type = tensorType' (Proxy :: Proxy (Tracer s t))

instance (T s t, Traceable (TList as)) => Traceable (TList (Tracer s t ':as)) where
  trace' i (u :+   us) = (k', (ins, _type:outs))
    where _type = tensorType' (Proxy :: Proxy (Tracer s t))
          (k, (ins, outs)) = trace' i us
          k' = do 
            (tabl, vals) <- k 
            fmap (:vals) <$> valueOf u tabl

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
instance (T s t) => Jit Tracer (Tracer s t) where
  type JitResult Tracer (Tracer s t) = Tracer s t
  type JitCache  Tracer (Tracer s t) = Tracer s t
  
  jit' = id
  jitInit = id

instance (T s t, JitTracer f) => Jit Tracer (Tracer s t -> f) where
  type JitResult Tracer (Tracer s t -> f) = Tracer s t -> f
  type JitCache  Tracer (Tracer s t -> f) = Tracer s t -> JitCache Tracer f

  jit' = id
  jitInit = id
