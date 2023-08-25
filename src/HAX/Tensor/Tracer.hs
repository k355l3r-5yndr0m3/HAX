{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE LiberalTypeSynonyms #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE DataKinds #-}
module HAX.Tensor.Tracer where
import Prelude hiding (lookup)

import HAX.Tensor.Typeclass
import HAX.Tensor.Shape

import Data.IntMap.Strict hiding (singleton)
import Data.List (singleton)
import Data.Proxy

import GHC.StableName

import MLIR
import MLIR.C.IR (Value)

import HAX.TList

import qualified MLIR.Dialect.Func as Func
import qualified Stablehlo.Dialect.Stablehlo as SHLO
import Data.Dynamic (Typeable)

import HAX.Jit

newtype Tracer (s :: Shape) t = Tracer (IntMap Value -> BlockM (IntMap Value, Value)) deriving (Typeable)

instance Trace Tracer where
--  placeholder i = (Tracer $ \ tbl -> (tbl, ) <$> blkArg i, i + 1)


instance (KnownShape s, Tensorial t, Num t) => Num (Tracer s t) where
  lhs + rhs = Tracer $ \ t0 -> do 
    (t1, _lhs) <- valueOf lhs t0
    (t2, _rhs) <- valueOf rhs t1
    (t2, ) <$> SHLO._AddOp _lhs _rhs resultType
    where resultType = rankedTensorType (fromIntegral <$> shapeVal (Proxy :: Proxy s)) (shloTensorType (Proxy :: Proxy t)) attributeNull

  lhs - rhs = Tracer $ \ t0 -> do 
    (t1, _lhs) <- valueOf lhs t0
    (t2, _rhs) <- valueOf rhs t1
    (t2, ) <$> SHLO._SubtractOp _lhs _rhs resultType
    where resultType = rankedTensorType (fromIntegral <$> shapeVal (Proxy :: Proxy s)) (shloTensorType (Proxy :: Proxy t)) attributeNull
    
  lhs * rhs = Tracer $ \ t0 -> do 
    (t1, _lhs) <- valueOf lhs t0
    (t2, _rhs) <- valueOf rhs t1
    (t2, ) <$> SHLO._MulOp _lhs _rhs resultType
    where resultType = rankedTensorType (fromIntegral <$> shapeVal (Proxy :: Proxy s)) (shloTensorType (Proxy :: Proxy t)) attributeNull
  
  signum operand = Tracer $ \ t0 -> do
    (t1, _operand) <- valueOf operand t0 
    (t1, ) <$> SHLO._SignOp _operand resultType
    where resultType = rankedTensorType (fromIntegral <$> shapeVal (Proxy :: Proxy s)) (shloTensorType (Proxy :: Proxy t)) attributeNull

  negate operand = Tracer $ \ t0 -> do 
    (t1, _operand) <- valueOf operand t0 
    (t1, ) <$> SHLO._NegOp _operand resultType
    where resultType = rankedTensorType (fromIntegral <$> shapeVal (Proxy :: Proxy s)) (shloTensorType (Proxy :: Proxy t)) attributeNull

  abs    operand = Tracer $ \ t0 -> do 
    (t1, _operand) <- valueOf operand t0
    (t1, ) <$> SHLO._AbsOp _operand resultType
    where resultType = rankedTensorType (fromIntegral <$> shapeVal (Proxy :: Proxy s)) (shloTensorType (Proxy :: Proxy t)) attributeNull

  fromInteger literal = Tracer $ \ t0 -> do 
    (t0, ) <$> SHLO._ConstantOp (denseSplatAttr resultType a) resultType 
    where resultType = rankedTensorType shape (shloTensorType (Proxy :: Proxy t)) attributeNull
          shape      = fromIntegral <$> shapeVal (Proxy :: Proxy s)
          a :: t     = fromInteger literal



instance (KnownShape s, Tensorial t, Fractional t) => Fractional (Tracer s t) where
  lhs / rhs = Tracer $ \ t0 -> do 
    (t1, _lhs) <- valueOf lhs t0
    (t2, _rhs) <- valueOf rhs t1
    (t2, ) <$> SHLO._DivOp _lhs _rhs resultType
    where resultType = rankedTensorType (fromIntegral <$> shapeVal (Proxy :: Proxy s)) (shloTensorType (Proxy :: Proxy t)) attributeNull

  fromRational literal = Tracer $ \ t0 -> do 
    (t0, ) <$> SHLO._ConstantOp (denseSplatAttr resultType a) resultType 
    where resultType = rankedTensorType shape (shloTensorType (Proxy :: Proxy t)) attributeNull
          shape      = fromIntegral <$> shapeVal (Proxy :: Proxy s)
          a :: t     = fromRational literal



valueOf :: forall s t. Tracer s t -> IntMap Value -> BlockM (IntMap Value, Value)
valueOf tracer table = do 
  -- NOTE: the $! should not be needed because it is a newtype (I guess because it is already strict???)
  --       I don't know how haskell work 
  --       Leave it here anyway
  hash <- blockMIO $ hashStableName <$> (makeStableName $! tracer)
  case (lookup hash table) of
    Just item -> return (table, item)
    Nothing   -> 
      let Tracer f = tracer 
      in do 
        (table', value) <- f table
        return (insert hash value table', value)


instance (T s t) => Traceable (Tracer s t) where
  trace' _ u = (fmap singleton <$> valueOf u empty, ([], [t]))
    where t  = rankedTensorType (fromInteger <$> shapeVal (Proxy :: Proxy s)) (shloTensorType (Proxy :: Proxy t)) attributeNull

--instance (T s t) => Traceable (TList '[Tracer s t]) where
--  trace' _ (u :+ (:@)) = (fmap singleton <$> valueOf u empty, ([], [t]))
--    where t  = rankedTensorType (fromInteger <$> shapeVal (Proxy :: Proxy s)) (shloTensorType (Proxy :: Proxy t)) attributeNull

instance (T s t, Traceable (TList as)) => Traceable (TList (Tracer s t ':as)) where
  trace' i (u :+   us) = (k', (ins, out:outs))
    where out              = rankedTensorType (fromInteger <$> shapeVal (Proxy :: Proxy s)) (shloTensorType (Proxy :: Proxy t)) attributeNull
          (k, (ins, outs)) = trace' i us
          k' = do 
            (tabl, vals) <- k 
            fmap (:vals) <$> valueOf u tabl


instance (T s t, Traceable f) => Traceable (Tracer s t -> f) where 
  trace' i f = (\ (ins, outs) -> (t:ins, outs)) <$> trace' (i + 1) (f argn)
    where argn = Tracer (\ a -> (a, ) <$> blkArg i)
          t    = rankedTensorType (fromInteger <$> shapeVal (Proxy :: Proxy s)) (shloTensorType (Proxy :: Proxy t)) attributeNull





traceDebug :: Traceable (a -> b) => (a -> b) -> IO ()
traceDebug (trace -> (value, (ins, outs))) = 
  withContext $ do 
    loadDialect_ Func.dialect
    loadDialect_ SHLO.dialect
    m <- moduleOp $ do 
      Func._FuncOp (stringAttr "main")
                   (typeAttr $ functionType ins outs)
                   Nothing Nothing Nothing $ do 
        bb0 <- addBlock ins
        defBlock bb0 $ do 
          _out <- value 
          Func._ReturnOp _out
    dumpModule m
    moduleDestroy m


type instance K Tracer f = f

instance (T s t) => Jit Tracer (Tracer s t) (Tracer s t) where
  jit' _ _ = id
  jit = error "jit should be used with a function."

instance (T s t) => Jit Tracer (TList '[Tracer s t]) (TList '[Tracer s t]) where
  jit' _ _ = id 
  jit  = id

instance (T s t, Jit Tracer (TList f) (TList f')) => Jit Tracer (TList (Tracer s t ': f)) (TList (Tracer s t ': f')) where
  jit' _ _ (a :+ as) = a :+ jit' Proxy Proxy as
  jit  = jit' pt pf 
    where pf :: Proxy (TList (Tracer s t ': f)) = Proxy
          pt :: Proxy Tracer = Proxy

instance (KnownShape s, Tensorial t, Jit Tracer f f') => Jit Tracer (Tracer s t -> f) (Tracer s t -> f') where
  jit' pt _ f tracer = jit' pt pf' (f tracer)
    where pf' = Proxy :: Proxy f
  jit = jit' pt pf 
    where pf = Proxy :: Proxy (Tracer s t -> f)
          pt = Proxy :: Proxy Tracer


