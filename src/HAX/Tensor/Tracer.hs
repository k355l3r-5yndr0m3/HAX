{-# LANGUAGE TypeFamilyDependencies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE DataKinds #-}
module HAX.Tensor.Tracer where
import Prelude hiding (lookup)

import HAX.Math 
import HAX.Tensor.Tensorial
import HAX.Tensor.Transform

import HAX.HList
import HAX.Jit

import Data.IntMap.Strict hiding (singleton)
import Data.List (singleton)
import Data.Proxy
import Data.Bifunctor


import GHC.StableName
import GHC.TypeLits

import MLIR
import qualified Stablehlo.Dialect.Stablehlo as SHLO
import Stablehlo.Dialect.Stablehlo.Attributes
import Control.Exception (assert)

-- Three mode of infomation transfer, through input and through output, or through the data itself
-- Passing through the returned BlockM monad is more elegant, but it emposes limits. Since sharing checking is 
--    evaluated before the monand, there is no way to share value
-- Passing through data structure might be better, but sharing might still be impossible

data Tracer (s :: Shape) t = Tracer { getLabel :: Word, getTracer :: IntMap Value -> Transform -> BlockM (IntMap Value, Value) }
-- NOTE: A possible problem with this is that if given a different transformation, it might not 
--       return the correctly transformed IR, if this Tracer has been passed through with anothet 
--       Transform prior. But this might not be problem. 
--       The solution is to include the Transform infomation in the look up key
sharing :: forall s t. Tracer s t -> IntMap Value -> Transform -> BlockM (IntMap Value, Value)
sharing tracer table transform = do 
  hash <- blockRunIO (hashStableName <$> (makeStableName $! tracer))
  case lookup hash table of
    Just item -> return (table, item)
    Nothing   -> 
      let Tracer _ f = tracer 
      in do 
        (table', value) <- f table transform
        return (insert hash value table', value)


-- NOTE: This does not share intermediate value, which is something to fix
capture' :: TypeGet (RankedTensorType e t) => Word -> Word -> Transform -> RankedTensorType e t -> Value -> BlockM (RankedTensorType e t, Value)
capture' top idx tf ranked value
  | top == idx = return (ranked, value)
  | top >  idx = 
    case tf of 
      Id          -> error "Unexpectedly short transformation stack"
      V dim other -> do 
        (RankedTensorType shape t e, val) <- capture' (top - 1) idx other ranked value
        let _map    = BroadcastMap $ take (length shape) [1..]
            ranked' = RankedTensorType (dim:shape) t e
        (ranked', ) <$> SHLO._BroadcastInDimOp _map val (toAnyType ranked')
  | otherwise  = error "Unexpectedly high transformation stack"

capture :: TypeGet (RankedTensorType e t) => Word -> Word -> Transform -> RankedTensorType e t -> (IntMap Value, Value) -> BlockM (IntMap Value, Value)
capture top idx tf ranked (tbl, value) = do 
  (_, value') <- capture' top idx tf ranked value
  return (tbl, value')

capsha :: forall s t. T s t => Tracer s t -> IntMap Value -> Transform -> Word -> BlockM (IntMap Value, Value)
capsha tracer table tf top = do 
  capture top (getLabel tracer) tf (tensorType (Proxy :: Proxy (Tracer s t))) =<< sharing tracer table tf

instance (KnownShape s, Tensorial t, Num t) => Num (Tracer s t) where
  lhs + rhs = Tracer label $ \ t0 (transformTruncate label -> tf) -> do 
    let _type = toAnyType $ applyTransform tf ranked
    -- (t1, _lhs) <- sharing lhs t0 tf
    -- (t2, _rhs) <- sharing rhs t1 tf
    (t1, _lhs) <- capsha lhs t0 tf label
    (t2, _rhs) <- capsha rhs t1 tf label
    (t2, ) <$> SHLO._AddOp _lhs _rhs _type
    where ranked = tensorType (Proxy :: Proxy (Tracer s t))
          label  = max (getLabel lhs) (getLabel rhs)


  lhs - rhs = Tracer label $ \ t0 (transformTruncate label -> tf) -> do 
    let _type = toAnyType $ applyTransform tf ranked
    (t1, _lhs) <- capsha lhs t0 tf label
    (t2, _rhs) <- capsha rhs t1 tf label
    (t2, ) <$> SHLO._SubtractOp _lhs _rhs _type
    where ranked = tensorType (Proxy :: Proxy (Tracer s t))
          label  = max (getLabel lhs) (getLabel rhs)
    

  lhs * rhs = Tracer label $ \ t0 (transformTruncate label -> tf) -> do 
    let _type = toAnyType $ applyTransform tf ranked
    (t1, _lhs) <- capsha lhs t0 tf label
    (t2, _rhs) <- capsha rhs t1 tf label
    (t2, ) <$> SHLO._MulOp _lhs _rhs _type
    where ranked = tensorType (Proxy :: Proxy (Tracer s t))
          label  = max (getLabel lhs) (getLabel rhs)

  signum operand = Tracer label $ \ t0 (transformTruncate label -> tf) -> do
    let _type = toAnyType $ applyTransform tf ranked
    (t1, _operand) <- sharing operand t0 tf 
    (t1, ) <$> SHLO._SignOp _operand _type
    where ranked = tensorType (Proxy :: Proxy (Tracer s t))
          label  = getLabel operand

  negate operand = Tracer label $ \ t0 (transformTruncate label -> tf) -> do 
    let _type = toAnyType $ applyTransform tf ranked
    (t1, _operand) <- sharing operand t0 tf 
    (t1, ) <$> SHLO._NegOp _operand _type
    where ranked = tensorType (Proxy :: Proxy (Tracer s t))
          label  = getLabel operand

  abs    operand = Tracer label $ \ t0 (transformTruncate label -> tf) -> do 
    let _type = toAnyType $ applyTransform tf ranked
    (t1, _operand) <- sharing operand t0 tf 
    (t1, ) <$> SHLO._AbsOp _operand _type
    where ranked = tensorType (Proxy :: Proxy (Tracer s t))
          label  = getLabel operand

  fromInteger (fromInteger -> value :: t) = Tracer 0 $ \ t0 _ -> do 
    let _type@(RankedTensorType _shape _ _) = ranked
    (t0, ) <$> SHLO._ConstantOp (denseSplatAttr _shape value) (toAnyType _type)
    where ranked = tensorType (Proxy :: Proxy (Tracer s t))

instance (T s t, Fractional t) => Fractional (Tracer s t) where
  lhs / rhs = Tracer label $ \ t0 (transformTruncate label -> tf) -> do 
    let _type = toAnyType $ applyTransform tf ranked
    (t1, _lhs) <- capsha lhs t0 tf label
    (t2, _rhs) <- capsha rhs t1 tf label
    (t2, ) <$> SHLO._DivOp _lhs _rhs _type
    where ranked = tensorType (Proxy :: Proxy (Tracer s t))
          label  = max (getLabel lhs) (getLabel rhs)

  fromRational (fromRational -> value :: t) = Tracer 0 $ \ t0 _ -> do 
    let _type@(RankedTensorType _shape _ _) = ranked
    (t0, ) <$> SHLO._ConstantOp (denseSplatAttr _shape value) (toAnyType _type)
    where ranked = tensorType (Proxy :: Proxy (Tracer s t))

instance TensorOp Tracer where
  broadcast :: forall t org map targ. (Broadcast org map targ, Tensorial t) => Tracer org t -> Proxy map -> Tracer targ t
  broadcast org p = Tracer label $ \ t0 (transformTruncate label -> tf) -> do 
    let _type = toAnyType $ applyTransform tf $ tensorType (Proxy :: Proxy (Tracer targ t))
        _map  = applyTransform tf $ getBroadcastMap p
    (t1, _org) <- sharing org t0 tf
    (t1, ) <$> SHLO._BroadcastInDimOp _map _org _type
    where label = getLabel org

  broadcast' :: forall org targ t. (Broadcast' org targ, Tensorial t) => Tracer org t -> Tracer targ t  
  broadcast' org = Tracer label $ \ t0 (transformTruncate label -> tf) -> do 
    let _type = toAnyType $ applyTransform tf $ tensorType (Proxy :: Proxy (Tracer targ t))
        _map  = applyTransform tf $ BroadcastMap $ take (fromIntegral orgRank) [targRank - orgRank..]
    (t1, _org) <- sharing org t0 tf
    (t1, ) <$> SHLO._BroadcastInDimOp _map _org _type 
      where orgRank              = fromInteger $ shapeRank (Proxy :: Proxy org)
            targRank             = fromInteger $ shapeRank (Proxy :: Proxy targ)
            label                = getLabel org

  prod :: forall l r p t. (TensorProductConstraint l r p, Tensorial t) => Tracer l t -> Tracer r t -> Tracer p t
  prod lhs rhs = Tracer label $ \ t0 (transformTruncate label -> tf) -> do 
    let _type = toAnyType         $ applyTransform tf       $ tensorType (Proxy :: Proxy (Tracer p t))
        _attr = applyTransform tf $ DotDimensionNumbersAttr { getBatchingDims = [], getContractingDims = [] }
    (t1, _lhs) <- capsha lhs t0 tf label
    (t2, _rhs) <- capsha rhs t1 tf label
    (t2, ) <$> SHLO._DotGeneralOp _attr Nothing _lhs _rhs _type -- This might not work
    where label = max (getLabel lhs) (getLabel rhs)


instance T s t => Traceable (Tracer s t) where
  trace' _ u = (fmap (fmap singleton) . (\x -> sharing u x Id), ([], [_type]))
    where _type = tensorType' (Proxy :: Proxy (Tracer s t))


instance (T s t, Traceable f) => Traceable (Tracer s t -> f) where 
  trace' i f = first (_type :) <$> trace' (i + 1) (f argn)
    where argn = Tracer 0 (\ a tf -> assert (tf == Id) $ (a, ) <$> blockArg i) 
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


class VMap f where 
  type Vectorized (i :: Nat) f = f' | f' -> f i
  vmap' :: forall (i :: Nat). KnownNat i => Word -> (Word -> f) -> Vectorized i f

instance T s t => VMap (Tracer s t) where
  type Vectorized i (Tracer s t) = Tracer (i ': s) t

  vmap' :: forall (i :: Nat). KnownNat i => Word -> (Word -> Tracer s t) -> Vectorized i (Tracer s t)
  vmap' terminatorLabel delayed = Tracer terminatorLabel $ \ tbl tf -> 
    capsha (delayed (terminatorLabel + 1)) tbl (V dim tf) (terminatorLabel + 1)
    where dim = fromInteger $ natVal (Proxy :: Proxy i)

instance (T s t, VMap f) => VMap (Tracer s t -> f) where
  type Vectorized i (Tracer s t -> f) = Tracer (i ': s) t -> Vectorized i f
  vmap' tl delayed arg = vmap' tl' delayed' 
    where tl'            = max tl (getLabel arg)
          delayed' label = 
            let arg' = Tracer label $ \ tbl tf -> 
                  case tf of 
                    V _ tf' -> sharing arg tbl tf' 
                    _       -> error "Unexpected transformation stack"
            in  delayed label arg'

vmap :: (VMap (a -> b), KnownNat i) => (a -> b) -> Vectorized i (a -> b)
vmap f = vmap' 0 (const f)
