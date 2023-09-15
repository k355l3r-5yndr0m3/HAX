{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ViewPatterns #-}
module HAX.Tensor.Tracer where
import Prelude hiding (lookup)

import HAX.Tensor.Tensorial

import Control.Exception
import Control.Monad.Primitive

import Data.IntMap.Strict hiding (singleton, null, map, foldl)
import Data.List (singleton)
import Data.Proxy
import Data.Primitive

import Foreign

import GHC.StableName
import GHC.TypeLits
import GHC.IO.Unsafe

import MLIR

import qualified Stablehlo.Dialect.Stablehlo as SHLO
import Stablehlo.Dialect.Stablehlo.Attributes

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

instance T s t => TraceableElement (Tracer s t) where
  constructTracer i = (i + 1, Tracer $ \ a -> (a, ) <$> blockArg i, [_type])
    where _type = tensorType' (Proxy :: Proxy (Tracer s t))

  deconstructTracer u = (fmap (fmap singleton) . sharing' u, ([], [_type]))
    where _type = tensorType' (Proxy :: Proxy (Tracer s t))


newtype BroadcastMap = BroadcastMap [Word64]
getBroadcastMap :: KnownShape s => Proxy s -> BroadcastMap
getBroadcastMap = BroadcastMap . map fromInteger . shapeVal
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
getReduceDims = ReduceDims . map fromInteger . shapeVal
instance AttrGet ReduceDims where
  attrGet (ReduceDims dims) = 
    if null dims then 
      attrGet $ DenseIntOrFPElements (RankedTensorType [0] I64 NullAttr) dims
    else 
      attrGet $ DenseIntOrFPElements (VectorType [fromIntegral $ length dims] I64) dims
instance DenseIntOrFPElementsAttr ReduceDims
instance DenseIntElementsAttr ReduceDims

newtype TransposePerm = TransposePerm [Word64]
getTransposePerm :: KnownShape s => Proxy s -> TransposePerm 
getTransposePerm = TransposePerm . map fromInteger . shapeVal
instance AttrGet TransposePerm where
  attrGet (TransposePerm perm) = 
    if null perm then 
      attrGet $ DenseIntOrFPElements (RankedTensorType [0] I64 NullAttr) perm
    else
      attrGet $ DenseIntOrFPElements (VectorType [fromIntegral $ length perm] I64) perm
instance DenseIntOrFPElementsAttr TransposePerm
instance DenseIntElementsAttr TransposePerm

unsafeReduceTracer :: forall s0 s1 t. (T s0 t, T s1 t) => Tracer s0 t -> (Value -> Value -> AnyType -> BlockM Value) -> t -> [Integer] -> Tracer s1 t
unsafeReduceTracer operand body (splat -> initvalue :: Tracer '[] t) dims = mkTracer $ do 
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

instance Tensorial t => TensorOp Tracer t where
  unsafeBroadcast :: forall s0 s1. (T s0 t, T s1 t) => Tracer s0 t -> [Integer] -> Tracer s1 t
  unsafeBroadcast operand idxmap@(BroadcastMap . fmap fromInteger -> _map) = 
    assert correctness $ mkTracer $ do 
    _operand <- sharing operand
    retval $ SHLO._BroadcastInDimOp _map _operand _type
    where correctness :: Bool
          correctness = 
            let isUnique :: [Integer] -> Bool
                isUnique []     = True
                isUnique (a:as) = notElem a as && isUnique as
                src = shapeVal (Proxy :: Proxy s0)
                dst = shapeVal (Proxy :: Proxy s1)
            in  isUnique idxmap && src == fmap (dst !!) (fromInteger <$> idxmap)
          _type = tensorType' (Proxy :: Proxy (Tracer s1 t))

  unsafeReduceAdd operand = unsafeReduceTracer operand SHLO._AddOp 0
  unsafeReduceMul operand = unsafeReduceTracer operand SHLO._MulOp 1

  unsafeDotGeneral :: forall s0 s1 s2. (T s0 t, T s1 t, T s2 t) => Tracer s0 t -> Tracer s1 t -> DotDimensionNumbersAttr -> Tracer s2 t
  unsafeDotGeneral lhs rhs attr = mkTracer $ do 
    _lhs <- sharing lhs 
    _rhs <- sharing rhs 
    retval $ SHLO._DotGeneralOp attr Nothing _lhs _rhs _type
    where _type = tensorType' (Proxy :: Proxy (Tracer s2 t))

  unsafeTranspose :: forall s0 s1. (KnownShape s0, KnownShape s1) => Tracer s0 t -> [Integer] -> Tracer s1 t
  unsafeTranspose operand perm = assert correctness $ mkTracer $ do 
    _operand <- sharing operand 
    if perm == [0..fromIntegral $ length perm - 1] then -- degenerate case
      return _operand 
    else 
      retval $ SHLO._TransposeOp attr _operand _type
    where _type = tensorType' (Proxy :: Proxy (Tracer s1 t))
          correctness = 
            let uniqueness :: [Integer] -> Bool
                uniqueness []     = True
                uniqueness (a:as) = notElem a as && uniqueness as
                operandShape = shapeVal (Proxy :: Proxy s0)
                resultShape  = shapeVal (Proxy :: Proxy s1)
            in  uniqueness perm && resultShape == map ((operandShape !!) . fromInteger) perm
          attr = TransposePerm (fromInteger <$> perm)

  splat :: forall s. T s t => t -> Tracer s t
  splat value = mkTracer $ do 
    retval $ SHLO._ConstantOp (denseSplatAttr shape value) _type
    where _type = tensorType' (Proxy :: Proxy (Tracer s t))
          shape = fromInteger <$> shapeVal (Proxy :: Proxy s)
  
  -- TODO: Implement a better linspace
  linspace :: forall n. (KnownNat n, Fractional t, Enum t) => (t, t) -> Tracer '[n] t
  --linspace (a, b) = mkTracer $ do
  --  retval $ SHLO._ConstantOp attr $ toAnyType _type
  --  where step  = (b - a) / (fromInteger nelem - 1)
  --        nelem = natVal (Proxy :: Proxy n)
  --        _data = take (fromInteger nelem) [a,a + step..]
  --        primArrayToByteArray (PrimArray r) = ByteArray r
  --        attr  = DenseElementsRawBuffer _type $ primArrayToByteArray $ primArrayFromList _data
  --        _type = tensorType (Proxy :: Proxy (Tracer '[n] t))
  linspace (a, b) = mkTracer $ 
    retval $ SHLO._ConstantOp attr $ toAnyType _type 
    where _type = tensorType (Proxy :: Proxy (Tracer '[n] t))
          attr  = DenseElementsRawBuffer _type $ unsafePerformIO buf
          nelem = fromInteger $ natVal (Proxy :: Proxy n)
          buf = do 
            buffer :: MutablePrimArray RealWorld t <- newPrimArray nelem 
            populate buffer ((0, a), (nelem - 1, b))
            undefined
          populate :: MutablePrimArray RealWorld t -> ((Int, t), (Int, t)) -> IO ()
          populate array ((offset, bottom), (len, top)) 
            | len <= 0  = return ()
            | even len  = do 
              writePrimArray array (offset + len - 1) top
              populate array ((offset, bottom), (len - 1, top - stepsize))
            | otherwise = do 
              let center = len `div` 2
                  value  = (a + b) / 2
              writePrimArray array center value
              populate array ((offset, bottom), (center, value - stepsize))
              populate array ((center + 1, value + stepsize), (center, top))
          stepsize = (b - a) / (fromIntegral nelem - 1)

instance (T s t, Floating t) => Floating (Tracer s t) where
  pi = fromRational (toRational (pi :: Double))
  sin operand = mkTracer $ do 
    _operand <- sharing operand 
    retval $ SHLO._SineOp _operand _type
    where _type = tensorType' (Proxy :: Proxy (Tracer s t))






