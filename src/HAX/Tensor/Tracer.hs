{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ViewPatterns #-}
module HAX.Tensor.Tracer where
import Prelude hiding (lookup)

import HAX.Tensor.Tensorial

import Control.Exception

import Data.IntMap.Strict hiding (singleton, null, map, foldl)
import Data.List (singleton)
import Data.Proxy
import Data.Primitive

import Foreign

import GHC.StableName
import GHC.TypeLits

import MLIR

import qualified Stablehlo.Dialect.Stablehlo as SHLO
import Stablehlo.Dialect.Stablehlo.Attributes
import GHC.IsList

newtype Tracer (s :: Shape) t = Tracer (IntMap Value -> BlockM (IntMap Value, Value))
-- For tracing
-- TODO: Put somewhere else
instance T s t => TraceableElement (Tracer s t) where
  constructTracer i = (i + 1, Tracer $ \ a -> (a, ) <$> blockArg i, [_type])
    where _type = tensorType' (Proxy :: Proxy (Tracer s t))

  deconstructTracer u = (fmap (fmap singleton) . sharing' u, ([], [_type]))
    where _type = tensorType' (Proxy :: Proxy (Tracer s t))

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

instance (T s t, TensorLiteral s, [i] ~ Literal s t) => IsList (Tracer s t) where
  type Item (Tracer s t) = ListItem (Literal s t)
  fromList l = mkTracer $ retval $ elemsConstant shape value
    where shape = fromInteger <$> shapeVal (Proxy :: Proxy s)
          value = fromTensorLiteral (Proxy :: Proxy s) literalPad id l :: [t]
          


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
    retval $ splatConstant shape a
    where shape  = fromInteger <$> shapeVal (Proxy :: Proxy s)
          a :: t = fromInteger literal

instance (T s t, Fractional t) => Fractional (Tracer s t) where
  lhs / rhs = mkTracer $ do 
    _lhs <- sharing lhs
    _rhs <- sharing rhs
    retval $ SHLO._DivOp _lhs _rhs _type
    where _type = tensorType' (Proxy :: Proxy (Tracer s t))

  fromRational literal = mkTracer $ do 
    retval $ splatConstant shape a
    where shape  = fromIntegral <$> shapeVal (Proxy :: Proxy s)
          a :: t = fromRational literal


-- TODO: Implement more trig funcs
instance (T s t, Floating t) => Floating (Tracer s t) where
  pi = fromRational (toRational (pi :: Double))
  sin operand = mkTracer $ do 
    _operand <- sharing operand 
    retval $ SHLO._SineOp _operand _type
    where _type = tensorType' (Proxy :: Proxy (Tracer s t))
  cos operand = mkTracer $ do 
    _operand <- sharing operand 
    retval $ SHLO._CosineOp _operand _type
    where _type = tensorType' (Proxy :: Proxy (Tracer s t))

  tanh operand = mkTracer $ do 
    _operand <- sharing operand 
    retval $ SHLO._TanhOp _operand _type
    where _type = tensorType' (Proxy :: Proxy (Tracer s t))

  exp operand = mkTracer $ do 
    _operand <- sharing operand 
    retval $ SHLO._ExpOp _operand _type 
    where _type = tensorType' (Proxy :: Proxy (Tracer s t))

  log operand = mkTracer $ do 
    _operand <- sharing operand 
    retval $ SHLO._LogOp _operand _type 
    where _type = tensorType' (Proxy :: Proxy (Tracer s t))

-- ConvertOp
instance ConvertOp Tracer where
  convert :: forall s f g. (T s f, T s g) => Tracer s f -> Tracer s g
  convert operand = mkTracer $ do 
    _operand <- sharing operand 
    retval $ SHLO._ConvertOp _operand _type 
    where _type = tensorType' (Proxy :: Proxy (Tracer s g))

-- ShapeOp
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


instance ShapeOp Tracer where
  unsafeBroadcast :: forall s0 s1 t. (T s0 t, T s1 t) => Tracer s0 t -> [Integer] -> Tracer s1 t
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

  unsafeTranspose :: forall s0 s1 t. (T s0 t, T s1 t) => Tracer s0 t -> [Integer] -> Tracer s1 t
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

  unsafeReshape :: forall s0 s1 t. (T s0 t, T s1 t) => Tracer s0 t -> Tracer s1 t
  unsafeReshape operand = assert correctness $ mkTracer $ do
    _operand <- sharing operand
    if operandShape == resultShape then 
      return _operand 
    else
      retval $ SHLO._ReshapeOp _operand _type
    where correctness = product (shapeVal (Proxy :: Proxy s0)) == product (shapeVal (Proxy :: Proxy s1))
          _type = tensorType' (Proxy :: Proxy (Tracer s1 t))
          operandShape = shapeVal (Proxy :: Proxy s0)
          resultShape  = shapeVal (Proxy :: Proxy s1)

  unsafeSlice :: forall s0 s1 t. (T s0 t, T s1 t) => Tracer s0 t -> [(Integer, Integer, Integer)] -> Tracer s1 t
  unsafeSlice operand slicing = mkTracer $ do
    _operand <- sharing operand
    retval $ SHLO._SliceOp (mkVec starts) (mkVec limits) (mkVec strides) _operand _type
    where _type = tensorType' (Proxy :: Proxy (Tracer s1 t))
          (starts, limits, strides) = unzip3 slicing
          mkVec :: [Integer] -> DenseIntOrFPElements (VectorType IntegerType) [Int64]
          mkVec vec = DenseIntOrFPElements (VectorType [fromIntegral $ length vec] I64) (fromIntegral <$> vec)

  unsafePad :: forall s0 s1 t. (T s0 t, T s1 t) => t -> Tracer s0 t -> [(Integer, Integer, Integer)] -> Tracer s1 t
  unsafePad padval operand padding = mkTracer $ do
    _operand <- sharing operand 
    _value   <- sharing value
    retval $ SHLO._PadOp (mkVec lower) (mkVec higher) (mkVec interior) _operand _value _type
    where _type = tensorType' (Proxy :: Proxy (Tracer s1 t))
          (lower, higher, interior) = unzip3 padding
          mkVec :: [Integer] -> DenseIntOrFPElements (VectorType IntegerType) [Int64]
          mkVec vec = DenseIntOrFPElements (VectorType [fromIntegral $ length vec] I64) (fromIntegral <$> vec)
          value :: Tracer '[] t = splat padval

  unsafeReverse :: forall s0 t. T s0 t => Tracer s0 t -> [Integer] -> Tracer s0 t
  unsafeReverse operand dims = mkTracer $ do
    _operand <- sharing operand 
    retval $ SHLO._ReverseOp dims' _operand _type
    where _type = tensorType' (Proxy :: Proxy (Tracer s0 t))
          dims' = DenseIntOrFPElements (VectorType [fromIntegral $ length dims] SI64) (fromInteger <$> dims :: [Int64])

  splat :: forall s t. T s t => t -> Tracer s t
  splat value = mkTracer $ 
    retval $ splatConstant shape value 
    where shape = fromInteger <$> shapeVal (Proxy :: Proxy s)

  unsafeScatter :: forall s0 s1 s2 t. (T s0 t, T s1 t, T s2 t) => Tracer s0 t -> Tracer s1 Int64 -> Tracer s2 t -> [Integer] -> [Integer] -> [Integer] -> Integer -> Tracer s0 t
  unsafeScatter input sctIdx upd uwd iwd sdtod ivd = mkTracer $ do 
    _input <- sharing input
    _sctIdx <- sharing sctIdx
    _upd <- sharing upd
    retval $ head <$> SHLO._ScatterOp attr (Just $ BoolAttr False) (Just $ BoolAttr False) [_input, _sctIdx, _upd] (do 
      blk <- blockGet [scalarType, scalarType]
      blockDef blk $ do 
        _1 <- blockArg 1
        SHLO._ReturnOp [_1]) [_type]
    where _type      = tensorType' (Proxy :: Proxy (Tracer s0 t))
          scalarType = tensorType' (Proxy :: Proxy (Tracer '[] t)) 
          attr       = ScatterDimensionNumbersAttr {
                         updateWindow = fromInteger <$> uwd,
                         insertWindow = fromInteger <$> iwd,
                         scatterDimsToOperandDims = fromInteger <$> sdtod,
                         indexVectorDim = fromInteger ivd
                       }

  unsafeGather :: forall s0 s1 s2 t. (T s0 t, T s1 t, T s2 t) => Tracer s0 t -> Tracer s1 Int64 -> [Integer] -> [Integer] -> [Integer] -> Integer -> [Integer] -> Tracer s2 t
  unsafeGather operand starts offsetAxes collapsedAxes startAxisMap idxVectorAxis sliceSizes = mkTracer $ do 
    _operand <- sharing operand
    _starts  <- sharing starts
    retval $ SHLO._GatherOp dimensionNumbers sliceSizesAttr (Just $ BoolAttr False) _operand _starts _type
    where dimensionNumbers = GatherDimensionNumbersAttr { offsetDims = fromInteger <$> offsetAxes, 
                                                          collapsedSliceDims = fromInteger <$> collapsedAxes,
                                                          startIdxMap = fromInteger <$> startAxisMap,
                                                          idxVecDim = fromInteger idxVectorAxis
                                                        }
          sliceSizesAttr = DenseIntOrFPElements (RankedTensorType [fromIntegral $ length sliceSizes] I64 NullAttr) (fromInteger <$> sliceSizes :: [Int64])
          _type = tensorType' (Proxy :: Proxy (Tracer s2 t))

  unsafeConcat :: forall s0 s1 s2 t. (Tensorial t, KnownShape s0, KnownShape s1, KnownShape s2) => Integer -> Tracer s0 t -> Tracer s1 t -> Tracer s2 t
  unsafeConcat dims lhs rhs = assert (lhsRank == rhsRank && lhsOtherAxes == rhsOtherAxes) $ mkTracer $ do 
    _lhs <- sharing lhs 
    _rhs <- sharing rhs
    retval $ SHLO._ConcatenateOp (IntegerAttr I64 $ fromInteger dims) [_lhs, _rhs] _type
    where _type = tensorType' (Proxy :: Proxy (Tracer s2 t))
          lhsRank = shapeRank (Proxy :: Proxy s0)
          rhsRank = shapeRank (Proxy :: Proxy s1)
          remove n a = take n a ++ drop (n + 1) a
          lhsOtherAxes = remove (fromInteger dims) $ shapeVal (Proxy :: Proxy s0)
          rhsOtherAxes = remove (fromInteger dims) $ shapeVal (Proxy :: Proxy s1)


instance MathOp Tracer where
  -- TODO: Implement a better linspace
  linspace :: forall n t. (KnownNat n, Fractional t, Enum t, Tensorial t) => (t, t) -> Tracer '[n] t
  linspace (low, high) = splat step * unsafeIota 0
    where step = (high - low) / (n - 1)
          n    = fromInteger $ natVal (Proxy :: Proxy n)

  unsafeReduceAdd operand = unsafeReduceTracer operand SHLO._AddOp 0
  unsafeReduceMul operand = unsafeReduceTracer operand SHLO._MulOp 1

  unsafeDotGeneral :: forall s0 s1 s2 t. (T s0 t, T s1 t, T s2 t) => Tracer s0 t -> Tracer s1 t -> DotDimensionNumbersAttr -> Tracer s2 t
  unsafeDotGeneral lhs rhs attr = mkTracer $ do 
    _lhs <- sharing lhs 
    _rhs <- sharing rhs 
    retval $ SHLO._DotGeneralOp attr Nothing _lhs _rhs _type
    where _type = tensorType' (Proxy :: Proxy (Tracer s2 t))

  -- TODO: Add error detection
  unsafeConvolution :: forall s0 s1 s2 t. (T s0 t, T s1 t, T s2 t) => Tracer s0 t -> Tracer s1 t -> Tracer s2 t
  unsafeConvolution lhs rhs = mkTracer $ do 
    _lhs <- sharing lhs 
    _rhs <- sharing rhs
    retval $ SHLO._ConvolutionOp nothing nothing nothing nothing 
                                 (Nothing :: Maybe (DenseIntOrFPElements (VectorType IntegerType) Bool)) 
                                 attr (IntegerAttr I64 1) (IntegerAttr I64 1) Nothing _lhs _rhs _type
    where attr = ConvDimensionNumbersAttr {
            inputBatchDim = 0,
            inputFeatDim  = fromInteger rank0 - 1,
            inputSpatDims = [1..fromInteger rank0 - 2],
            
            kernelInputFeatDim  = 0,
            kernelOutputFeatDim = fromInteger rank1 - 1,
            kernelSpatDims      = [1..fromInteger rank1 - 2],
            
            outputBatchDim = 0,
            outputFeatDim  = fromInteger rank2 - 1,
            outputSpatDims = [1..fromInteger rank2 - 2]
          }
          _type = tensorType' (Proxy :: Proxy (Tracer s2 t))
          nothing :: Maybe (DenseIntOrFPElements (VectorType IntegerType) [Int64]) = Nothing
          rank0 = shapeRank (Proxy :: Proxy s0)
          rank1 = shapeRank (Proxy :: Proxy s1)
          rank2 = shapeRank (Proxy :: Proxy s2)

  unsafeIota :: forall s t. T s t => Integer -> Tracer s t
  unsafeIota dims = assert (dims < rank) $ mkTracer $ do 
    retval $ SHLO._IotaOp (IntegerAttr I64 (fromInteger dims)) _type
    where _type = tensorType' (Proxy :: Proxy (Tracer s t))
          rank  = shapeRank (Proxy :: Proxy s)



instance Tensorial t => SelectOp Tracer t where
  branch :: forall s. KnownShape s => Tracer s t -> Tracer s t -> Tracer '[] Bool -> Tracer s t
  branch false true cond = mkTracer $ do 
    _false <- sharing false 
    _true  <- sharing true 
    _cond  <- sharing cond
    retval $ SHLO._SelectOp _cond _true _false _type
    where _type = tensorType' (Proxy :: Proxy (Tracer s t))

  select :: forall s. KnownShape s => Tracer s t -> Tracer s t -> Tracer s Bool -> Tracer s t
  select false true cond = mkTracer $ do 
    _false <- sharing false 
    _true  <- sharing true 
    _cond  <- sharing cond
    retval $ SHLO._SelectOp _cond _true _false _type
    where _type = tensorType' (Proxy :: Proxy (Tracer s t))


instance EqualOp Tracer where
  isEQ :: forall s t. T s t => Tracer s t -> Tracer s t -> Tracer s Bool
  isEQ lhs rhs = mkTracer $ do
    _lhs <- sharing lhs
    _rhs <- sharing rhs
    retval $ SHLO._CompareOp ComparisonDirectionEQ (Just ct) _lhs _rhs tt
    where ct = comparisonType (Proxy :: Proxy t)
          tt = tensorType' (Proxy :: Proxy (Tracer s Bool))
  isNE :: forall s t. T s t => Tracer s t -> Tracer s t -> Tracer s Bool
  isNE lhs rhs = mkTracer $ do
    _lhs <- sharing lhs
    _rhs <- sharing rhs
    retval $ SHLO._CompareOp ComparisonDirectionNE (Just ct) _lhs _rhs tt
    where ct = comparisonType (Proxy :: Proxy t)
          tt = tensorType' (Proxy :: Proxy (Tracer s Bool))

instance OrderOp Tracer where
  isGT :: forall s t. T s t => Tracer s t -> Tracer s t -> Tracer s Bool
  isGT lhs rhs = mkTracer $ do
    _lhs <- sharing lhs
    _rhs <- sharing rhs
    retval $ SHLO._CompareOp ComparisonDirectionGT (Just ctype) _lhs _rhs _type
    where _type = tensorType' (Proxy :: Proxy (Tracer s Bool))
          ctype = comparisonType (Proxy :: Proxy t)
  isGE :: forall s t. T s t => Tracer s t -> Tracer s t -> Tracer s Bool
  isGE lhs rhs = mkTracer $ do
    _lhs <- sharing lhs
    _rhs <- sharing rhs
    retval $ SHLO._CompareOp ComparisonDirectionGE (Just ctype) _lhs _rhs _type
    where _type = tensorType' (Proxy :: Proxy (Tracer s Bool))
          ctype = comparisonType (Proxy :: Proxy t)
  isLT :: forall s t. T s t => Tracer s t -> Tracer s t -> Tracer s Bool
  isLT lhs rhs = mkTracer $ do
    _lhs <- sharing lhs
    _rhs <- sharing rhs
    retval $ SHLO._CompareOp ComparisonDirectionLT (Just ctype) _lhs _rhs _type
    where _type = tensorType' (Proxy :: Proxy (Tracer s Bool))
          ctype = comparisonType (Proxy :: Proxy t)
  isLE :: forall s t. T s t => Tracer s t -> Tracer s t -> Tracer s Bool
  isLE lhs rhs = mkTracer $ do
    _lhs <- sharing lhs
    _rhs <- sharing rhs
    retval $ SHLO._CompareOp ComparisonDirectionLE (Just ctype) _lhs _rhs _type
    where _type = tensorType' (Proxy :: Proxy (Tracer s Bool))
          ctype = comparisonType (Proxy :: Proxy t)
