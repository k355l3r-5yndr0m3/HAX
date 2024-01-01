{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE ImpredicativeTypes #-}
module HAX.Tensor.Tracer where
import Prelude hiding (lookup)

import HAX.Tensor.Tensorial
import HAX.Utils

import Control.Exception

import Data.Proxy
import Data.Primitive

import Foreign

import GHC.StableName

import MLIR

import qualified Stablehlo.Dialect.Stablehlo as SHLO
import Stablehlo.Dialect.Stablehlo.Attributes

import GHC.IsList
import Data.List (nub, sort, uncons, singleton)
import Control.Monad (forM)
import GHC.TypeLits (KnownNat)
import Data.Maybe (fromMaybe, fromJust)


newtype Tracer (s :: Shape) t = Tracer (VarTable Value -> BlockM (VarTable Value, Value))
newtype TracerM a = TracerM (VarTable Value -> BlockM (VarTable Value, a))
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

-- Some how the new jiting method produce hash collision !?
sharing' :: forall s t. T s t => Tracer s t -> VarTable Value -> BlockM (VarTable Value, Value)
sharing' tracer@(Tracer f) table = do 
  name <- blockRunIO (makeStableName $! tracer)
  case variableLookup name table of
    Just item -> return (table, item)
    Nothing   -> do 
      (table', item) <- f table
      return (variableInsert name item table', item)

sharing :: forall s t. T s t => Tracer s t -> TracerM Value 
sharing tracer = TracerM (sharing' tracer)

retval :: BlockM a -> TracerM a
retval v = TracerM $ \ table -> 
  (table, ) <$> v


-- ConvertOp
instance ConvertOp Tracer where
  convert :: forall s f g. (T s f, T s g) => Tracer s f -> Tracer s g
  convert operand = mkTracer $ do 
    _operand <- sharing operand 
    retval $ SHLO._ConvertOp _operand _type 
    where _type = tensorTypeOf (Biproxy :: Biproxy s g)

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
  where _type = tensorTypeOf (Biproxy :: Biproxy s1 t)
        _dims = ReduceDims (fromInteger <$> dims)
        scalar = tensorTypeOf (Biproxy :: Biproxy '[] t)

instance TensorOp Tracer where
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
          _type = tensorTypeOf (Biproxy :: Biproxy s1 t)

  unsafeTranspose :: forall s0 s1 t. (T s0 t, T s1 t) => Tracer s0 t -> [Integer] -> Tracer s1 t
  unsafeTranspose operand perm = assert correctness $ mkTracer $ do 
    _operand <- sharing operand 
    if perm == [0..fromIntegral $ length perm - 1] then -- degenerate case
      return _operand 
    else 
      retval $ SHLO._TransposeOp attr _operand _type
    where _type = tensorTypeOf (Biproxy :: Biproxy s1 t)
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
          _type = tensorTypeOf (Biproxy :: Biproxy s1 t)
          operandShape = shapeVal (Proxy :: Proxy s0)
          resultShape  = shapeVal (Proxy :: Proxy s1)

  unsafeSlice :: forall s0 s1 t. (T s0 t, T s1 t) => Tracer s0 t -> [(Integer, Integer, Integer)] -> Tracer s1 t
  unsafeSlice operand slicing = mkTracer $ do
    _operand <- sharing operand
    retval $ SHLO._SliceOp (mkVec starts) (mkVec limits) (mkVec strides) _operand _type
    where _type = tensorTypeOf (Biproxy :: Biproxy s1 t)
          (starts, limits, strides) = unzip3 slicing
          mkVec :: [Integer] -> DenseIntOrFPElements (VectorType IntegerType) [Int64]
          mkVec vec = DenseIntOrFPElements (VectorType [fromIntegral $ length vec] I64) (fromIntegral <$> vec)

  unsafePad :: forall s0 s1 t. (T s0 t, T s1 t) => t -> Tracer s0 t -> [(Integer, Integer, Integer)] -> Tracer s1 t
  unsafePad padval operand padding = mkTracer $ do
    _operand <- sharing operand 
    _value   <- sharing value
    retval $ SHLO._PadOp (mkVec lower) (mkVec higher) (mkVec interior) _operand _value _type
    where _type = tensorTypeOf (Biproxy :: Biproxy s1 t)
          (lower, higher, interior) = unzip3 padding
          mkVec :: [Integer] -> DenseIntOrFPElements (VectorType IntegerType) [Int64]
          mkVec vec = DenseIntOrFPElements (VectorType [fromIntegral $ length vec] I64) (fromIntegral <$> vec)
          value :: Tracer '[] t = splat padval

  unsafeReverse :: forall s0 t. T s0 t => Tracer s0 t -> [Integer] -> Tracer s0 t
  unsafeReverse operand dims = mkTracer $ do
    _operand <- sharing operand 
    retval $ SHLO._ReverseOp dims' _operand _type
    where _type = tensorTypeOf (Biproxy :: Biproxy s0 t)
          dims' = DenseIntOrFPElements (VectorType [fromIntegral $ length dims] I64) (fromInteger <$> dims :: [Int64])

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
    where _type      = tensorTypeOf (Biproxy :: Biproxy s0 t)
          scalarType = tensorTypeOf (Biproxy :: Biproxy '[] t)
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
          _type = tensorTypeOf (Biproxy :: Biproxy s2 t)

  unsafeConcat :: forall s0 s1 s2 t. (Tensorial t, KnownShape s0, KnownShape s1, KnownShape s2) => Integer -> Tracer s0 t -> Tracer s1 t -> Tracer s2 t
  unsafeConcat dims lhs rhs = assert (lhsRank == rhsRank && lhsOtherAxes == rhsOtherAxes) $ mkTracer $ do 
    _lhs <- sharing lhs 
    _rhs <- sharing rhs
    retval $ SHLO._ConcatenateOp (IntegerAttr I64 $ fromInteger dims) [_lhs, _rhs] _type
    where _type = tensorTypeOf (Biproxy :: Biproxy s2 t)
          lhsRank = shapeRank (Proxy :: Proxy s0)
          rhsRank = shapeRank (Proxy :: Proxy s1)
          remove n a = take n a ++ drop (n + 1) a
          lhsOtherAxes = remove (fromInteger dims) $ shapeVal (Proxy :: Proxy s0)
          rhsOtherAxes = remove (fromInteger dims) $ shapeVal (Proxy :: Proxy s1)

  unsafeReduceAdd operand = unsafeReduceTracer operand SHLO._AddOp 0
  unsafeReduceMul operand = unsafeReduceTracer operand SHLO._MulOp 1

  unsafeDotGeneral :: forall s0 s1 s2 t. (T s0 t, T s1 t, T s2 t) => Tracer s0 t -> Tracer s1 t -> DotDimensionNumbersAttr -> Tracer s2 t
  unsafeDotGeneral lhs rhs attr = mkTracer $ do 
    _lhs <- sharing lhs 
    _rhs <- sharing rhs 
    retval $ SHLO._DotGeneralOp attr Nothing _lhs _rhs _type
    where _type = tensorTypeOf (Biproxy :: Biproxy s2 t)

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
          _type = tensorTypeOf (Biproxy :: Biproxy s2 t)
          nothing :: Maybe (DenseIntOrFPElements (VectorType IntegerType) [Int64]) = Nothing
          rank0 = shapeRank (Proxy :: Proxy s0)
          rank1 = shapeRank (Proxy :: Proxy s1)
          rank2 = shapeRank (Proxy :: Proxy s2)

  unsafeIota :: forall s t. T s t => Int -> Tracer s t
  unsafeIota dims = assert (dims < rank) $ mkTracer $ do 
    retval $ SHLO._IotaOp (IntegerAttr I64 (fromIntegral dims)) _type
    where _type = tensorTypeOf (Biproxy :: Biproxy s t)
          rank  = fromInteger $ shapeRank (Proxy :: Proxy s)



-- instance Tensorial t => SelectOp Tracer t where
  branch :: forall s t. (T s t) => Tracer s t -> Tracer s t -> Tracer '[] Bool -> Tracer s t
  branch false true cond = mkTracer $ do 
    _false <- sharing false 
    _true  <- sharing true 
    _cond  <- sharing cond
    retval $ SHLO._SelectOp _cond _true _false _type
    where _type = tensorTypeOf (Biproxy :: Biproxy s t)

  select :: forall s t. (T s t) => Tracer s t -> Tracer s t -> Tracer s Bool -> Tracer s t
  select false true cond = mkTracer $ do 
    _false <- sharing false 
    _true  <- sharing true 
    _cond  <- sharing cond
    retval $ SHLO._SelectOp _cond _true _false _type
    where _type = tensorTypeOf (Biproxy :: Biproxy s t)


-- instance EqualOp Tracer where
  isEQ :: forall s t. T s t => Tracer s t -> Tracer s t -> Tracer s Bool
  isEQ lhs rhs = mkTracer $ do
    _lhs <- sharing lhs
    _rhs <- sharing rhs
    retval $ SHLO._CompareOp ComparisonDirectionEQ (Just ct) _lhs _rhs tt
    where ct = comparisonType (Proxy :: Proxy t)
          tt = tensorTypeOf (Biproxy :: Biproxy s Bool)
  isNE :: forall s t. T s t => Tracer s t -> Tracer s t -> Tracer s Bool
  isNE lhs rhs = mkTracer $ do
    _lhs <- sharing lhs
    _rhs <- sharing rhs
    retval $ SHLO._CompareOp ComparisonDirectionNE (Just ct) _lhs _rhs tt
    where ct = comparisonType (Proxy :: Proxy t)
          tt = tensorTypeOf (Biproxy :: Biproxy s Bool)

-- instance OrderOp Tracer where
  isGT :: forall s t. T s t => Tracer s t -> Tracer s t -> Tracer s Bool
  isGT lhs rhs = mkTracer $ do
    _lhs <- sharing lhs
    _rhs <- sharing rhs
    retval $ SHLO._CompareOp ComparisonDirectionGT (Just ctype) _lhs _rhs _type
    where _type = tensorTypeOf (Biproxy :: Biproxy s Bool)
          ctype = comparisonType (Proxy :: Proxy t)
  isGE :: forall s t. T s t => Tracer s t -> Tracer s t -> Tracer s Bool
  isGE lhs rhs = mkTracer $ do
    _lhs <- sharing lhs
    _rhs <- sharing rhs
    retval $ SHLO._CompareOp ComparisonDirectionGE (Just ctype) _lhs _rhs _type
    where _type = tensorTypeOf (Biproxy :: Biproxy s Bool)
          ctype = comparisonType (Proxy :: Proxy t)
  isLT :: forall s t. T s t => Tracer s t -> Tracer s t -> Tracer s Bool
  isLT lhs rhs = mkTracer $ do
    _lhs <- sharing lhs
    _rhs <- sharing rhs
    retval $ SHLO._CompareOp ComparisonDirectionLT (Just ctype) _lhs _rhs _type
    where _type = tensorTypeOf (Biproxy :: Biproxy s Bool)
          ctype = comparisonType (Proxy :: Proxy t)
  isLE :: forall s t. T s t => Tracer s t -> Tracer s t -> Tracer s Bool
  isLE lhs rhs = mkTracer $ do
    _lhs <- sharing lhs
    _rhs <- sharing rhs
    retval $ SHLO._CompareOp ComparisonDirectionLE (Just ctype) _lhs _rhs _type
    where _type = tensorTypeOf (Biproxy :: Biproxy s Bool)
          ctype = comparisonType (Proxy :: Proxy t)

  unsafePairwiseAdd :: forall s t. (T s t) => Tracer s t -> Tracer s t -> Tracer s t
  unsafePairwiseAdd lhs rhs = mkTracer $ do 
     _lhs <- sharing lhs
     _rhs <- sharing rhs
     retval $ SHLO._AddOp _lhs _rhs _type
     where _type = tensorTypeOf (Biproxy :: Biproxy s t)


  unsafePairwiseSub  :: forall s t. (T s t) => Tracer s t -> Tracer s t -> Tracer s t
  unsafePairwiseSub lhs rhs = mkTracer $ do 
     _lhs <- sharing lhs
     _rhs <- sharing rhs
     retval $ SHLO._SubtractOp _lhs _rhs _type
     where _type = tensorTypeOf (Biproxy :: Biproxy s t)
     
  unsafePairwiseMul  :: forall s t. (T s t) => Tracer s t -> Tracer s t -> Tracer s t
  unsafePairwiseMul lhs rhs = mkTracer $ do 
     _lhs <- sharing lhs
     _rhs <- sharing rhs
     retval $ SHLO._MulOp _lhs _rhs _type
     where _type = tensorTypeOf (Biproxy :: Biproxy s t)
   
  unsafePairwiseSignum :: forall s t. (T s t) => Tracer s t -> Tracer s t
  unsafePairwiseSignum operand = mkTracer $ do
     _operand <- sharing operand
     retval $ SHLO._SignOp _operand _type
     where _type = tensorTypeOf (Biproxy :: Biproxy s t)
 
  unsafePairwiseNegate :: forall s t. (T s t) => Tracer s t -> Tracer s t
  unsafePairwiseNegate operand = mkTracer $ do 
     _operand <- sharing operand 
     retval $ SHLO._NegOp _operand _type
     where _type = tensorTypeOf (Biproxy :: Biproxy s t)
 
  unsafePairwiseAbs    :: forall s t. (T s t) => Tracer s t -> Tracer s t
  unsafePairwiseAbs    operand = mkTracer $ do 
     _operand <- sharing operand
     retval $ SHLO._AbsOp _operand _type
     where _type = tensorTypeOf (Biproxy :: Biproxy s t)

  unsafePairwiseDiv  :: forall s t. (T s t) => Tracer s t -> Tracer s t -> Tracer s t
  unsafePairwiseDiv lhs rhs = mkTracer $ do 
     _lhs <- sharing lhs
     _rhs <- sharing rhs
     retval $ SHLO._DivOp _lhs _rhs _type
     where _type = tensorTypeOf (Biproxy :: Biproxy s t)

  unsafePairwiseSin :: forall s t. (T s t) => Tracer s t -> Tracer s t
  unsafePairwiseSin operand = mkTracer $ do 
    _operand <- sharing operand 
    retval $ SHLO._SineOp _operand _type
    where _type = tensorTypeOf (Biproxy :: Biproxy s t)
  unsafePairwiseCos :: forall s t. (T s t) => Tracer s t -> Tracer s t
  unsafePairwiseCos operand = mkTracer $ do 
    _operand <- sharing operand 
    retval $ SHLO._CosineOp _operand _type
    where _type = tensorTypeOf (Biproxy :: Biproxy s t)
  unsafePairwiseTanh :: forall s t. (T s t) => Tracer s t -> Tracer s t
  unsafePairwiseTanh operand = mkTracer $ do 
    _operand <- sharing operand 
    retval $ SHLO._TanhOp _operand _type
    where _type = tensorTypeOf (Biproxy :: Biproxy s t)

  unsafePairwiseExp    :: forall s t. (T s t) => Tracer s t -> Tracer s t
  unsafePairwiseExp operand = mkTracer $ do 
     _operand <- sharing operand 
     retval $ SHLO._ExpOp _operand _type 
     where _type = tensorTypeOf (Biproxy :: Biproxy s t)
  unsafePairwiseLog    :: forall s t. (T s t) => Tracer s t -> Tracer s t
  unsafePairwiseLog operand = mkTracer $ do 
     _operand <- sharing operand 
     retval $ SHLO._LogOp _operand _type 
     where _type = tensorTypeOf (Biproxy :: Biproxy s t)

  unsafeArgmax :: forall s s' t. (T s t, T s' t, Ord t) => Int -> Tracer s t -> Tracer s' Int64
  unsafeArgmax axis operand = mkTracer $ do 
    _operand <- sharing operand
    _indices <- sharing indices
    _startOp <- sharing startOp
    _startId <- sharing startId
    retval $ (!! 1) <$> SHLO._ReduceOp redims [_operand, _indices, _startOp, _startId] (do 
      blk0 <- blockGet [tensorTypeOf (Biproxy :: Biproxy '[] t), tensorTypeOf (Biproxy :: Biproxy '[] Int64),
                        tensorTypeOf (Biproxy :: Biproxy '[] t), tensorTypeOf (Biproxy :: Biproxy '[] Int64)]
      blockDef blk0 $ do 
        lhsVal <- blockArg 0
        lhsIdx <- blockArg 1
        rhsVal <- blockArg 2
        rhsIdx <- blockArg 3
        lhGTrh <- SHLO._CompareOp ComparisonDirectionGT Nothing lhsVal rhsVal $ tensorTypeOf (Biproxy :: Biproxy '[] Bool)
        lhEQrh <- SHLO._CompareOp ComparisonDirectionEQ Nothing lhsVal rhsVal $ tensorTypeOf (Biproxy :: Biproxy '[] Bool)
        val   <- SHLO._SelectOp lhGTrh lhsVal rhsVal $ tensorTypeOf (Biproxy :: Biproxy '[] t)
        idx'  <- SHLO._SelectOp lhGTrh lhsIdx rhsIdx $ tensorTypeOf (Biproxy :: Biproxy '[] Int64)
        idx'' <- SHLO._MaxOp lhsIdx rhsIdx $ tensorTypeOf (Biproxy :: Biproxy '[] Int64)
        idx   <- SHLO._SelectOp lhEQrh idx'' idx' $ tensorTypeOf (Biproxy :: Biproxy '[] Int64)
        SHLO._ReturnOp [val, idx]
        ) [tensorTypeOf (Biproxy :: Biproxy s' t), tensorTypeOf (Biproxy :: Biproxy s' Int64)]
    where indices :: Tracer  s  Int64 = unsafeIota $ fromIntegral axis
          startOp :: Tracer '[] t     = splat maxIdent
          startId :: Tracer '[] Int64 = splat (-1)
          redims = ReduceDims [fromIntegral axis]

  unsafeMultiDimArgmax :: forall s s' t. (Ord t, T s t, T s' t) => [Int] -> Tracer s t -> Tracer s' Int64
  unsafeMultiDimArgmax (nub -> reduceDims) operand = 
    assert (maximum reduceDims < fromInteger (shapeRank' @s)) $ 
      mkTracer $ do
        _operand <- sharing operand
        _iotas   <- forM iotas sharing
        _initValue <- sharing initValue
        _initIdx   <- sharing initIdx

        let reductionShape = trimIdx reduceDims operandShape
            typing shape = tentype' @t shape:replicate reduceDimNum (tentype' @Int64 shape)
        
        _argmax0 <- tail <$> retval (SHLO._ReduceOp rd (_operand:_iotas ++ _initValue:replicate reduceDimNum _initIdx) (do 
          blk0 <- blockGet (typing [] ++ typing [])
          blockDef blk0 $ do 
            (fromJust . uncons -> (lhs, lhsIdx), 
             fromJust . uncons -> (rhs, rhsIdx)) <- splitAt (reduceDimNum + 1) <$> forM [0..1 + fromIntegral reduceDimNum * 2] blockArg
            lhsGT  <- SHLO._CompareOp ComparisonDirectionGT Nothing lhs rhs (scaltype @Bool)
            lhsLT  <- SHLO._CompareOp ComparisonDirectionLT Nothing lhs rhs (scaltype @Bool)
            maxIdx0 <- forM (zip lhsIdx rhsIdx) $ \(l, r) -> SHLO._SelectOp lhsGT l r (scaltype @Int64)
            maxIdx1 <- forM (zip lhsIdx rhsIdx) $ \(l, r) -> SHLO._SelectOp lhsLT r l (scaltype @Int64)
            maxIdx2 <- forM (zip maxIdx0 maxIdx1) $ \(l, r) -> SHLO._MaxOp l r (scaltype @Int64)
            maxVal  <- SHLO._MaxOp lhs rhs (scaltype @t)
            SHLO._ReturnOp (maxVal:maxIdx2)) 
          (typing reductionShape))
        _argmax1 <- forM _argmax0 (\am -> retval (SHLO._ReshapeOp am (tentype' @Int64 (reductionShape ++ [1]))))
        assert (reductionShape ++ [fromIntegral reduceDimNum] == resultShape) $ 
          retval $ SHLO._ConcatenateOp (IntegerAttr I64 $ fromIntegral (length reductionShape)) _argmax1 (tentype @s' @Int64)
    where operandShape = fromInteger <$> shapeVal' @s
          resultShape  = fromInteger <$> shapeVal' @s'
          reduceDimNum = length reduceDims
          iotas :: [Tracer s Int64]   = fmap unsafeIota reduceDims 
          initIdx :: Tracer '[] Int64 = -1
          initValue :: Tracer '[] t   = splat maxIdent 
          rd = ReduceDims (fromIntegral <$> reduceDims)
          trimIdx :: [Int] -> [a] -> [a]
          trimIdx (sort -> idx) (zip [0..] -> list) = 
            let trim' _      []          = []
                trim' []     a           = snd <$> a
                trim' (i:is) ((j, a):as) | i == j    = trim' is as
                                         | otherwise = a:trim' (i:is) as
            in  trim' idx list


class JNT (r :: Z) where
  fromTracer :: Tracer s t -> r s t
  toTracer   :: r s t -> Tracer s t
instance JNT Tracer where
  fromTracer = id
  toTracer   = id
