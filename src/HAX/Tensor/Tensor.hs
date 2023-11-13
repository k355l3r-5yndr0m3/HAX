{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilyDependencies #-}
module HAX.Tensor.Tensor where
import Prelude hiding (lookup, pred)

import HAX.Tensor.Tracer
import HAX.Tensor.Tensorial

import HAX.Jit hiding (jit)
import HAX.PjRt
import HAX.PjRt.Plugin (ShapeInfo(..))
import HAX.PjRt.BufferType
import HAX.Utils

import Data.Proxy
import Data.Primitive hiding (newArray)

import Foreign

import GHC.IO.Unsafe (unsafePerformIO)
import GHC.IsList

newtype Tensor (s :: Shape) a = Tensor { getTensorBuffer :: Buffer }
newtype AnyTsr = AnyTsr { getAnyTsrBuffer :: Buffer }

toTensor :: forall s t. T s t => AnyTsr -> Maybe (Tensor s t)
toTensor (getAnyTsrBuffer -> buffer) = 
  if (shape == bufferDimensions buffer) && (_type == bufferElementType buffer) then 
    Just (Tensor buffer)
  else
    Nothing
  where shape = fromInteger <$> shapeVal (Proxy :: Proxy s)
        _type = pjrtBufferType (Proxy :: Proxy t)

toTensor' :: forall s t. T s t => AnyTsr -> Tensor s t
toTensor' (getAnyTsrBuffer -> buffer) = 
  if (shape == bufferDimensions buffer) && (_type == bufferElementType buffer) then 
    Tensor buffer
  else
    error $ "Wrong shape and/or dtype (Actural: " ++ show shape ++ ")"
  where shape = fromInteger <$> shapeVal (Proxy :: Proxy s)
        _type = pjrtBufferType (Proxy :: Proxy t)


withAnyTsr :: AnyTsr -> (forall s t. T s t => Tensor s t -> a) -> a
withAnyTsr (AnyTsr buffer) func = 
  reifyShape shape $ \shape' -> 
    let elemtype = bufferElementType buffer 
    in  if elemtype == f32 then 
          func (same shape' (Proxy :: Proxy Float))
        else if elemtype == u8 then 
          func (same shape' (Proxy :: Proxy Word8))
        else if elemtype == pred then 
          func (same shape' (Proxy :: Proxy Bool))
        else if elemtype == s64 then
          func (same shape' (Proxy :: Proxy Int64))
        else 
          error "Unsupported tensor type"
  where shape = fromIntegral <$> bufferDimensions buffer
        same :: T s t => Proxy s -> Proxy t -> Tensor s t
        same _ _ = Tensor buffer

anyTsrShape :: AnyTsr -> [Int]
anyTsrShape = fmap fromIntegral . bufferDimensions . getAnyTsrBuffer

anyTsrType  :: AnyTsr -> BufferType
anyTsrType = bufferElementType . getAnyTsrBuffer

debugTensorShape :: Tensor s t -> [Int]
debugTensorShape = fmap fromIntegral . bufferDimensions . getTensorBuffer

getScalar :: Tensorial t => Tensor '[] t -> t
getScalar (Tensor b) = unsafePerformIO (toHaskell . (`indexByteArray` 0) <$> bufferToHostBuffer b)

-- Pretty print tensor
-- TODO: Fix, because this is just bad
--       consider using bytestring
instance (T s a, Show (StorageType a), Prim (StorageType a)) => Show (Tensor s a) where
  show (Tensor b) = unsafePerformIO $ do
    hostbuff <- bufferToHostBuffer b
    let elemCount = sizeofByteArray hostbuff `div` elemSize
        shape     = (\ (fromIntegral -> a) -> (a - 1, a - 1)) <$> shapeVal (Proxy :: Proxy s)
    return $ fst $ formater (elemCount - 1) shape hostbuff ""
    where elemSize = staticSizeOf (Proxy :: Proxy a)
          formater :: Int -> [(Int, Int)] -> ByteArray -> String -> (String, Int)
          formater offs [] buf s = 
            let a :: StorageType a = indexByteArray buf offs
            in  (show a ++ s, offs - 1)
          formater offs ((idx, ext):ies) buf s
            | idx == 0  =
              let (s', offs') = formater offs ies buf ((if idx == ext then ']' else ','):s)
              in  ('[':s', offs')
            | otherwise = 
              let c = if idx == ext then ']' else ','
                  (s', offs') = formater offs ies buf (c:s)
              in  formater offs' ((idx - 1, ext):ies) buf s'

instance (T s t, TensorLiteral s, [i] ~ Literal s t) => IsList (Tensor s t) where
  type Item (Tensor s t) = ListItem (Literal s t)

  fromList = tensorFromArray . fromTensorLiteral (Proxy :: Proxy s) (fromHaskell literalPad) fromHaskell
    where tensorFromArray a = unsafePerformIO $ do 
            buffer <- mallocArray $ length a
            pokeArray buffer a
            tensorFromHostBufferGC defaultDevice buffer 

tensorToPrimArray :: Tensor s t -> PrimArray (StorageType t)
tensorToPrimArray (Tensor buffer) = unsafePerformIO $ conv <$> bufferToHostBuffer buffer
  where conv (ByteArray a) = PrimArray a

tensorToHostBuffer :: Tensor s t -> IO (Int, Ptr (StorageType t))
tensorToHostBuffer (Tensor buffer) = bufferToHostBuffer' buffer

tensorFromHostBufferGC :: forall s t. T s t => Device -> Ptr (StorageType t) -> IO (Tensor s t)
tensorFromHostBufferGC device hostBuffer = Tensor <$>
  clientBufferFromHostBufferGC client hostBuffer (pjrtBufferType p) (Shape shape) device
  where p :: Proxy t = Proxy
        shape = fromIntegral <$> shapeVal (Proxy :: Proxy s)

tensorSplat :: forall s t. T s t => Device -> t -> IO (Tensor s t)
tensorSplat device a = do
  content <- mallocArray elemCount
  pokeArray content $ replicate elemCount $ fromHaskell a
  tensorFromHostBufferGC device content
  where elemCount = fromIntegral $ product $ shapeVal (Proxy :: Proxy s)

instance JitReify (Tensor s t) where
  jitReify (Annotated (a:as)) = (Tensor a, as)
  jitReify (Annotated [])     = error "Program did not produced enough outputs"

  jitUnreify (Annotated args) (Tensor buffer) = Annotated (args ++ [buffer])


type instance JitTransform (Tracer s t) = Tensor s t

-- Implement a jit the convert from function of tracers to tensors 
type family JitTracerTransform f
type instance JitTracerTransform (Tensor s t) = Tracer s t
type family JitTracer f where 
  JitTracer (a ->  b) = JitTracerTransform a -> JitTracer b
  JitTracer (a <&> b) = JitTracer a <&> JitTracer b
  JitTracer a         = JitTracerTransform a

-- NOTE: Putting NOINLINE pragma here decrease instances of 
--       repeated compilation, but does not elimenate all 
--       instances
{-# NOINLINE jit #-}
jit :: forall f f'. (Traceable f, f' ~ JitResult f, Jit f', f ~ JitTracer f') => f -> f'
jit f = jit' $! jitData
  where jitData = (Annotated [] :: Annotated [Buffer] f', compile f)

-- TODO: Solve repeated compilation
--       This is probably because the LoadedExecutable ref count to zero 
--       so it needs to be repeatedly recompiled
--       Possible solution, stableptr

instance ConvertOp Tensor where
  convert = jit convert

instance TensorOp Tensor where
  unsafeBroadcast operand dims = jit (`unsafeBroadcast` dims) operand
  unsafeTranspose operand perm = jit (`unsafeTranspose` perm) operand
  unsafeReshape = jit unsafeReshape
  unsafeSlice operand slicing = jit (`unsafeSlice` slicing) operand
  unsafeReverse operand dims = jit (`unsafeReverse` dims) operand
  unsafeScatter input indices update uwd iwd sdtod ivd = jit (\inp ind upd -> unsafeScatter inp ind upd uwd iwd sdtod ivd) input indices update
  unsafeGather operand start offsetAxes collapsedAxes startAxesMap idxVectorAxis sliceSizes = jit (\op st -> unsafeGather op st offsetAxes collapsedAxes startAxesMap idxVectorAxis sliceSizes) operand start
  unsafeConcat d = jit (unsafeConcat d)

  unsafePad t v p = jit (\v' -> unsafePad t v' p) v

  splat a = unsafePerformIO $ tensorSplat defaultDevice a

-- instance MathOp Tensor where
  unsafeLinspace axis = jit . unsafeLinspace axis

  unsafeDotGeneral lhs rhs attr = jit (\ _lhs _rhs -> unsafeDotGeneral _lhs _rhs attr) lhs rhs

  unsafeReduceAdd operand axies = jit (`unsafeReduceAdd` axies) operand
  unsafeReduceMul operand axies = jit (`unsafeReduceMul` axies) operand

  unsafeIota i = jit (unsafeIota i)
  unsafeConvolution = jit unsafeConvolution

  unsafeMultiIota ds d = jit $ unsafeMultiIota ds d

-- instance Tensorial t => SelectOp Tensor t where
  branch = jit branch
  select = jit select

  unsafePairwiseAdd = jit unsafePairwiseAdd
  unsafePairwiseSub = jit unsafePairwiseSub
  unsafePairwiseMul = jit unsafePairwiseMul
  unsafePairwiseDiv = jit unsafePairwiseDiv

  unsafePairwiseAbs = jit unsafePairwiseAbs
  unsafePairwiseNegate = jit unsafePairwiseNegate
  unsafePairwiseSignum = jit unsafePairwiseSignum
  unsafePairwiseSin = jit unsafePairwiseSin
  unsafePairwiseCos = jit unsafePairwiseCos
  unsafePairwiseTanh = jit unsafePairwiseTanh
  unsafePairwiseExp = jit unsafePairwiseExp
  unsafePairwiseLog = jit unsafePairwiseLog

-- instance EqualOp Tensor where
  isEQ = jit isEQ
  isNE = jit isNE

  isGT = jit isGT
  isGE = jit isGE
  isLT = jit isLT
  isLE = jit isLE

instance (Num t, T s t) => Num (Tensor s t) where
  (+) = unsafePairwiseAdd
  (-) = unsafePairwiseSub
  (*) = unsafePairwiseMul

  abs = unsafePairwiseAbs
  negate = unsafePairwiseNegate
  signum = unsafePairwiseSignum

  fromInteger = splat . fromInteger

instance (Fractional t, T s t) => Fractional (Tensor s t) where
  (/) = unsafePairwiseDiv
  fromRational = splat . fromRational

instance (Floating t, T s t) => Floating (Tensor s t) where
  pi = splat pi
  exp = unsafePairwiseExp
  log = unsafePairwiseLog
  sin = unsafePairwiseSin
  cos = unsafePairwiseCos
  tanh = unsafePairwiseTanh
