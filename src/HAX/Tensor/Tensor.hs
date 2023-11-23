{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilyDependencies #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE DefaultSignatures #-}
module HAX.Tensor.Tensor where
import Prelude hiding (lookup, pred)

import HAX.Tensor.Tracer
import HAX.Tensor.Tensorial

import HAX.Jit
import HAX.PjRt
import HAX.PjRt.Plugin (ShapeInfo(..))
import HAX.PjRt.BufferType
import HAX.Utils

import Data.Proxy
import Data.Primitive hiding (newArray)
import Data.Kind (Type)
import Data.Bifunctor (Bifunctor(first))
import Data.Coerce (coerce)
import Data.Maybe (fromJust)

import Foreign
import Foreign.C (CIntPtr)

import MLIR

import GHC.IO.Unsafe (unsafePerformIO)
import GHC.TypeError
import GHC.IsList
import GHC.Generics
import Control.Exception (assert)


newtype Tensor (s :: Shape) a = Tensor { getTensorBuffer :: Buffer }
newtype Tensor' = Tensor' { getTensor'Buffer :: Buffer }

toTensor :: forall s t. T s t => Tensor' -> Maybe (Tensor s t)
toTensor (getTensor'Buffer -> buffer) = 
  if (shape == bufferDimensions buffer) && (_type == bufferElementType buffer) then 
    Just (Tensor buffer)
  else
    Nothing
  where shape = fromInteger <$> shapeVal (Proxy :: Proxy s)
        _type = pjrtBufferType (Proxy :: Proxy t)

toTensor' :: forall s t. T s t => Tensor' -> Tensor s t
toTensor' (getTensor'Buffer -> buffer) = 
  if (shape == bufferDimensions buffer) && (_type == bufferElementType buffer) then 
    Tensor buffer
  else
    error $ "Wrong shape and/or dtype (Correct: " ++ show (bufferDimensions buffer) ++ " " ++ show (bufferElementType buffer) ++ ", Incorrect: " ++ show shape ++ " " ++ show _type ++ ")"
  where shape = fromInteger <$> shapeVal (Proxy :: Proxy s)
        _type = pjrtBufferType (Proxy :: Proxy t)


withTensor' :: Tensor' -> (forall s t. T s t => Tensor s t -> a) -> a
withTensor' (Tensor' buffer) func = 
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

tensor'Shape :: Tensor' -> [Int]
tensor'Shape = fmap fromIntegral . bufferDimensions . getTensor'Buffer

tensor'Type  :: Tensor' -> BufferType
tensor'Type = bufferElementType . getTensor'Buffer

whatTensor :: Tensor' -> ([Int], BufferType)
whatTensor anytsr = (tensor'Shape anytsr, tensor'Type anytsr)

debugTensorShape :: Tensor s t -> [Int]
debugTensorShape = fmap fromIntegral . bufferDimensions . getTensorBuffer

getScalar :: Tensorial t => Tensor '[] t -> t
getScalar (Tensor b) = unsafePerformIO (toHaskell . (`indexByteArray` 0) <$> bufferToHostBuffer b)

-- Pretty print tensor
-- TODO: Fix, because this is just bad
--       consider using bytestring
instance T s t => Show (Tensor s t) where
  show (Tensor b) = unsafePerformIO $ do
    hostbuff <- bufferToHostBuffer b
    let elemCount = sizeofByteArray hostbuff `div` elemSize
        shape     = (\ (fromIntegral -> a) -> (a - 1, a - 1)) <$> shapeVal (Proxy :: Proxy s)
    return $ fst $ formater (elemCount - 1) shape hostbuff ""
    where elemSize = staticSizeOf (Proxy :: Proxy t)
          formater :: Int -> [(Int, Int)] -> ByteArray -> String -> (String, Int)
          formater offs [] buf s = 
            let a :: StorageType t = indexByteArray buf offs
            in  (showTensorial a ++ s, offs - 1)
          formater offs ((idx, ext):ies) buf s
            | idx == 0  =
              let (s', offs') = formater offs ies buf ((if idx == ext then ']' else ','):s)
              in  ('[':s', offs')
            | otherwise = 
              let c = if idx == ext then ']' else ','
                  (s', offs') = formater offs ies buf (c:s)
              in  formater offs' ((idx - 1, ext):ies) buf s'
instance Show Tensor' where
  show anytsr = withTensor' anytsr show

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


-- Implement a jit the convert from function of tracers to tensors 
instance ConvertOp Tensor where
  convert = jitT convert

instance TensorOp Tensor where
  assumeEqShape :: forall s s' t. (KnownShape s, KnownShape s') => Tensor s t -> Tensor s' t
  assumeEqShape = assert (shapeVal (Proxy :: Proxy s) == shapeVal (Proxy :: Proxy s')) coerce

  unsafeBroadcast operand dims = jitT (`unsafeBroadcast` dims) operand
  unsafeTranspose operand perm = jitT (`unsafeTranspose` perm) operand
  unsafeReshape = jitT unsafeReshape
  unsafeSlice operand slicing = jitT (`unsafeSlice` slicing) operand
  unsafeReverse operand dims = jitT (`unsafeReverse` dims) operand
  unsafeScatter input indices update uwd iwd sdtod ivd = jitT (\inp ind upd -> unsafeScatter inp ind upd uwd iwd sdtod ivd) input indices update
  unsafeGather operand start offsetAxes collapsedAxes startAxesMap idxVectorAxis sliceSizes = jitT (\op st -> unsafeGather op st offsetAxes collapsedAxes startAxesMap idxVectorAxis sliceSizes) operand start
  unsafeConcat d = jitT (unsafeConcat d)

  unsafePad t v p = jitT (\v' -> unsafePad t v' p) v

  splat a = unsafePerformIO $ tensorSplat defaultDevice a

  unsafeLinspace axis = jitT . unsafeLinspace axis

  unsafeDotGeneral lhs rhs attr = jitT (\ _lhs _rhs -> unsafeDotGeneral _lhs _rhs attr) lhs rhs

  unsafeReduceAdd operand axies = jitT (`unsafeReduceAdd` axies) operand
  unsafeReduceMul operand axies = jitT (`unsafeReduceMul` axies) operand

  unsafeIota i = jitT (unsafeIota i)
  unsafeConvolution = jitT unsafeConvolution

  unsafeMultiIota ds d = jitT $ unsafeMultiIota ds d

  branch = jitT branch
  select = jitT select

  unsafePairwiseAdd = jitT unsafePairwiseAdd
  unsafePairwiseSub = jitT unsafePairwiseSub
  unsafePairwiseMul = jitT unsafePairwiseMul
  unsafePairwiseDiv = jitT unsafePairwiseDiv

  unsafePairwiseAbs = jitT unsafePairwiseAbs
  unsafePairwiseNegate = jitT unsafePairwiseNegate
  unsafePairwiseSignum = jitT unsafePairwiseSignum
  unsafePairwiseSin = jitT unsafePairwiseSin
  unsafePairwiseCos = jitT unsafePairwiseCos
  unsafePairwiseTanh = jitT unsafePairwiseTanh
  unsafePairwiseExp = jitT unsafePairwiseExp
  unsafePairwiseLog = jitT unsafePairwiseLog

-- instance EqualOp Tensor where
  isEQ = jitT isEQ
  isNE = jitT isNE

  isGT = jitT isGT
  isGE = jitT isGE
  isLT = jitT isLT
  isLE = jitT isLE

  unsafeSplit   = jitT unsafeSplit
  unsafeSoftmax = jitT unsafeSoftmax

  unsafeArgmax  = jitT unsafeArgmax

-- Maybe merge Tensor and Tracer
class Jit f where
  type JitF f

  jit' :: CIntPtr -> [(Buffer, AnyType)] -> f -> JitF f
  default jit' :: (Jitter f, JitF f ~ JitT f) => CIntPtr -> [(Buffer, AnyType)] -> f -> JitF f
  jit' _ (unzip -> (args, ins)) !t = unsafePerformIO $ do 
    fst . reifier <$> loadedExecutableExecute1Await program args Nothing nout
    where (main, outs, reifier) = jitOut t
          nout = length outs
          program = compile (ins, main, outs)

instance forall (r :: Shape -> Type -> Type) (s :: Shape) t. Jitter (r s t) => Jit (r s t) where
  type JitF (r s t) = JitT (r s t)

instance Jitter [t] => Jit [t] where
  type JitF [t] = JitT [t]

instance Jitter (l <&> r) => Jit (l <&> r) where
  type JitF (l <&> r) = JitT (l <&> r)

instance Jitter (l, r) => Jit (l, r) where
  type JitF (l, r) = JitT (l, r)

instance (Jitter a, Jit b) => Jit (a -> b) where
  type JitF (a -> b) = JitT a -> JitF b
  jit' i args !f a = jit' i' args' f'
    where (i', t, a') = jitIn i a
          f'          = f t
          args'       = args ++ a'

-- Jit Nested Transformation
instance TypeError (Text "cannot jit this function") => JNT Tensor where
  fromTracer = undefined
  toTracer   = undefined

class GJitter t where
  type GJitT t :: k -> Type
  gJitIn :: CIntPtr -> GJitT t x -> (CIntPtr, t x, [(Buffer, AnyType)])
  gJitOut :: t x -> (StableCache Value -> BlockM (StableCache Value, [Value]), [AnyType], [Buffer] -> (GJitT t x, [Buffer]))

instance GJitter V1 where
  type GJitT V1 = V1
  gJitIn i a = (i, a, [])
  gJitOut a = (pure . (,[]), [], (a,))

instance (GJitter f, GJitter g) => GJitter (f :+: g) where
  type GJitT (f :+: g) = GJitT f :+: GJitT g
  gJitIn i (L1 f) = (i', L1 f', f'')
    where (i', f', f'') = gJitIn i f
  gJitIn i (R1 g) = (i', R1 g', g'')
    where (i', g', g'') = gJitIn i g
  gJitOut (L1 f) = (t, t', first L1 . t'')
    where (t, t', t'') = gJitOut f
  gJitOut (R1 g) = (t, t', first R1 . t'')
    where (t, t', t'') = gJitOut g

instance GJitter f => GJitter (M1 i t f) where
  type GJitT (M1 i t f) = M1 i t (GJitT f)
  gJitIn i (M1 t) = (i', M1 t', t'')
    where (i', t', t'') = gJitIn i t
  gJitOut (M1 f) = (i, i', first M1 . i'')
    where (i, i', i'') = gJitOut f

instance GJitter U1 where
  type GJitT U1 = U1
  gJitIn i a = (i, a, [])
  gJitOut a = (pure . (,[]), [], (a,))

instance (GJitter f, GJitter g) => GJitter (f :*: g) where
  type GJitT (f :*: g) = GJitT f :*: GJitT g
  gJitIn i (f :*: g) = (i'', f' :*: g', f'' ++ g'')
    where (i' , f', f'') = gJitIn i  f
          (i'', g', g'') = gJitIn i' g
  gJitOut (a :*: b) = (\tbl -> do 
    (tbl' , a') <- ac tbl 
    (tbl'', b') <- bc tbl' 
    return (tbl'', a' ++ b'), 
    at ++ bt, 
    \bs -> let (a', bs')  = ar bs
               (b', bs'') = br bs'
           in  (a' :*: b', bs''))
    where (ac, at, ar) = gJitOut a
          (bc, bt, br) = gJitOut b

instance Jitter t => GJitter (K1 i t) where
  type GJitT (K1 i t) = K1 i (JitT t)
  gJitIn i (K1 t) = (i', K1 t', t'')
    where (i', t', t'') = jitIn i t
  gJitOut (K1 t) = (i, i', first K1 . i'')
    where (i, i', i'') = jitOut t

class Jitter t where
  type JitT t
  jitIn :: CIntPtr -> JitT t -> (CIntPtr, t, [(Buffer, AnyType)])
  default jitIn :: (Generic t, Generic (JitT t), GJitter (Rep t), GJitT (Rep t) ~ Rep (JitT t)) => CIntPtr -> JitT t -> (CIntPtr, t, [(Buffer, AnyType)])
  jitIn i (from -> t) = (i', t', t'')
    where (i', to -> t', t'') = gJitIn i t

  jitOut :: t -> (StableCache Value -> BlockM (StableCache Value, [Value]), [AnyType], [Buffer] -> (JitT t, [Buffer]))
  default jitOut :: (Generic t, Generic (JitT t), GJitter (Rep t), Rep (JitT t) ~ GJitT (Rep t)) => t -> (StableCache Value -> BlockM (StableCache Value, [Value]), [AnyType], [Buffer] -> (JitT t, [Buffer]))
  jitOut (from -> t) = (i, i', first to . i'')
    where (i, i', i'') = gJitOut t

instance Jitter Integer where
  type JitT Integer = Integer
  jitIn = (, , [])
  jitOut i = (pure . (,[]), [], (i,))

instance Jitter Rational where
  type JitT Rational = Rational
  jitIn = (, , [])
  jitOut i = (pure . (,[]), [], (i,))

instance Jitter (Proxy a) where
  type JitT (Proxy a) = Proxy a
  jitIn = (, , [])
  jitOut i = (pure . (,[]), [], (i,))

instance Jitter Int where
  type JitT Int = Int
  jitIn = (, , [])
  jitOut i = (pure . (,[]), [], (i,))

instance Jitter Float where
  type JitT Float = Float
  jitIn = (, , [])
  jitOut i = (pure . (,[]), [], (i,))

instance Jitter Bool where
  type JitT Bool = Bool
  jitIn = (, , [])  
  jitOut i = (pure . (,[]), [], (i,))

instance Jitter (a -> b) where
  type JitT (a -> b) = TypeError (Text "jit only support first order function")
  jitIn = undefined
  jitOut = undefined

instance Jitter a => Jitter [a] where
  type JitT [a] = [JitT a]
instance (Jitter a, Jitter b) => Jitter (a, b) where
  type JitT (a, b) = (JitT a, JitT b)
instance (Jitter a, Jitter b) => Jitter (a <&> b) where
  type JitT (a <&> b) = JitT a <&> JitT b

instance (JNT r, T s t) => Jitter (r s t) where
  type JitT (r s t) = Tensor s t
  jitIn i (Tensor buffer) = (i + 1, fromTracer . Tracer $ \t -> (t, ) <$> blockArg i, [(buffer, tensorType' (Proxy :: Proxy (Tracer s t)))])
  jitOut (toTracer -> Tracer f) = (fmap (fmap (: [])) . f, [tensorType' (Proxy :: Proxy (Tracer s t))], 
    \case 
      []   -> error "Not enough output"
      a:as -> (Tensor a, as))


type family ReverseJit f = f' | f' -> f where
  ReverseJit (a -> b)     = ReverseJit a -> ReverseJit b
  ReverseJit [a]          = [ReverseJit a]
  ReverseJit (a, b)       = (ReverseJit a, ReverseJit b)
  ReverseJit (a <&> b)    = ReverseJit a <&> ReverseJit b
  ReverseJit (Tensor s t) = Tracer s t
  ReverseJit Int          = Int


-- TODO: Implement caching
jit :: Jit f => f -> JitF f
jit = jit' 0 []

jitT :: (Jit f, f ~ ReverseJit (JitF f)) => f -> JitF f
jitT = jit
