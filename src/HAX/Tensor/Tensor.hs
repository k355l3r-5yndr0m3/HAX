{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilyDependencies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE LambdaCase #-}
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

import Control.Exception (assert)

import Data.Proxy
import Data.Primitive hiding (newArray)
import Data.Kind (Type)
import Data.Bifunctor
import Data.Coerce (coerce)

import Foreign
import Foreign.C (CIntPtr)

import MLIR

import GHC.IO.Unsafe (unsafePerformIO)
import GHC.TypeError
import GHC.IsList
import GHC.Generics

import System.Mem (performMajorGC)

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
  unsafeScatter input indices _update uwd iwd sdtod ivd = jitT (\inp ind upd -> unsafeScatter inp ind upd uwd iwd sdtod ivd) input indices _update
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

  unsafeArgmax = jitT unsafeArgmax
  unsafeMultiDimArgmax axes = jitT (unsafeMultiDimArgmax axes)

-- instance EqualOp Tensor where
  isEQ = jitT isEQ
  isNE = jitT isNE

  isGT = jitT isGT
  isGE = jitT isGE
  isLT = jitT isLT
  isLE = jitT isLE

instance FusedOp Tensor where
  mse          = jitT mse
  mha          = jitT mha
  mean'        = jitT mean'
  relu         = jitT relu
  sigmoid      = jitT sigmoid
  softmax      = jitT . softmax
  softmax'     = jitT softmax'
  countTrue    = jitT countTrue
  leakyrelu    = jitT leakyrelu
  leakyrelu'   = jitT leakyrelu'
  crossEntropy = jitT crossEntropy

-- Jit Nested Transformation
instance TypeError (Text "cannot jit this function") => JNT Tensor where
  fromTracer = undefined
  toTracer   = undefined

data CacheHit g = CacheHit LoadedExecutable Int ([Buffer] -> JitT g)
newtype Cache  a b = Cache  { unCache  :: JitC  a b }
newtype Cache' a b = Cache' { unCache' :: JitC' a b }
class Jit'' t where
  type JitT' t :: k -> Type
  type JitC' t b :: Type
  jitIn'    :: (Cache' t b, [Buffer]) -> JitT' t p -> (b, [Buffer])
  jitCache'   :: (f -> (CIntPtr, [AnyType]) -> b) -> (t p -> f) -> (CIntPtr, [AnyType]) -> Cache' t b
  jitOut' :: t p -> (VarTable Value -> BlockM (VarTable Value, [Value]), [AnyType], [Buffer] -> (JitT' t p, [Buffer]))
instance Jit'' V1 where
  type JitT' V1   = V1
  type JitC' V1 b = b
  jitIn'  (Cache' q, args) _ = (q, args)
  jitCache' continue f = Cache' . continue (f undefined)
  jitOut' a = (pure . (,[]), [], (a,))
instance Jit'' U1 where
  type JitT' U1   = U1
  type JitC' U1 b = b
  jitIn' (Cache' q, args) _ = (q, args)
  jitCache' continue f = Cache' . continue (f U1)
  jitOut' a = (pure . (,[]), [], (a,))
instance Jit'' f => Jit'' (M1 i t f) where
  type JitT' (M1 i t f)   = M1 i t (JitT' f)
  type JitC' (M1 i t f) b = Cache' f b
  jitIn' (Cache' q, args) (M1 p) = jitIn' (q, args) p
  jitCache' continue f = Cache' . jitCache' continue (f . M1)
  jitOut' (M1 f) = let (i, i', i'') = jitOut' f in (i, i', first M1 . i'')
instance (Jit'' f, Jit'' g) => Jit'' (f :+: g) where
  type JitT' (f :+: g)   = JitT' f :+: JitT' g
  type JitC' (f :+: g) b = (Cache' f b, Cache' g b)
  jitIn' (Cache' (f, g), args) = \case L1 l -> jitIn' (f, args) l
                                       R1 r -> jitIn' (g, args) r
  jitCache' continue f state = Cache' (jitCache' continue (f . L1) state, jitCache' continue (f . R1) state)
  jitOut' = \case L1 f -> let (t, t', t'') = jitOut' f in (t, t', first L1 . t'')
                  R1 g -> let (t, t', t'') = jitOut' g in (t, t', first R1 . t'')
instance (Jit'' f, Jit'' g) => Jit'' (f :*: g) where
  type JitT' (f :*: g)   = JitT' f :*: JitT' g
  type JitC' (f :*: g) b = Cache' f (Cache' g b)
  jitIn' (Cache' q, args) (f :*: g) = jitIn' (jitIn' (q, args) f) g
  jitCache' continues f = let f' a b = f (a :*: b) in Cache' . jitCache' (jitCache' continues) f'
  jitOut' (a :*: b) = (\tbl -> do 
    (tbl' , a') <- ac tbl 
    (tbl'', b') <- bc tbl' 
    return (tbl'', a' ++ b'), 
    at ++ bt, \bs -> let (a', bs')  = ar bs
                         (b', bs'') = br bs' in  (a' :*: b', bs''))
    where (ac, at, ar) = jitOut' a
          (bc, bt, br) = jitOut' b
instance Jit' t => Jit'' (K1 i t) where
  type JitT' (K1 i t)   = K1 i (JitT t)
  type JitC' (K1 i t) b = Cache t b
  jitIn' (Cache' q, args) (K1 p) = jitIn (q, args) p
  jitCache' continue f = Cache' . jitCache continue (f . K1)
  jitOut' (K1 t) = let (i, i', i'') = jitOut t in (i, i', first K1 . i'')
class Jit' t where
  type JitT t
  type JitC t b 
  jitIn :: (Cache t b, [Buffer]) -> JitT t -> (b, [Buffer])
  default jitIn :: (Generic (JitT t), Rep (JitT t) ~ JitT' (Rep t), JitC t b ~ JitC' (Rep t) b, Jit'' (Rep t)) => (Cache t b, [Buffer]) -> JitT t -> (b, [Buffer])
  jitIn (coerce -> p :: Cache' (Rep t) b, args) (from -> i) = jitIn' (p, args) i

  jitCache :: (f -> (CIntPtr, [AnyType]) -> b) -> (t -> f) -> (CIntPtr, [AnyType]) -> Cache t b
  default jitCache :: (JitC t b ~ JitC' (Rep t) b, Jit'' (Rep t), Generic t) => (f -> (CIntPtr, [AnyType]) -> b) -> (t -> f) -> (CIntPtr, [AnyType]) -> Cache t b
  jitCache continue f = coerce . jitCache' continue (f . to)

  jitOut :: t -> (VarTable Value -> BlockM (VarTable Value, [Value]), [AnyType], [Buffer] -> (JitT t, [Buffer]))
  default jitOut :: (Generic t, Generic (JitT t), Jit'' (Rep t), Rep (JitT t) ~ JitT' (Rep t)) => t -> (VarTable Value -> BlockM (VarTable Value, [Value]), [AnyType], [Buffer] -> (JitT t, [Buffer]))
  jitOut (from -> t) = let (a, b, c) = jitOut' t in (a, b, first to . c) 
instance (JNT r, T s t) => Jit' (r s t) where
  type JitT (r s t)   = Tensor s t
  type JitC (r s t) b = b
  jitIn (Cache b, args) (Tensor arg) = (b, args++[arg])
  jitCache c f (i, t) = Cache $ c (f $ fromTracer . Tracer $ \x -> (x, ) <$> blockArg i) (i + 1, t++[tensorTypeOf (Biproxy :: Biproxy s t)])
  jitOut (toTracer -> Tracer f) = (fmap (fmap (: [])) . f, [tensorTypeOf (Biproxy :: Biproxy s t)], \case []   -> error "Not enough output"
                                                                                                          a:as -> (Tensor a, as))
instance (Jit' a, Jit' b) => Jit' (a, b) where
  type JitT (a, b)   = (JitT a, JitT b)
  type JitC (a, b) c = JitC' (Rep (a, b)) c
instance (Jit' a, Jit' b) => Jit' (a <&> b) where
  type JitT (a <&> b)   = (JitT a <&> JitT b)
  type JitC (a <&> b) c = JitC' (Rep (a <&> b)) c
instance Jit' a => Jit' [a] where
  type JitT [a]   = [JitT a]
  type JitC [a] b = JitC' (Rep [a]) b
instance Jit' Bool where
  type JitT Bool   = Bool
  type JitC Bool b = JitC' (Rep Bool) b
instance Jit' (Proxy a) where
  type JitT (Proxy a)   = Proxy a
  type JitC (Proxy a) b = JitC' (Rep (Proxy a)) b -- TODO: Maybe implement this manually 
instance Jit' Int  where 
  type JitT Int   = Int
  type JitC Int b = ([b], [b]) -- TODO: Do something better than linear search
  jitIn  (Cache (lt0, ge0), ins) idx 
    | idx >= 0  = (ge0 !! idx, ins)
    | otherwise = (lt0 !! (-idx - 1), ins)
  jitCache continue f state = Cache ([continue (f i) state | i <- [-1, -2]], [continue (f i) state | i <- [0..maxBound]])
  jitOut idx = (pure . (, []), [], (idx, ))
instance TypeError (Text "Jit can only be applied to first order function.") => Jit' (a -> b) where
  type JitT _   = TypeError (Text "Jit can only be applied to first order function.")
  type JitC _ _ = TypeError (Text "Jit can only be applied to first order function.")
  jitIn    = undefined
  jitCache   = undefined
  jitOut = undefined
type family JitE f = r | r -> f where
  JitE (a -> b) = Cache a (JitE b)
  JitE a        = CacheHit a
type family JitF f = r where
  JitF (a -> b) = JitT a -> JitF b
  JitF a        = JitT a
class Jit f where
  jit'    :: (JitE f, [Buffer]) -> JitF f
  default jit' :: (Jit' f, JitE f ~ CacheHit f, JitF f ~ JitT f) => (JitE f, [Buffer]) -> JitF f
  jit' (CacheHit executable coarity constructor, arguments) = unsafePerformIO $ constructor <$> do
    results <- loadedExecutableExecute1Await executable arguments Nothing coarity
    performMajorGC -- prevent the haskell from comsuming too much memory because the gc does not know how much memory the program own
    return results

  jit''   :: f -> (CIntPtr, [AnyType]) -> JitE f
  default jit'' :: (Jit' f, JitE f ~ CacheHit f, JitF f ~ JitT f) => f -> (CIntPtr, [AnyType]) -> JitE f
  jit'' f (_, inputs) = CacheHit executable (length outputs) (\a -> let (g, b) = reifier a in assert (null b) g)
    where (main, outputs, reifier) = jitOut f
          executable = compile (inputs, main, outputs)
instance (Jit' a, Jit b) => Jit (a -> b) where
  jit' cache = jit' . jitIn cache
  jit'' = jitCache jit''
instance (JNT r, T s t) => Jit (r s t) where
instance Jit' a => Jit [a] where
instance (Jit' a, Jit' b) => Jit (a, b) where
instance (Jit' a, Jit' b) => Jit (a <&> b) where

type family ReverseJit f = f' | f' -> f
type instance ReverseJit (a -> b)     = ReverseJit a -> ReverseJit b
type instance ReverseJit [a]          = [ReverseJit a]
type instance ReverseJit (a, b)       = (ReverseJit a, ReverseJit b)
type instance ReverseJit (a <&> b)    = ReverseJit a <&> ReverseJit b
type instance ReverseJit (Tensor s t) = Tracer s t
type instance ReverseJit Bool         = Bool
type instance ReverseJit Int          = Int
type instance ReverseJit (Proxy a)    = Proxy a

jit :: Jit f => f -> JitF f
jit = jit' . (, []) . (`jit''` (0, []))

jitT :: (Jit f, f ~ ReverseJit (JitF f)) => f -> JitF f
jitT = jit

