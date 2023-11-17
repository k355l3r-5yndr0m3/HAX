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

import Foreign
import Foreign.C (CIntPtr)

import MLIR

import GHC.IO.Unsafe (unsafePerformIO)
import GHC.TypeError
import GHC.IsList
import GHC.Generics
import Data.Kind (Type)


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
    error $ "Wrong shape and/or dtype (Correct: " ++ show (bufferDimensions buffer) ++ " " ++ show (bufferElementType buffer) ++ ", Incorrect: " ++ show shape ++ " " ++ show _type ++ ")"
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

  unsafeSplit = jitT unsafeSplit


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

-- Maybe merge Tensor and Tracer
class Jit f where
  type JitF f

  jit' :: CIntPtr -> [(Buffer, AnyType)] -> f -> JitF f
  default jit' :: (JitOut f, JitF f ~ JitO f) => CIntPtr -> [(Buffer, AnyType)] -> f -> JitF f
  jit' _ (unzip -> (args, ins)) !t = unsafePerformIO $ do 
    program <- compile (ins, main, outs)
    fst . reifier <$> loadedExecutableExecute1Await program args Nothing nout
    where (main, outs, reifier) = jitOut t
          nout = length outs

instance forall r (s :: Shape) t. JitOut (r s t) => Jit (r s t) where
  type JitF (r s t) = JitO (r s t)

instance JitOut [t] => Jit [t] where
  type JitF [t] = JitO [t]

instance JitOut (l <&> r) => Jit (l <&> r) where
  type JitF (l <&> r) = JitO (l <&> r)

instance JitOut (l, r) => Jit (l, r) where
  type JitF (l, r) = JitO (l, r)

instance (JitIn a, Jit b) => Jit (a -> b) where
  type JitF (a -> b) = JitI a -> JitF b
  jit' i args !f a = jit' i' args' f'
    where (i', t, a') = jitIn i a
          f'          = f t
          args'       = args ++ a'

class GJitIn t where
  type GJitI t :: k -> Type
  gJitIn :: CIntPtr -> GJitI t x -> (CIntPtr, t x, [(Buffer, AnyType)])
instance GJitIn V1 where
  type GJitI V1 = V1
  gJitIn i a = (i, a, [])
instance GJitIn U1 where
  type GJitI U1 = U1
  gJitIn i a = (i, a, [])
instance (GJitIn f, GJitIn g) => GJitIn (f :+: g) where
  type GJitI (f :+: g) = GJitI f :+: GJitI g
  gJitIn i (L1 f) = (i', L1 f', f'')
    where (i', f', f'') = gJitIn i f
  gJitIn i (R1 g) = (i', R1 g', g'')
    where (i', g', g'') = gJitIn i g
instance (GJitIn f, GJitIn g) => GJitIn (f :*: g) where
  type GJitI (f :*: g) = GJitI f :*: GJitI g
  gJitIn i (f :*: g) = (i'', f' :*: g', f'' ++ g'')
    where (i' , f', f'') = gJitIn i  f
          (i'', g', g'') = gJitIn i' g
instance GJitIn f => GJitIn (M1 i t f) where
  type GJitI (M1 i t f) = M1 i t (GJitI f)
  gJitIn i (M1 t) = (i', M1 t', t'')
    where (i', t', t'') = gJitIn i t
instance JitIn t => GJitIn (K1 i t) where
  type GJitI (K1 i t) = K1 i (JitI t)
  gJitIn i (K1 t) = (i', K1 t', t'')
    where (i', t', t'') = jitIn i t

class JitIn t where
  type JitI t
  jitIn :: CIntPtr -> JitI t -> (CIntPtr, t, [(Buffer, AnyType)])
  default jitIn :: (Generic t, Generic (JitI t), GJitIn (Rep t), GJitI (Rep t) ~ Rep (JitI t)) => CIntPtr -> JitI t -> (CIntPtr, t, [(Buffer, AnyType)])
  jitIn i (from -> t) = (i', t', t'')
    where (i', to -> t', t'') = gJitIn i t

instance JitIn Integer where
  type JitI Integer = Integer
  jitIn = (, , [])
instance JitIn Rational where
  type JitI Rational = Rational
  jitIn = (, , [])
instance JitIn (Proxy a) where
  type JitI (Proxy a) = Proxy a
  jitIn = (, , [])
instance JitIn Int where
  type JitI Int = Int
  jitIn = (, , [])
instance JitIn Float where
  type JitI Float = Float
  jitIn = (, , [])
instance JitIn Bool where
  type JitI Bool = Bool
  jitIn = (, , [])  
instance JitIn (a -> b) where
  type JitI (a -> b) = TypeError (Text "jit only support first order function")
  jitIn = undefined

instance JitIn a => JitIn [a] where
  type JitI [a] = [JitI a]
  jitIn i []     = (i, [], [])
  jitIn i (a:as) = (i'', a':as', a'' ++ as'')
    where (i' , a' , a'' ) = jitIn i  a
          (i'', as', as'') = jitIn i' as
instance (JitIn a, JitIn b) => JitIn (a, b) where
  type JitI (a, b) = (JitI a, JitI b)
  jitIn i (a, b) = (i'', (a', b'), a'' ++ b'')
    where (i' , a', a'') = jitIn i  a
          (i'', b', b'') = jitIn i' b
instance (JitIn a, JitIn b) => JitIn (a <&> b) where
  type JitI (a <&> b) = JitI a <&> JitI b
  jitIn i (a :&: b) = (i'', a' :&: b', a'' ++ b'')
    where (i' , a', a'') = jitIn i  a
          (i'', b', b'') = jitIn i' b

instance T s t => JitIn (Tracer s t) where
  type JitI (Tracer s t) = Tensor s t
  jitIn i (Tensor buffer) = (i + 1, Tracer $ \t -> (t, ) <$> blockArg i, [(buffer, tensorType' (Proxy :: Proxy (Tracer s t)))])

class JitOut t where
  type JitO t
  jitOut :: t -> (StableNameHashTable Value -> BlockM (StableNameHashTable Value, [Value]), [AnyType], [Buffer] -> (JitO t, [Buffer]))
instance JitOut a => JitOut [a] where
  type JitO [a] = [JitO a]
  jitOut []     = (pure . (, []), [], ([], ))
  jitOut (t:ts) = (\tbl -> do 
    (tbl', t') <- tc tbl
    (tbl'', ts') <- tsc tbl' 
    return (tbl'', t'++ts'), tt++tst, \bs -> 
      let (t' , bs')  = tr bs
          (ts', bs'') = tsr bs'
      in  (t':ts', bs''))
    where (tc, tt, tr)    = jitOut t
          (tsc, tst, tsr) = jitOut ts
instance (JitOut a, JitOut b) => JitOut (a, b) where
  type JitO (a, b) = (JitO a, JitO b)
  jitOut (a, b) = (\tbl -> do 
    (tbl' , a') <- ac tbl 
    (tbl'', b') <- bc tbl' 
    return (tbl'', a' ++ b'), 
    at ++ bt, 
    \bs -> 
      let (a', bs')  = ar bs
          (b', bs'') = br bs'
      in  ((a', b'), bs''))
    where (ac, at, ar) = jitOut a
          (bc, bt, br) = jitOut b
instance (JitOut a, JitOut b) => JitOut (a <&> b) where
  type JitO (a <&> b) = (JitO a <&> JitO b)
  jitOut (a :&: b) = (\tbl -> do 
    (tbl' , a') <- ac tbl 
    (tbl'', b') <- bc tbl' 
    return (tbl'', a' ++ b'), 
    at ++ bt, 
    \bs -> 
      let (a', bs')  = ar bs
          (b', bs'') = br bs'
      in  (a' :&: b', bs''))
    where (ac, at, ar) = jitOut a
          (bc, bt, br) = jitOut b
instance T s t => JitOut (Tracer s t) where
  type JitO (Tracer s t) = Tensor s t
  jitOut (Tracer t) = (\tbl -> do 
    fmap (:[]) <$> t tbl, [tensorType' (Proxy :: Proxy (Tracer s t))], \case 
      []   -> error "Not enough output"
      a:as -> (Tensor a, as))
instance JitOut (Proxy a) where
  type JitO (Proxy a) = Proxy a
  jitOut _ = (pure . (, []), [], (Proxy, ))

type family ReverseJit f = f' | f' -> f where
  ReverseJit (a -> b)     = ReverseJit a -> ReverseJit b
  ReverseJit [a]          = [ReverseJit a]
  ReverseJit (a, b)       = (ReverseJit a, ReverseJit b)
  ReverseJit (a <&> b)    = ReverseJit a <&> ReverseJit b
  ReverseJit (Tensor s t) = Tracer s t


-- This new jit might cause problem because of the unsafePerformIO, NOINLINE might solve one instance in the test
-- Given how slow the test ran, I guess that recompilation occure every time jit is called
-- TODO: Implement cache
jit :: Jit f => f -> JitF f
jit = jit' 0 []

jitT :: (Jit f, f ~ ReverseJit (JitF f)) => f -> JitF f
jitT = jit

type Jitter p = (JitIn p, JitOut p)
