{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilyDependencies #-}
module HAX.Tensor.Tensor where
import Prelude hiding (lookup)

import HAX.Tensor.Tracer
import HAX.Tensor.Tensorial

import HAX.Jit hiding (jit)
import HAX.PjRt
import HAX.PjRt.Plugin (ShapeInfo(..))
import HAX.Utils

import Data.Proxy
import Data.Primitive hiding (newArray)
import Data.Int

import Foreign

import GHC.IO.Unsafe (unsafePerformIO)
import GHC.IsList
import GHC.TypeLits

newtype Tensor (s :: Shape) a = Tensor { getUnderlyingBuffer :: Buffer }
debugTensorShape :: Tensor s t -> IO [Int64]
debugTensorShape = bufferDimensions . getUnderlyingBuffer

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
instance (T s t, Num t) => Num (Tensor s t) where
  (+) = jit (+)
  (-) = jit (-)
  (*) = jit (*)

  signum = jit signum
  abs    = jit abs
  negate = jit negate

  fromInteger = unsafePerformIO . tensorSplat defaultDevice . fromIntegral

instance (T s t, Fractional t) => Fractional (Tensor s t) where
  (/) = jit (/)
  recip = jit recip

  fromRational = unsafePerformIO . tensorSplat defaultDevice . fromRational

instance (T s t, Floating t) => Floating (Tensor s t) where
  pi = unsafePerformIO $ tensorSplat defaultDevice pi

  sin = jit sin
  cos = jit cos
  tan = jit tan

  tanh = jit tanh

  exp = jit exp
  log = jit log

instance Tensorial t => ShapeOp Tensor t where
  unsafeBroadcast operand dims = jit (`unsafeBroadcast` dims) operand
  unsafeTranspose operand perm = jit (`unsafeTranspose` perm) operand

  splat a = unsafePerformIO $ tensorSplat defaultDevice a

instance (Num t, Tensorial t) => MathOp Tensor t where
  linspace :: forall n. (KnownNat n, Fractional t, Enum t) => (t, t) -> Tensor '[n] t
  linspace = jit . linspace

  unsafeDotGeneral lhs rhs attr = jit (\ _lhs _rhs -> unsafeDotGeneral _lhs _rhs attr) lhs rhs

  unsafeReduceAdd operand axies = jit (`unsafeReduceAdd` axies) operand
  unsafeReduceMul operand axies = jit (`unsafeReduceMul` axies) operand

  unsafeIota i = jit (unsafeIota i)

instance Tensorial t => SelectOp Tensor t where
  branch = jit branch
  select = jit select

