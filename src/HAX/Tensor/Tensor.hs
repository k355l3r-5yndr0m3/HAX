{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilyDependencies #-}
{-# LANGUAGE MagicHash #-}
module HAX.Tensor.Tensor where
import Prelude hiding (lookup)

import HAX.Tensor.Tracer
import HAX.Tensor.Tensorial

import HAX.Jit
import HAX.PjRt
import HAX.PjRt.Plugin (ShapeInfo(..))
import HAX.PjRt.HostBufferSemantics
import HAX.Utils

import Control.Exception

import Data.Proxy
import Data.Kind
import Data.Primitive

import GHC.IO.Unsafe (unsafePerformIO)
import GHC.IsList

import MLIR
import qualified Stablehlo.Dialect.Stablehlo as SHLO

newtype Tensor (s :: Shape) a = Tensor { getUnderlyingBuffer :: Buffer }
class Trace (t :: Shape -> Type -> Type) where
  auto :: forall s a. T s a => Tensor s a -> t s a

instance Trace Tensor where
  auto = id

instance Trace Tracer where
  auto :: forall s a. T s a => Tensor s a -> Tracer s a
  auto tensor = Tracer $ \ t0 -> do 
    buffer <- blockRunIO $ bufferToHostBuffer $ getUnderlyingBuffer tensor
    let _attr = denseElemsAttr shape buffer (Proxy :: Proxy a)
    (t0, ) <$> SHLO._ConstantOp _attr _type
    where _type = tensorType' (Proxy :: Proxy (Tracer s a))
          shape = fromInteger <$> shapeVal (Proxy :: Proxy s)

-- Pretty print tensor
instance (T s a, Show a, Prim a) => Show (Tensor s a) where
  show (Tensor b) = unsafePerformIO $ do
    hostbuff <- bufferToHostBuffer b
    let elemCount = sizeofByteArray hostbuff `div` elemSize
        shape     = (\ (fromIntegral -> a) -> (a - 1, a - 1)) <$> shapeVal (Proxy :: Proxy s)
    return $ fst $ formater (elemCount - 1) shape hostbuff ""
    where elemSize = staticSizeOf (Proxy :: Proxy a)
          formater :: Int -> [(Int, Int)] -> ByteArray -> String -> (String, Int)
          formater offs [] buf s = 
            let a :: a = indexByteArray buf offs
            in  (show a ++ s, offs - 1)
          formater offs ((idx, ext):ies) buf s
            | idx == 0  =
              let (s', offs') = formater offs ies buf (',':s)
              in  ('[':s', offs')
            | otherwise = 
              let c = if idx == ext then ']' else ','
                  (s', offs') = formater offs ies buf (c:s)
              in  formater offs' ((idx - 1, ext):ies) buf s'


resizeList :: Int -> a -> [a] -> [a]
resizeList len pad list
  | len > 0   = 
    case list of
      []     -> replicate len pad
      (a:as) -> a:resizeList (len - 1) pad as
  | otherwise = []

class T s t => ListToTensor s t where
  type Padding s t
  regularize :: Proxy s -> t -> [Padding s t] -> [Padding s t] 
  padding    :: Proxy s -> t -> Padding s t
  flatten    :: Proxy s -> [Padding s t] -> [t]

instance T '[r0] t => ListToTensor '[r0] t where
  type Padding '[r0] t = t
  regularize _ = resizeList n 
    where n = fromInteger $ shapeValHead (Proxy :: Proxy '[r0])
  padding _ i = i
  flatten _ = id

instance (T (a ': as ': ass) t, ListToTensor (as ': ass) t) => ListToTensor (a ': as ': ass) t where
  type Padding  (a ': as ': ass) t = [Padding (as ': ass) t]
  regularize p t l = resizeList n (padding p t) l'
    where n  = fromInteger $ shapeValHead (Proxy :: Proxy (a ': as ': ass))
          l' = fmap (regularize (Proxy :: Proxy (as ': ass)) t) l
  padding _ i = replicate n (padding (Proxy :: Proxy (as ': ass)) i)
    where n = fromInteger $ shapeValHead (Proxy :: Proxy (as ': ass))
  flatten _ = concatMap (flatten (Proxy :: Proxy (as ': ass)))

instance ListToTensor s t => IsList (Tensor s t) where
  type Item (Tensor s t) = Padding s t
  fromList l = unsafePerformIO $ tensorFromHostBuffer defaultDevice l'
    where p = Proxy :: Proxy s
          l' = primArrayFromList $ flatten p $ regularize p (nullElement :: t) l
  toList = error "TODO: Implement"

tensorFromHostBuffer :: forall s a. (KnownShape s, Tensorial a) => Device -> PrimArray a -> IO (Tensor s a)
tensorFromHostBuffer device (PrimArray buffer#) = Tensor <$> do 
  (e, b) <- clientBufferFromHostBuffer client (ByteArray buffer#) (pjrtBufferType p) (Shape shape) kImmutableOnlyDuringCall device
  eventAwait e
  return b
  where p = Proxy :: Proxy a
        shape = fromIntegral <$> shapeVal (Proxy :: Proxy s)

tensorUnity :: forall s t. T s t => Device -> IO (Tensor s t)
tensorUnity device = tensorFromHostBuffer device (primArrayFromList bufferData)
  where bufferData = replicate (fromIntegral $ product $ shapeVal (Proxy :: Proxy s)) unitElement
        
tensorNullity :: forall s t. T s t => Device -> IO (Tensor s t)
tensorNullity device = tensorFromHostBuffer device (primArrayFromList bufferData)
  where bufferData = replicate (fromIntegral $ product $ shapeVal (Proxy :: Proxy s)) nullElement

tensorSplat :: forall s t. T s t => Device -> t -> IO (Tensor s t)
tensorSplat device a = tensorFromHostBuffer device (primArrayFromList $ replicate elemCount a)
  where elemCount = fromIntegral $ product $ shapeVal (Proxy :: Proxy s)
  

type JitCacheTensor f = ([Buffer], LoadedExecutable, Annotated Int f)
type JitTensor f = (Jit Tensor f, JitCacheTensor f ~ JitCache Tensor f)
instance Jit Tensor (Proxy (Tracer s t)) where
  type JitResult Tensor (Proxy (Tracer s t)) = Proxy (Tensor s t)
  type JitCache  Tensor (Proxy (Tracer s t)) = JitCacheTensor (Proxy (Tracer s t))
  
  jit' _ = Proxy
  jitInit _ = error "jitInit was not given a function"

  jitReify l = (Proxy, l)

instance T s t => Jit Tensor (Tracer s t) where
  type JitResult Tensor (Tracer s t) = Tensor s t
  type JitCache  Tensor (Tracer s t) = JitCacheTensor (Tracer s t)

  jit' (argumentStack, executable, _) = assert (null excess) output
    where (output, excess) = (jitReify . unsafePerformIO) (loadedExecutableExecute1Await executable argumentStack Nothing 1)

  jitInit _ = error "jitInit was not given a function"

  jitReify (a:as) = (Tensor a, as)
  jitReify _      = error "Computation does not produce all demanded results"

instance (JitTensor a, JitTensor b) => Jit Tensor (a <+> b) where
  type JitResult Tensor (a <+> b) = JitResult Tensor a <+> JitResult Tensor b
  type JitCache  Tensor (a <+> b) = JitCacheTensor (a <+> b)

  jit' (argumentStack, executable, Annotated coarity) = assert (null excess) output
    where (output, excess) = (jitReify . unsafePerformIO) (loadedExecutableExecute1Await executable argumentStack Nothing coarity)
  
  jitInit _ = error "jitInit was not given a function"
  jitReify r0 = (a :+: b, r2)
    where (a, r1) = jitReify r0
          (b, r2) = jitReify r1
  

instance (T s t, JitTensor f) => Jit Tensor (Tracer s t -> f) where
  type JitResult Tensor (Tracer s t -> f) = Tensor s t -> JitResult Tensor f
  type JitCache  Tensor (Tracer s t -> f) = JitCacheTensor (Tracer s t -> f)

  jit' (argumentStack, executable, Annotated coarity) t = jit' (argumentStack++[getUnderlyingBuffer t], executable, Annotated coarity)

  jitInit f = ([], executable, Annotated coarity)
    where (coarity, executable) = unsafePerformIO $ compile f

  jitReify = error "Should not be possible"

instance (T s t, Num t) => Num (Tensor s t) where
  (+) = jit (+)
  (-) = jit (-)
  (*) = jit (*)

  signum = jit signum
  abs    = jit abs
  negate = jit negate

  fromInteger = error "This is problematic"

instance (KnownShape s, Tensorial t, Fractional t) => Fractional (Tensor s t) where
  (/) = jit (/)
  recip = jit recip

  fromRational = error "This is problematic"

instance Tensorial t => TensorOp Tensor t where
  unsafeBroadcast operand dims = jit (`unsafeBroadcast` dims) operand
  unsafeReduce operand body initvalue redims = jit (\ _operand -> unsafeReduce _operand body initvalue redims) operand
  unsafeDotGeneral lhs rhs attr = jit (\ _lhs _rhs -> unsafeDotGeneral _lhs _rhs attr) lhs rhs

  splat a = unsafePerformIO $ tensorSplat defaultDevice a

